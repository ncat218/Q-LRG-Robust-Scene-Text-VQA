"""Matched full/crop S1b factorial, on frozen Qwen and frozen A2 readers."""

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from filelock import FileLock

from qlrg.common import object_hash, read_json, sha256, write_json
from qlrg.evaluation import SCORER_REVISION, evaluator, paired_bootstrap
from qlrg.experiment import paired_images
from qlrg.s1_run import make_adapter
from scripts.compute_cost import count_flops
from scripts.crop_fusion import FusionQwen, fit_token_canvas
from scripts.crop_restore_followup import load_boxes
from scripts.followup_common import bind, finish, load_study, write_table


READERS = ('A0', 'A2')
BUDGETS = ('fair576', 'native')
LAYOUTS = ('full', 'crop', 'fusion')


def specifications(readers=READERS):
    return [(reader, budget, layout, flags) for reader in readers for budget in BUDGETS
            for layout in LAYOUTS for flags in ((False,), (True,)) if layout != 'fusion'] + [
        (reader, budget, 'fusion', flags) for reader in readers for budget in BUDGETS
        for flags in ((False, False), (True, False), (False, True), (True, True))]


def arm_name(reader, budget, layout, flags):
    return '__'.join((reader, budget, layout, ''.join(str(int(f)) for f in flags)))


def view_plan(image, box, layout, budget, flags):
    names = ('full', 'crop') if layout == 'fusion' else (layout,)
    # A failed crop-only request uses the full image, with its requested restoration flag.
    # Fusion fallback uses full only; a crop-only restoration flag has no effect there.
    effective = names if box is not None or layout == 'full' else ('full',)
    active = flags if len(effective) == len(names) else (flags[0],)
    views = [image.crop(box) if name == 'crop' else image for name in effective]
    tokens = (576 // len(views)) if budget == 'fair576' else None
    if tokens is not None:
        views = [fit_token_canvas(v, tokens)[0] for v in views]
    return views, effective, active, tokens


def comparison_pairs(readers):
    pairs = []
    for reader in readers:
        for budget in BUDGETS:
            name = lambda layout, flags: arm_name(reader, budget, layout, flags)
            baseline = name('full', (False,))
            names = [name(layout, flags) for r, b, layout, flags in specifications([reader]) if b == budget]
            pairs.extend((baseline, n) for n in names if n != baseline)
            pairs.extend([
                (name('crop', (False,)), name('crop', (True,))),
                (name('fusion', (False, False)), name('fusion', (True, False))),
                (name('fusion', (False, False)), name('fusion', (False, True))),
                (name('fusion', (False, False)), name('fusion', (True, True))),
                (name('fusion', (True, False)), name('fusion', (True, True))),
                (name('fusion', (False, True)), name('fusion', (True, True))),
            ])
    if set(readers) == set(READERS):
        pairs.extend((arm_name('A0', b, l, f), arm_name('A2', b, l, f))
                     for r, b, l, f in specifications(['A0']))
    return sorted(set(pairs))


@torch.inference_mode()
def predict_image(model, restorer, lora, scorer, image, proposals, root, readers, gates, profile_flops=False):
    _, lr4 = paired_images(root, image)
    rows, measured, features = [], [], {}
    specs = specifications(readers)
    # Alternate reader order by image to avoid assigning all warm-up effects to A0.
    if int(object_hash(image['image_id'])[:8], 16) % 2:
        specs = list(reversed(specs))
    for question in image['questions']:
        qid = question['question_id']
        proposal = proposals[(image['image_id'], qid)]
        box = proposal['box']
        if box is not None and not (0 <= box[0] < box[2] <= lr4.width and 0 <= box[1] < box[3] <= lr4.height):
            raise ValueError(f'Invalid source crop: {qid}')
        refs = scorer._compute_answer_scores(question['answers'])
        for reader, budget, layout, flags in specs:
            views, names, active, tokens = view_plan(lr4, box, layout, budget, flags)
            batch, grids = model.prepare_views(views, question['question'], exact_budget=tokens is not None)
            # Vision is frozen; reuse full features within one image and crop features within one query.
            key = (budget, names, tuple(tuple(g) for g in grids), None if names == ('full',) else qid)
            if key not in features:
                torch.cuda.synchronize(); started = time.perf_counter()
                base = tuple(model.encode_views(batch))
                torch.cuda.synchronize(); vision_seconds = time.perf_counter() - started
                restored, restore_times = [], []
                for feature, grid in zip(base, grids):
                    torch.cuda.synchronize(); started = time.perf_counter()
                    value = restorer(feature, grid)[0].to(model.dtype)
                    if value.shape != feature.shape or not torch.isfinite(value).all():
                        raise RuntimeError('Invalid restored features')
                    torch.cuda.synchronize()
                    restored.append(value); restore_times.append(time.perf_counter() - started)
                features[key] = (base, tuple(restored), vision_seconds, restore_times)
            base, restored, vision_seconds, restore_times = features[key]
            selected = tuple(r if enabled else f for f, r, enabled in zip(base, restored, active))
            context = lora.attached() if reader == 'A2' else nullcontext()
            with context:
                gate_key = '__'.join((reader, budget, layout))
                if gate_key not in gates:
                    if not torch.equal(model.prefill_logits(batch), model.prefill_logits(batch, base)):
                        raise RuntimeError(f'Identity injection failed: {gate_key}')
                    gates[gate_key] = {'question_id': qid, 'logits_exact': True}
                torch.cuda.reset_peak_memory_stats()
                result = model.generate(batch, selected)
                peak = torch.cuda.max_memory_allocated()
            restore_seconds = sum(t for t, enabled in zip(restore_times, active) if enabled)
            selector_seconds = proposal['selector_seconds'] if layout != 'full' else 0.0
            reader_seconds = vision_seconds + restore_seconds + result['generation_seconds']
            row = {**result, 'arm': arm_name(reader, budget, layout, flags), 'reader': reader,
                   'budget': budget, 'layout': layout, 'requested_restore': list(flags),
                   'effective_views': list(names), 'effective_restore': list(active),
                   'image_id': image['image_id'], 'question_id': qid,
                   'score': refs.get(scorer.answer_processor(result['answer']), 0.0),
                   'crop_xyxy': box if layout != 'full' else None,
                   'textocr_retained': proposal.get('textocr_retained') if layout != 'full' else None,
                   'fallback': box is None and layout != 'full',
                   'view_tokens': [int(h*w) for h,w in grids], 'visual_tokens': sum(int(h*w) for h,w in grids),
                   'vision_seconds': vision_seconds, 'restore_seconds': restore_seconds,
                   'reader_seconds': reader_seconds, 'selector_seconds': selector_seconds,
                   'pipeline_seconds': reader_seconds + selector_seconds, 'peak_cuda_bytes': peak}
            rows.append(row)
            if profile_flops and question == image['questions'][0]:
                # A separate uncached pass. Eight forced decode steps, independent of answer length.
                def operation():
                    fresh = tuple(model.encode_views(batch))
                    values = tuple(restorer(f, g)[0].to(model.dtype) if enabled else f
                                   for f,g,enabled in zip(fresh, grids, active))
                    attach = lora.attached() if reader == 'A2' else nullcontext()
                    with attach, model.replacement(batch, values):
                        return model.model.generate(**batch, do_sample=False, num_beams=1, use_cache=True,
                                                    min_new_tokens=8, max_new_tokens=8,
                                                    repetition_penalty=model.config['repetition_penalty'])
                _, flops = count_flops(operation)
                measured.append({'arm': row['arm'], 'image_id': image['image_id'],
                                 'registered_reader_flops': flops, 'fixed_decode_tokens': 8})
        # Do not retain question-specific tensors across queries.
        features = {k:v for k,v in features.items() if k[1] == ('full',)}
    return {'rows': rows, 'identity': gates, 'flops': measured}


def report(output, rows, manifest, readers, gates, flops):
    cases = [(im['image_id'], q['question_id']) for im in manifest['images'] for q in im['questions']]
    groups = [iid for iid, _ in cases]
    metrics, scores, indexed = [], {}, {}
    for spec in specifications(readers):
        name = arm_name(*spec)
        values = [r for r in rows if r['arm'] == name]
        if [(r['image_id'],r['question_id']) for r in values] != cases:
            raise ValueError(f'Missing, duplicate or different evaluation questions: {name}')
        if spec[1] == 'fair576' and any(r['visual_tokens'] != 576 for r in values):
            raise ValueError('Fair token budget changed')
        indexed[name] = values
        scores[name] = np.asarray([r['score'] for r in values])
        cost = [r['registered_reader_flops'] for r in flops if r['arm'] == name]
        row = {'arm': name, 'questions': len(cases), 'accuracy_pct': 100*float(scores[name].mean()),
               'visual_tokens_mean': mean(r['visual_tokens'] for r in values),
               'visual_tokens_max': max(r['visual_tokens'] for r in values),
               'fallbacks': sum(r['fallback'] for r in values),
               'peak_cuda_bytes': max(r['peak_cuda_bytes'] for r in values),
               'profiled_images': len(cost), 'registered_reader_flops_decode8_mean': mean(cost) if cost else None}
        for component in ('vision','restore','reader','pipeline'):
            times = [1000*r[component+'_seconds'] for r in values]
            row[component+'_p50_ms'] = float(np.median(times))
            row[component+'_p95_ms'] = float(np.percentile(times,95))
        metrics.append(row)
    comparisons, errors = [], []
    for left,right in comparison_pairs(readers):
        a,b=scores[left],scores[right]
        comparisons.append({'reference':left,'target':right,'questions':len(cases),
                            **paired_bootstrap(a,b,groups,10000,42),
                            'score_improved':int(sum(b>a)), 'score_worsened':int(sum(b<a))})
        # Primary same-layout toggle errors; no automatic causal labels.
        if left.split('__')[:3] == right.split('__')[:3]:
            for x,y in zip(indexed[left],indexed[right]):
                if x['score'] == y['score']:continue
                errors.append({'reference':left,'target':right,'image_id':x['image_id'],
                               'question_id':x['question_id'],'reference_answer':x['answer'],
                               'target_answer':y['answer'],'reference_score':x['score'],
                               'target_score':y['score'],'cause':'unreviewed'})
    write_table(output/'accuracy.csv',metrics)
    write_table(output/'paired.csv',comparisons)
    if errors:write_table(output/'error_cases.csv',errors)
    if flops:write_table(output/'flops.csv',flops)
    return {'split':manifest['split'],'condition':'LR4','questions':len(cases),'images':len(manifest['images']),
            'accuracy':metrics,'comparisons':comparisons,'identity':gates,
            'bootstrap_unit':'image_id','bootstrap_resamples':10000,'scorer_revision':SCORER_REVISION,
            'timing_policy':'Reused frozen vision/restore timings; selector times replayed from archived proposals. Not fresh end-to-end latency.',
            'flops_policy':'Registered reader operators including vision and restoration, fixed decode8; unsupported operators and selector FLOPs excluded.'}


def run(output):
    with FileLock(str(output/'.run.lock'), timeout=0):
        launch,paths,config,manifest,checkpoints,binding=load_study(output,'crop_matrix',('S1b','A2'))
        source=Path(launch['selector_benchmark'])
        proposals,archived=load_boxes(source,binding['manifest_hash'],checkpoints['A2']['provenance'])
        proposals=proposals['visionrl2_7b']
        readers=launch['readers']
        if not readers or len(set(readers)) != len(readers) or not set(readers)<=set(READERS):
            raise ValueError('Invalid readers')
        if launch.get('smoke_images'):
            manifest={**manifest,'images':manifest['images'][:launch['smoke_images']]}
        binding.update(readers=readers,arms=[arm_name(*s) for s in specifications(readers)],
                       smoke_images=launch.get('smoke_images'),selector_binding=object_hash(archived),
                       worker_sha256=sha256(Path(__file__)),flops_images=launch.get('flops_images',3))
        bind(output,binding)
        torch.manual_seed(42)
        model=FusionQwen(config,paths['hf_hub'])
        torch.backends.cuda.matmul.allow_tf32=False
        adapters={}
        for name,record in checkpoints.items():
            adapter=make_adapter(model,name,record['config']).eval().requires_grad_(False)
            adapter.load_state_dict(record['state']);adapters[name]=adapter
        rows,gates,flops=[],{},[]
        scorer=evaluator()
        for index,image in enumerate(manifest['images']):
            shard=output/'shards'/f"{image['image_id']}.json"
            if shard.exists():result=read_json(shard)
            else:
                result=predict_image(model,adapters['S1b'],adapters['A2'],scorer,image,proposals,
                                     paths['textvqa'],readers,gates,index<launch.get('flops_images',3))
                write_json(shard,result)
            rows.extend(result['rows']);gates.update(result['identity']);flops.extend(result['flops'])
            if index<3 or (index+1)%50==0:
                print(f"Matrix {index+1}/{len(manifest['images'])} images / {len(rows)} predictions",flush=True)
        finish(output,binding,report(output,rows,manifest,readers,gates,flops),rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    run(parser.parse_args().output)
