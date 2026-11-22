"""One matched old/new evaluation: full, crop and fusion, two readers/budgets.

All checkpoints are fixed before inference. CLEAN uses the *same LR4-selected*
crop, so this is a quality/control comparison, not a new clean-image selector.
"""

import argparse
import csv
import io
import json
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from filelock import FileLock

from qlrg.common import atomic_write, object_hash, read_json, sha256, write_json
from qlrg.evaluation import SCORER_REVISION, evaluator, paired_bootstrap
from qlrg.experiment import paired_images
from qlrg.s1_manifests import read_s1_manifests
from scripts.checkpoint_matrix_plan import comparison_pairs, module_plan, resolve_training, specifications
from scripts.compute_cost import count_flops


def write_table(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, stream.getvalue().encode('utf-8'))


def training_records(launch, allow_pending=False):
    records, pending = {}, []
    for label, root, method in [('new_A0', launch['a0_training'], None),
                                ('new_A2_restoration', launch['a2_training'], 'restoration'),
                                ('new_A2_ce_only', launch['a2_training'], 'ce_only')]:
        if not (Path(root) / 'COMPLETE.json').exists() and allow_pending:
            pending.append(label)
            continue
        for view in ('full', 'crop'):
            records[label + '_' + view] = resolve_training(root, view, method)
    return records, pending


def validate_state(record):
    if record['baseline_kept']:
        zero = torch.load(record['selection_source'], map_location='cpu', weights_only=True)
        if (zero['examples'] != 0 or zero['binding'] != record['selection_binding']
                or any(torch.count_nonzero(zero['adapter'][key]) for key in ('output.weight', 'output.bias'))):
            raise ValueError('An aliased baseline contains a trained/nonzero correction')
        return None
    path = Path(record['source'])
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state['binding'] != record['checkpoint_binding'] or state['arm'] != 'S1b'
            or state['examples'] != record['examples']):
        raise ValueError('Refit state does not match completed training: ' + str(path))
    expected_reader = record['training_binding'].get('reader', 'A0')
    if state.get('reader', 'A0') != expected_reader or state.get('method', 'restoration') != record['method']:
        raise ValueError('Refit objective/reader changed')
    if not state['adapter'] or any(not torch.isfinite(v).all() for v in state['adapter'].values()):
        raise ValueError('Invalid/nonfinite visual module state')
    from qlrg.s1_model import VisualResidual
    cfg = record['training_binding']['base_config']
    module = VisualResidual(state['adapter']['output.weight'].shape[0], cfg['hidden'], cfg['spatial_blocks'])
    module.load_state_dict(state['adapter'], strict=True)
    return state['adapter']


def preflight(output, allow_pending=False):
    launch = read_json(output / 'launch.json')
    records, pending = training_records(launch, allow_pending)
    for record in records.values():
        validate_state(record)
    source = Path(launch['source_matrix'])
    complete, binding = read_json(source / 'COMPLETE.json'), read_json(source / 'run.json')
    if complete['binding'] != object_hash(binding) or binding['stage'] != 'crop_matrix':
        raise ValueError('Archived matrix source binding changed')
    for name in ('run.json', 'summary.json', 'predictions.jsonl'):
        if sha256(source / name) != complete['artifacts'][name]:
            raise ValueError('Archived source artifact changed: ' + name)
    for record in binding['checkpoints'].values():
        if sha256(Path(record['source'])) != record['sha256']:
            raise ValueError('Original 11305 checkpoint changed')
    for record in records.values():
        if record['training_binding']['shared'] != binding['shared']:
            raise ValueError('Training and old comparison environments differ')
    return {'source_matrix_verified': True, 'verified_training_records': list(records),
            'pending_training_groups': pending, 'ready': not pending}


def profiles_for(records):
    profiles = {'old11305': {'full': 'old11305', 'crop': 'old11305', 'readers': ['A0', 'A2']}}
    for label in ('new_A0', 'new_A2_restoration', 'new_A2_ce_only'):
        profiles[label] = {v: None if records[label + '_' + v]['baseline_kept'] else label + '_' + v
                           for v in ('full', 'crop')}
        # A0-trained modules are also tested through A2 to quantify composition.
        # A2-trained feature and CE-only controls are evaluated with their fixed reader.
        profiles[label]['readers'] = ['A0', 'A2'] if label == 'new_A0' else ['A2']
    return profiles


@torch.inference_mode()
def predict_image(model, adapters, lora, scorer, image, proposals, root, specs, gates, profile_flops):
    from scripts.crop_fusion import fit_token_canvas
    clean, lr4 = paired_images(root, image)
    pixels = {'CLEAN': clean, 'LR4': lr4}
    cache, rows, flops = {}, [], []
    ordered = list(reversed(specs)) if int(object_hash(image['image_id'])[:8], 16) % 2 else specs
    for question in image['questions']:
        qid = question['question_id']
        proposal = proposals[(image['image_id'], qid)]
        box = proposal['box']
        if box is not None and not (0 <= box[0] < box[2] <= lr4.width and 0 <= box[1] < box[3] <= lr4.height):
            raise ValueError('Archived crop outside image: ' + str(qid))
        refs = scorer._compute_answer_scores(question['answers'])
        for spec in ordered:
            layout, budget, condition = (spec[k] for k in ('layout', 'budget', 'condition'))
            names, modules = module_plan(layout, tuple(spec['modules']), box is not None)
            source = pixels[condition]
            views = [source.crop(box) if name == 'crop' else source for name in names]
            tokens = 576 // len(views) if budget == 'fair576' else None
            if tokens is not None:
                views = [fit_token_canvas(v, tokens)[0] for v in views]
            batch, grids = model.prepare_views(views, question['question'], exact_budget=tokens is not None)
            key = (condition, budget, names, tuple(tuple(g) for g in grids),
                   None if names == ('full',) else qid)
            if key not in cache:
                torch.cuda.synchronize(); started = time.perf_counter()
                base = tuple(model.encode_views(batch))
                torch.cuda.synchronize()
                cache[key] = {'base': base, 'vision_seconds': time.perf_counter() - started, 'restored': {}}
            entry = cache[key]
            selected, restore_times = [], []
            for slot, (base, grid, module) in enumerate(zip(entry['base'], grids, modules)):
                if module is None:
                    selected.append(base); restore_times.append(0.)
                    continue
                restore_key = (slot, module)
                if restore_key not in entry['restored']:
                    torch.cuda.synchronize(); started = time.perf_counter()
                    value = adapters[module](base, grid)[0].to(model.dtype)
                    if value.shape != base.shape or not torch.isfinite(value).all():
                        raise RuntimeError('Restoration shape/finite gate failed: ' + module)
                    torch.cuda.synchronize()
                    entry['restored'][restore_key] = value, time.perf_counter() - started
                value, seconds = entry['restored'][restore_key]
                selected.append(value); restore_times.append(seconds)
            context = lora.attached() if spec['reader'] == 'A2' else nullcontext()
            with context:
                gate = '__'.join((spec['reader'], condition, budget, layout))
                if gate not in gates:
                    if not torch.equal(model.prefill_logits(batch), model.prefill_logits(batch, entry['base'])):
                        raise RuntimeError('Identity injection failed: ' + gate)
                    gates[gate] = {'question_id': qid, 'logits_exact': True}
                torch.cuda.reset_peak_memory_stats()
                result = model.generate(batch, tuple(selected))
                peak = torch.cuda.max_memory_allocated()
            vision = entry['vision_seconds']; restoration = sum(restore_times)
            reader_seconds = vision + restoration + result['generation_seconds']
            selector_seconds = proposal['selector_seconds'] if layout != 'full' else 0.
            row = {**result, **spec, 'image_id': image['image_id'], 'question_id': qid,
                   'score': refs.get(scorer.answer_processor(result['answer']), 0.),
                   'effective_views': list(names), 'effective_modules': list(modules),
                   'crop_xyxy': box if layout != 'full' else None,
                   'textocr_retained': proposal.get('textocr_retained') if layout != 'full' else None,
                   'fallback': box is None and layout != 'full',
                   'view_tokens': [int(h * w) for h, w in grids],
                   'visual_tokens': sum(int(h * w) for h, w in grids),
                   'vision_seconds': vision, 'restore_seconds': restoration,
                   'reader_seconds': reader_seconds, 'selector_seconds': selector_seconds,
                   'pipeline_seconds': reader_seconds + selector_seconds, 'peak_cuda_bytes': peak}
            rows.append(row)
            if profile_flops and question == image['questions'][0]:
                def operation():
                    fresh = tuple(model.encode_views(batch))
                    values = tuple(adapters[m](f, g)[0].to(model.dtype) if m else f
                                   for f, g, m in zip(fresh, grids, modules))
                    attach = lora.attached() if spec['reader'] == 'A2' else nullcontext()
                    with attach, model.replacement(batch, values):
                        return model.model.generate(**batch, do_sample=False, num_beams=1, use_cache=True,
                                                    min_new_tokens=8, max_new_tokens=8,
                                                    repetition_penalty=model.config['repetition_penalty'])
                _, count = count_flops(operation)
                flops.append({'arm': spec['arm'], 'image_id': image['image_id'],
                              'registered_reader_flops': count, 'fixed_decode_tokens': 8})
        # Only full-image tensors survive to another question; no disk feature cache.
        cache = {k: v for k, v in cache.items() if k[2] == ('full',)}
    return {'rows': rows, 'identity': gates, 'flops': flops}


def report(output, rows, manifest, specs, table, gates, flops):
    cases = [(im['image_id'], q['question_id']) for im in manifest['images'] for q in im['questions']]
    groups = [iid for iid, _ in cases]
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row['arm']].append(row)
    if set(by_arm) != {s['arm'] for s in specs}:
        raise ValueError('Unexpected/missing canonical arms')
    costs = defaultdict(list)
    for f in flops:
        costs[f['arm']].append(f['registered_reader_flops'])
    scores, metrics = {}, {}
    for spec in specs:
        name = spec['arm']; values = by_arm[name]
        if [(r['image_id'], r['question_id']) for r in values] != cases:
            raise ValueError('Every arm must contain the same ordered 5,000 questions: ' + name)
        if spec['budget'] == 'fair576' and any(r['visual_tokens'] != 576 for r in values):
            raise ValueError('A fair-budget arm does not use exactly 576 tokens')
        scores[name] = np.asarray([r['score'] for r in values])
        metric = {**spec, 'questions': len(cases), 'images': len(set(groups)),
                  'accuracy_pct': 100 * float(scores[name].mean()),
                  'visual_tokens_mean': mean(r['visual_tokens'] for r in values),
                  'visual_tokens_max': max(r['visual_tokens'] for r in values),
                  'fallbacks': sum(r['fallback'] for r in values),
                  'peak_cuda_bytes': max(r['peak_cuda_bytes'] for r in values),
                  'profiled_images': len(costs[name]),
                  'registered_reader_flops_decode8_mean': mean(costs[name]) if costs[name] else None}
        for component in ('vision', 'restore', 'reader', 'pipeline'):
            times = [1000 * r[component + '_seconds'] for r in values]
            metric[component + '_p50_ms'] = float(np.median(times))
            metric[component + '_p95_ms'] = float(np.percentile(times, 95))
        metrics[name] = metric
    comparisons, errors = [], []
    for left, right in comparison_pairs(table):
        a, b = scores[left], scores[right]
        comparisons.append({'reference': left, 'target': right, 'questions': len(cases),
                            **paired_bootstrap(a, b, groups, 10000, 42),
                            'score_improved': int(sum(b > a)), 'score_worsened': int(sum(b < a)),
                            'correct_to_wrong': int(sum((a == 1) & (b == 0))),
                            'wrong_to_correct': int(sum((a == 0) & (b == 1)))})
        for x, y in zip(by_arm[left], by_arm[right]):
            if x['score'] != y['score']:
                errors.append({'reference': left, 'target': right, 'image_id': x['image_id'],
                               'question_id': x['question_id'], 'reference_answer': x['answer'],
                               'target_answer': y['answer'], 'reference_score': x['score'],
                               'target_score': y['score'], 'crop_xyxy': y['crop_xyxy'],
                               'textocr_retained': y['textocr_retained'], 'cause': 'unreviewed'})
    display = [{**metrics[r['arm']], **r} for r in table]
    write_table(output / 'accuracy.csv', list(metrics.values()))
    write_table(output / 'comparison_table.csv', display)
    write_table(output / 'paired.csv', comparisons)
    if errors:
        write_table(output / 'error_cases.csv', errors)
    if flops:
        write_table(output / 'flops.csv', flops)
    return {'split': manifest['split'], 'conditions': ['CLEAN', 'LR4'], 'questions_per_arm': len(cases),
            'images': len(manifest['images']), 'unique_arms': len(specs), 'table_rows': len(table),
            'predictions': len(rows), 'accuracy': display, 'comparisons': comparisons, 'identity': gates,
            'bootstrap_unit': 'image_id', 'bootstrap_resamples': 10000, 'scorer_revision': SCORER_REVISION,
            'uncertainty': 'Pointwise CI conditional on fixed checkpoints; no seed variance/multiplicity correction.',
            'timing_policy': 'Frozen vision/restoration timing reused; archived selector timing. Not fresh end-to-end latency.',
            'flops_policy': 'Registered reader ops, vision + restoration + prefill + 8 forced decode tokens; excludes unsupported ops and selector.',
            'crop_policy': 'Same archived LR4/query-selected Vision-RL2 7B box + 20% padding for CLEAN and LR4.',
            'selection_policy': 'Use completed all-train refits selected only on val_inner; zero winner aliases the no-module baseline.'}


def run(output):
    from qlrg.s1_run import make_adapter
    from scripts.crop_fusion import FusionQwen
    from scripts.crop_restore_followup import load_boxes
    from scripts.followup_common import bind, finish, load_study
    from scripts.s1_convergence import merged_train
    with FileLock(str(output / '.run.lock'), timeout=0):
        ready = preflight(output)
        write_json(output / 'preflight.json', ready)
        launch, paths, config, manifest, old, binding = load_study(output, 'checkpoint_matrix', ('S1b', 'A2'))
        archived = read_json(Path(launch['source_matrix']) / 'run.json')
        if archived['manifest_hash'] != binding['manifest_hash'] or archived['shared'] != binding['shared']:
            raise ValueError('Dataset or inference environment differs from the original comparison matrix')
        records, pending = training_records(launch)
        manifests, _ = read_s1_manifests(Path(launch['data_dir']), paths['textvqa'])
        all_train_hash = object_hash(merged_train(manifests))
        for record in records.values():
            source = record['training_binding']
            if (source['official_train_hash'] != all_train_hash
                    or source['dev_manifest_hash'] != object_hash(manifests['val_inner'])
                    or source['train_manifest_hash'] != object_hash(manifests['train_inner']['images'])
                    or source['shared'] != binding['shared']):
                raise ValueError('New checkpoint model/dataset/protocol differs from evaluation binding')
            if source.get('reader') == 'A2' and source['a2']['sha256'] != old['A2']['provenance']['sha256']:
                raise ValueError('A2-conditioned training used another reader')
        proposals, selector = load_boxes(Path(launch['selector_benchmark']), binding['manifest_hash'], old['A2']['provenance'])
        profiles = profiles_for(records)
        specs, table = specifications(profiles)
        binding.update(profiles=profiles, training_checkpoints=records, arms=specs, table=table,
                       selector_binding=object_hash(selector), source_matrix_binding=object_hash(archived),
                       flops_images=launch['flops_images'], code_files={p.name: sha256(p) for p in [
                           Path(__file__), Path(__file__).with_name('checkpoint_matrix_plan.py'),
                           Path(__file__).with_name('crop_fusion.py'), Path(__file__).with_name('followup_common.py')]})
        bind(output, binding)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = FusionQwen(config, paths['hf_hub'])
        lora = make_adapter(model, 'A2', old['A2']['config']).eval().requires_grad_(False)
        lora.load_state_dict(old['A2']['state'], strict=True)
        adapters = {'old11305': make_adapter(model, 'S1b', old['S1b']['config']).eval().requires_grad_(False)}
        adapters['old11305'].load_state_dict(old['S1b']['state'], strict=True)
        for name, record in records.items():
            state = validate_state(record)
            if state is not None:
                adapter = make_adapter(model, 'S1b', record['training_binding']['base_config']).eval().requires_grad_(False)
                adapter.load_state_dict(state, strict=True)
                adapters[name] = adapter
        write_json(output / 'plan.json', {'unique_generations_per_question': len(specs), 'display_rows': len(table),
                                         'total_predictions': len(specs) * 5000, 'profiles': profiles})
        print(f'Locked {len(specs)} unique arms x 5000 questions; {len(table)} table rows after aliases.', flush=True)
        rows, gates, flops = [], {}, []
        scorer = evaluator()
        for index, image in enumerate(manifest['images']):
            shard = output / 'shards' / (image['image_id'] + '.json')
            if shard.exists():
                result = read_json(shard)
                if result['binding'] != object_hash(binding):
                    raise ValueError('Resume shard belongs to another checkpoint matrix')
            else:
                result = predict_image(model, adapters, lora, scorer, image, proposals['visionrl2_7b'], paths['textvqa'],
                                       specs, gates, index < launch['flops_images'])
                result['binding'] = object_hash(binding)
                write_json(shard, result)
            rows.extend(result['rows']); gates.update(result['identity']); flops.extend(result['flops'])
            if index < 3 or (index + 1) % 25 == 0:
                print(f'Checkpoint matrix {index + 1}/{len(manifest["images"])} images / {len(rows)} predictions', flush=True)
        finish(output, binding, report(output, rows, manifest, specs, table, gates, flops), rows)
        print('COMPLETE: all checkpoints, all 5000 questions, CLEAN/LR4, both token budgets.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        run(args.output)
    except BaseException as exc:
        # Independent small journal; no source lookups/locals from a potentially failing NFS.
        frames, tb = [], exc.__traceback__
        while tb is not None:
            frames.append({'file': tb.tb_frame.f_code.co_filename, 'line': tb.tb_lineno,
                           'function': tb.tb_frame.f_code.co_name})
            tb = tb.tb_next
        journal = Path.home() / '.qlrg_recovery' / args.output.name
        journal.mkdir(parents=True, exist_ok=True)
        (journal / ('error-' + str(os.getpid()) + '.json')).write_text(json.dumps({
            'type': type(exc).__name__, 'message': str(exc), 'frames': frames}), encoding='utf-8')
        raise
