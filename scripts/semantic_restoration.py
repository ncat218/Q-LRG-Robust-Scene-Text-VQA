"""Matched continuation of 11305 S1b: control, answer KD, answer KD + patch KD."""

import argparse
import io
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from PIL import ImageFilter
from torch.nn import functional as F

from qlrg.common import atomic_write, object_hash, read_json, read_jsonl, sha256, write_json, write_jsonl
from qlrg.evaluation import evaluator, paired_bootstrap
from qlrg.grid_alignment import image_grid
from qlrg.qwen_wrapper import Qwen, assert_aligned
from qlrg.s1_data import training_examples
from qlrg.s1_manifests import read_s1_manifests
from qlrg.s1_model import teacher_forced_loss, text_training
from qlrg.s1_run import make_adapter
from scripts.compute_cost import count_flops
from scripts.followup_common import bind, load_study, write_table
from scripts.latent_methods import text_mask
from scripts.semantic_losses import METHODS, semantic_objective, valid_resume_offset


CONDITIONS = ('CLEAN', 'LR4', 'BLUR4', 'LR4_BLUR4')


class SemanticFeatures:
    """One full-image pair in memory, no persistent feature/crop cache."""

    def __init__(self, model, root, ocr, blur_radius):
        self.model, self.root, self.ocr, self.blur_radius = model, root, ocr, blur_radius
        self.current = None

    def load(self, image, question, include_blur=False):
        key = (image['image_id'], include_blur)
        if key == self.current:
            return
        from qlrg.experiment import paired_images
        clean, lr4 = paired_images(self.root, image)
        self.images = {'CLEAN': clean, 'LR4': lr4}
        if include_blur:
            self.images.update(BLUR4=clean.filter(ImageFilter.GaussianBlur(self.blur_radius)),
                               LR4_BLUR4=lr4.filter(ImageFilter.GaussianBlur(self.blur_radius)))
        batches = {c: self.model.prepare(p, question['question']) for c, p in self.images.items()}
        for batch in batches.values():
            assert_aligned(batches['CLEAN'], batch)
        self.grid = image_grid(batches['CLEAN']['image_grid_thw'][0].tolist())
        self.features, self.vision_seconds = {}, {}
        for c, batch in batches.items():
            torch.cuda.synchronize(); started = time.perf_counter()
            self.features[c] = self.model.encode(batch).detach()
            torch.cuda.synchronize(); self.vision_seconds[c] = time.perf_counter() - started
        self.mask = text_mask(self.ocr, image, self.grid, self.model.device)
        self.current = key

    def prompt(self, condition, question):
        batch = self.model.prepare(self.images[condition], question)
        if image_grid(batch['image_grid_thw'][0].tolist()) != self.grid:
            raise ValueError('Question or degradation changed the aligned feature grid')
        return batch


def setup(output):
    launch, paths, model_config, official, old, base = load_study(output, 'semantic_restoration', ('S1b', 'A2'))
    settings = launch['semantic_config']
    if (settings['methods'] != list(METHODS) or settings['extra_epochs'] != 2 or settings['train_reader'] != 'A0'
            or settings['early_stopping'] or settings['feature_disk_cache'] or settings['official_val_for_selection']):
        raise ValueError('Run the locked three-arm, fixed-exposure study')
    manifests, hashes = read_s1_manifests(Path(launch['data_dir']), paths['textvqa'])
    if (len(manifests['train_inner']['images']) != 19758
            or sum(len(im['questions']) for im in manifests['train_inner']['images']) != 31129
            or len(manifests['val_inner']['images']) != 2195
            or sum(len(im['questions']) for im in manifests['val_inner']['images']) != 3473):
        raise ValueError('Expected the full audited train_inner and val_inner, no pilot subset')
    config = {**old['S1b']['config'], **settings}
    source = read_json(Path(launch['s1_run']) / 's1/run.json')
    if config['clean_fraction'] != source['config']['clean_fraction']:
        raise ValueError('Do not confound the objective with a changed CLEAN fraction')
    binding = {**base, 'settings': settings, 'base_config': old['S1b']['config'],
               'dataset_hashes': hashes, 'ocr_sha256': sha256(paths['textocr']),
               'loss_code_sha256': sha256(Path(__file__).with_name('semantic_losses.py')),
               'worker_sha256': sha256(Path(__file__)),
               'paper_ledger_sha256': sha256(Path(launch['repo']) / 'docs/semantic_restoration_study.md'),
               'initialization': '11305 S1b weights; fresh identical AdamW in each arm',
               'primary_checkpoint': 'latest.pt after exactly 62258 extra question visits',
               'official_selection': False}
    bind(output, binding)
    return launch, paths, model_config, manifests, old, config, binding


def model_and_adapter(model_config, paths, old, config):
    torch.manual_seed(config['seed']); torch.backends.cuda.matmul.allow_tf32 = False
    model = Qwen(model_config, paths['hf_hub'])
    adapter = make_adapter(model, 'S1b', old['S1b']['config'])
    adapter.load_state_dict(old['S1b']['state'], strict=True)
    return model, adapter


def checkpoint(path, adapter, optimizer, offset, binding, method):
    buffer = io.BytesIO()
    torch.save({'binding': object_hash(binding), 'method': method, 'adapter': adapter.state_dict(),
                'optimizer': optimizer.state_dict(), 'examples': offset, 'cpu_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state()}, buffer)
    atomic_write(path, buffer.getvalue())


def probe(output):
    _, paths, model_config, manifests, old, config, binding = setup(output)
    model, adapter = model_and_adapter(model_config, paths, old, config)
    cache = SemanticFeatures(model, paths['textvqa'], read_json(paths['textocr']), config['blur_radius'])
    probes = []
    for image in manifests['train_inner']['images'][:8]:
        question = image['questions'][0]
        answer = Counter(a.strip() for a in question['answers']).most_common(1)[0][0]
        cache.load(image, question)
        prompt = cache.prompt('LR4', question['question'])
        if not torch.equal(model.prefill_logits(prompt), model.prefill_logits(prompt, cache.features['LR4'])):
            raise RuntimeError('Native/injection prefill identity failed')
        with text_training(model.model, True):
            for method in METHODS:
                adapter.zero_grad(set_to_none=True)
                loss, terms, info = semantic_objective(model, adapter, cache, question, 'LR4', answer, config, method)
                norms = {}
                for name, term in terms.items():
                    gradients = torch.autograd.grad(term, tuple(adapter.parameters()), retain_graph=True, allow_unused=True)
                    norms[name] = math.sqrt(sum(float(g.detach().square().sum()) for g in gradients if g is not None))
                if not math.isfinite(float(loss.detach())) or norms['task'] <= 0:
                    raise RuntimeError('Answer CE does not reach the visual module')
                if info['kd_positions'] and norms.get('answer_kd', 0) <= 0:
                    raise RuntimeError('Eligible semantic KD has no visual gradient')
                if info['contrast_valid_negatives'] and norms.get('patch', 0) <= 0:
                    raise RuntimeError('Patch KD has no visual gradient')
                if any(p.grad is not None or p.requires_grad for p in model.model.parameters()):
                    raise RuntimeError('Reader/vision encoder must remain frozen')
                probes.append({'method': method, 'question_id': question['question_id'], 'gradient_norms': norms, **info})
        if any(p['kd_positions'] for p in probes if p['method'] == 'answer_kd'):
            break
    if not any(p['kd_positions'] for p in probes):
        raise RuntimeError('No usable teacher positions in eight probe questions; inspect before training')
    write_json(output / 'gradient_probe.json', {'binding': object_hash(binding), 'passed': True, 'probes': probes})
    print('GPU gradient/teacher/injection gates passed for all three objectives.', flush=True)


def feature_diagnostics(base, restored, clean, mask):
    clean, base, restored = clean.float(), base.float(), restored.float()
    out = {'feature_mse': float(F.mse_loss(restored, clean)),
           'raw_feature_mse': float(F.mse_loss(base, clean)),
           'residual_norm_ratio': float((restored - base).norm() / base.norm().clamp_min(1e-8)),
           'text_tokens': int(mask.sum()) if mask is not None else 0,
           'text_feature_mse': None, 'background_feature_mse': None}
    if mask is not None:
        if mask.any():
            out['text_feature_mse'] = float(F.mse_loss(restored[mask], clean[mask]))
        if (~mask).any():
            out['background_feature_mse'] = float(F.mse_loss(restored[~mask], clean[~mask]))
    direction = clean - base
    out['correction_direction_cosine'] = (float(F.cosine_similarity((restored - base).flatten()[None],
            direction.flatten()[None])[0]) if direction.norm() > 1e-8 else None)
    return out


@torch.inference_mode()
def eval_image(model, adapter, lora, cache, image, scorer, label, official, profile=False):
    cache.load(image, image['questions'][0], include_blur=official)
    rows, flops = [], []
    conditions = CONDITIONS if official else ('CLEAN', 'LR4')
    for condition in conditions:
        base = cache.features[condition]
        torch.cuda.synchronize(); started = time.perf_counter()
        restored = adapter(base, cache.grid)[0].to(model.dtype)
        torch.cuda.synchronize(); seconds = time.perf_counter() - started
        variants = [('A0', None, base), ('old11305', None, restored), ('A2', lora, base),
                    ('A2_old11305', lora, restored)] if label == 'baseline' else [
                    (label, None, restored), ('A2_' + label, lora, restored)] if official else [(label, None, restored)]
        for question in image['questions']:
            refs = scorer._compute_answer_scores(question['answers'])
            prompt = cache.prompt(condition, question['question'])
            for arm, attached, value in variants:
                with attached.attached() if attached is not None else nullcontext():
                    torch.cuda.reset_peak_memory_stats()
                    result = model.generate(prompt, value)
                    peak = torch.cuda.max_memory_allocated()
                    if profile and question == image['questions'][0]:
                        enabled = arm not in ('A0', 'A2')
                        def operation():
                            fresh = model.encode(prompt)
                            replacement = adapter(fresh, cache.grid)[0].to(model.dtype) if enabled else fresh
                            with model.replacement(prompt, replacement):
                                return model.model.generate(**prompt, do_sample=False, num_beams=1, use_cache=True,
                                                            min_new_tokens=8, max_new_tokens=8,
                                                            repetition_penalty=model.config['repetition_penalty'])
                        _, count = count_flops(operation)
                        flops.append({'arm': arm, 'condition': condition, 'image_id': image['image_id'],
                                      'registered_reader_flops': count, 'fixed_decode_tokens': 8})
                use_module = arm not in ('A0', 'A2')
                restore_seconds = seconds if use_module else 0.
                rows.append({**result, 'image_id': image['image_id'], 'question_id': question['question_id'],
                             'condition': condition, 'arm': arm,
                             'score': refs.get(scorer.answer_processor(result['answer']), 0.),
                             'visual_tokens': len(base), 'vision_seconds': cache.vision_seconds[condition],
                             'restore_seconds': restore_seconds,
                             'reader_seconds': cache.vision_seconds[condition] + restore_seconds + result['generation_seconds'],
                             'peak_cuda_bytes': peak, **feature_diagnostics(base, value, cache.features['CLEAN'], cache.mask)})
    return rows, flops


@torch.inference_mode()
def validate(model, adapter, cache, manifest, label):
    scorer = evaluator(); rows = []
    adapter.eval()
    for image in manifest['images']:
        values, _ = eval_image(model, adapter, None, cache, image, scorer, label, False)
        rows.extend(values)
    scores = {c: 100 * np.mean([r['score'] for r in rows if r['condition'] == c]) for c in ('CLEAN', 'LR4')}
    return {k: float(v) for k, v in scores.items()}, rows


def train_method(output, method):
    worker_started = time.perf_counter()
    _, paths, model_config, manifests, old, config, binding = setup(output)
    work = output / method; work.mkdir(exist_ok=True)
    bound = {**binding, 'method': method}
    bind(work, bound)
    total = 2 * sum(len(im['questions']) for im in manifests['train_inner']['images'])
    one_epoch = total // 2
    if (work / 'TRAIN_COMPLETE.json').exists():
        done = read_json(work / 'TRAIN_COMPLETE.json')
        if done['binding'] != object_hash(bound) or sha256(work / 'latest.pt') != done['checkpoint_sha256']:
            raise ValueError('Completed continuation checkpoint changed')
        return
    model, adapter = model_and_adapter(model_config, paths, old, config)
    cache = SemanticFeatures(model, paths['textvqa'], read_json(paths['textocr']), config['blur_radius'])
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    examples = list(training_examples(manifests['train_inner']['images'], 2, config['seed'], config['clean_fraction']))
    offset = 0
    if (work / 'latest.pt').exists():
        saved = torch.load(work / 'latest.pt', map_location='cpu', weights_only=True)
        if saved['binding'] != object_hash(bound) or saved['method'] != method:
            raise ValueError('Continuation resume binding changed')
        offset = saved['examples']; adapter.load_state_dict(saved['adapter']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
        if not valid_resume_offset(offset, one_epoch, config['accumulate']):
            raise ValueError('Invalid resume example offset')
    interrupted = [False]
    signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    every = max(1, math.ceil(total / config['accumulate'] / config['checkpoint_writes_per_arm']))
    if offset == one_epoch and not (work / 'val_epoch1.json').exists():
        scores, rows = validate(model, adapter, cache, manifests['val_inner'], method)
        write_json(work / 'val_epoch1.json', {'epoch': 1, 'scores': scores, 'role': 'monitoring_only_no_selection'})
        write_jsonl(work / 'val_epoch1_predictions.jsonl', rows)
    window = []
    while offset < total:
        end = one_epoch if offset < one_epoch else total
        adapter.train()
        with text_training(model.model, True):
            while offset < end:
                group = examples[offset:min(end, offset + config['accumulate'])]
                optimizer.zero_grad(set_to_none=True)
                stats = defaultdict(float)
                for image, question, condition, answer in group:
                    cache.load(image, question)
                    loss, terms, info = semantic_objective(model, adapter, cache, question, condition, answer, config, method)
                    (loss / len(group)).backward()
                    stats['loss'] += float(loss.detach()) / len(group)
                    for key, value in terms.items():
                        stats[key] += float(value.detach()) / len(group)
                    for key, value in info.items():
                        stats[key] += float(value) / len(group)
                norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), config['max_grad_norm'], error_if_nonfinite=True))
                if not math.isfinite(stats['loss']) or any(p.grad is not None for p in model.model.parameters()):
                    raise RuntimeError('Nonfinite loss or gradient reached the frozen reader')
                optimizer.step(); offset += len(group)
                stats.update(examples=offset, epoch=offset / one_epoch, grad_norm=norm,
                             peak_cuda_bytes=torch.cuda.max_memory_allocated())
                window.append(dict(stats))
                update = math.ceil(offset / config['accumulate'])
                if update % 200 == 0 or offset == end:
                    with (work / 'training.jsonl').open('a', encoding='utf-8') as stream:
                        import json
                        stream.write(json.dumps({'examples': offset, 'epoch': offset / one_epoch,
                            **{k: sum(r[k] for r in window) / len(window) for k in window[0] if k not in ('examples', 'epoch')}}) + '\n')
                    print(f'{method}: {offset}/{total} visits, extra epoch {offset/one_epoch:.3f}, loss {stats["loss"]:.4f}', flush=True)
                    window = []
                if update % every == 0 or interrupted[0]:
                    checkpoint(work / 'latest.pt', adapter, optimizer, offset, bound, method)
                if interrupted[0]:
                    raise SystemExit('SIGTERM: saved continuation checkpoint for resume')
        checkpoint(work / 'latest.pt', adapter, optimizer, offset, bound, method)
        epoch = 1 if offset == one_epoch else 2
        if epoch == 1:
            checkpoint(work / 'epoch1.pt', adapter, optimizer, offset, bound, method)
        scores, rows = validate(model, adapter, cache, manifests['val_inner'], method)
        write_json(work / f'val_epoch{epoch}.json', {'epoch': epoch, 'scores': scores, 'role': 'monitoring_only_no_selection'})
        write_jsonl(work / f'val_epoch{epoch}_predictions.jsonl', rows)
        # Record RNG after validation, so resume neither skips nor repeats a data update.
        checkpoint(work / 'latest.pt', adapter, optimizer, offset, bound, method)
        print(f'{method}: validation extra epoch {epoch}: {scores}; fixed endpoint unchanged.', flush=True)
    write_json(work / 'TRAIN_COMPLETE.json', {'binding': object_hash(bound), 'examples': offset,
               'checkpoint_sha256': sha256(work / 'latest.pt'), 'primary_endpoint': 'fixed_two_extra_epochs',
               'this_invocation_worker_seconds': time.perf_counter() - worker_started})


def evaluate_method(output, method):
    _, paths, model_config, manifests, old, config, binding = setup(output)
    work = output / 'evaluation' / method; work.mkdir(parents=True, exist_ok=True)
    model, adapter = model_and_adapter(model_config, paths, old, config)
    source = old['S1b']['provenance']['sha256']
    if method != 'baseline':
        trained = read_json(output / method / 'TRAIN_COMPLETE.json')
        path = output / method / 'latest.pt'
        if sha256(path) != trained['checkpoint_sha256']:
            raise ValueError('Final continuation checkpoint changed')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['binding'] != object_hash({**binding, 'method': method}) or saved['examples'] != 62258:
            raise ValueError('All arms must have exactly the same additional training exposure')
        adapter.load_state_dict(saved['adapter']); source = trained['checkpoint_sha256']
    adapter.eval().requires_grad_(False)
    lora = make_adapter(model, 'A2', old['A2']['config']).eval().requires_grad_(False)
    lora.load_state_dict(old['A2']['state'])
    cache = SemanticFeatures(model, paths['textvqa'], read_json(paths['textocr']), config['blur_radius'])
    eval_binding = {'run': object_hash(binding), 'method': method, 'checkpoint_sha256': source,
                    'manifest_hash': binding['manifest_hash'], 'conditions': list(CONDITIONS)}
    bind(work, eval_binding)
    rows, flops = [], []
    scorer = evaluator()
    for index, image in enumerate(manifests['official_val']['images']):
        shard = work / 'shards' / (image['image_id'] + '.json')
        if shard.exists():
            data = read_json(shard)
            if data['binding'] != object_hash(eval_binding):
                raise ValueError('Evaluation shard belongs to another checkpoint')
        else:
            with torch.inference_mode():
                values, costs = eval_image(model, adapter, lora, cache, image, scorer, method, True, index < 3)
            data = {'binding': object_hash(eval_binding), 'rows': values, 'flops': costs}
            write_json(shard, data)
        rows.extend(data['rows']); flops.extend(data['flops'])
        if index < 2 or (index + 1) % 100 == 0:
            print(f'Eval {method}: {index+1}/3166 images', flush=True)
    write_jsonl(work / 'predictions.jsonl', rows)
    write_json(work / 'flops.json', flops)
    write_json(work / 'COMPLETE.json', {'binding': object_hash(eval_binding), 'predictions_sha256': sha256(work / 'predictions.jsonl')})


def summarize(output):
    _, _, _, manifests, _, _, binding = setup(output)
    cases = [(im['image_id'], q['question_id']) for im in manifests['official_val']['images'] for q in im['questions']]
    groups = [iid for iid, _ in cases]
    by_arm, flops = defaultdict(list), []
    for method in ('baseline', *METHODS):
        work = output / 'evaluation' / method
        done = read_json(work / 'COMPLETE.json')
        if sha256(work / 'predictions.jsonl') != done['predictions_sha256']:
            raise ValueError('Final predictions checksum changed')
        for row in read_jsonl(work / 'predictions.jsonl'):
            by_arm[(row['arm'], row['condition'])].append(row)
        flops.extend(read_json(work / 'flops.json'))
    arms = ['A0', 'old11305', 'A2', 'A2_old11305', *METHODS, *('A2_' + m for m in METHODS)]
    metrics, scores = [], {}
    for arm in arms:
        for condition in CONDITIONS:
            values = by_arm[(arm, condition)]
            if [(r['image_id'], r['question_id']) for r in values] != cases:
                raise ValueError('Final cohorts differ between checkpoint/condition')
            scores[(arm, condition)] = np.asarray([r['score'] for r in values])
            costs = [f['registered_reader_flops'] for f in flops if f['arm'] == arm and f['condition'] == condition]
            metric = {'arm': arm, 'condition': condition, 'questions': 5000, 'images': 3166,
                      'accuracy_pct': float(100 * scores[(arm, condition)].mean()),
                      'visual_tokens_mean': float(np.mean([r['visual_tokens'] for r in values])),
                      'reader_p50_ms': float(1000 * np.median([r['reader_seconds'] for r in values])),
                      'reader_p95_ms': float(1000 * np.percentile([r['reader_seconds'] for r in values], 95)),
                      'peak_cuda_bytes': max(r['peak_cuda_bytes'] for r in values),
                      'registered_reader_flops_decode8_mean': float(np.mean(costs)), 'flops_images': len(costs)}
            for key in ('feature_mse', 'raw_feature_mse', 'text_feature_mse', 'background_feature_mse',
                        'residual_norm_ratio', 'correction_direction_cosine'):
                eligible = [r[key] for r in values if r[key] is not None]
                metric[key] = float(np.mean(eligible)) if eligible else None
            metrics.append(metric)
    comparisons, interactions, errors = [], [], []
    for reference, target in [('A0', 'old11305'), ('A2', 'A2_old11305')]:
        for condition in CONDITIONS:
            comparisons.append({'reference': reference, 'target': target, 'condition': condition,
                **paired_bootstrap(scores[(reference, condition)], scores[(target, condition)], groups, 10000, 42),
                'score_improved': int(sum(scores[(target, condition)] > scores[(reference, condition)])),
                'score_worsened': int(sum(scores[(target, condition)] < scores[(reference, condition)])),
                'correct_to_wrong': int(sum((scores[(reference, condition)] == 1) & (scores[(target, condition)] == 0))),
                'wrong_to_correct': int(sum((scores[(reference, condition)] == 0) & (scores[(target, condition)] == 1)))})
    for prefix, baseline, original in [('', 'A0', 'old11305'), ('A2_', 'A2', 'A2_old11305')]:
        for method in METHODS:
            target = prefix + method
            references = [baseline, original] + ([prefix + 'continued'] if method != 'continued' else [])
            if method == 'answer_kd_patch':
                references.append(prefix + 'answer_kd')
            for reference in references:
                for condition in CONDITIONS:
                    a, b = scores[(reference, condition)], scores[(target, condition)]
                    comparisons.append({'reference': reference, 'target': target, 'condition': condition,
                        **paired_bootstrap(a, b, groups, 10000, 42), 'score_improved': int(sum(b > a)),
                        'score_worsened': int(sum(b < a)), 'correct_to_wrong': int(sum((a == 1) & (b == 0))),
                        'wrong_to_correct': int(sum((a == 0) & (b == 1)))})
                    if reference == prefix + 'continued' and condition in ('CLEAN', 'LR4'):
                        for left, right in zip(by_arm[(reference, condition)], by_arm[(target, condition)]):
                            if left['score'] != right['score']:
                                errors.append({'reference': reference, 'target': target, 'condition': condition,
                                    'image_id': left['image_id'], 'question_id': left['question_id'],
                                    'reference_answer': left['answer'], 'target_answer': right['answer'],
                                    'reference_score': left['score'], 'target_score': right['score'],
                                    'reference_feature_mse': left['feature_mse'], 'target_feature_mse': right['feature_mse'],
                                    'target_text_feature_mse': right['text_feature_mse'],
                                    'target_direction_cosine': right['correction_direction_cosine'], 'cause': 'unreviewed'})
                delta = (scores[(target, 'LR4')] - scores[(reference, 'LR4')]
                         - scores[(target, 'CLEAN')] + scores[(reference, 'CLEAN')])
                interactions.append({'reference': reference, 'target': target,
                                     **paired_bootstrap(np.zeros(5000), delta, groups, 10000, 42)})
    write_table(output / 'accuracy.csv', metrics)
    write_table(output / 'paired.csv', comparisons)
    write_table(output / 'degradation_interaction.csv', interactions)
    if errors:
        write_table(output / 'error_cases.csv', errors)
    write_json(output / 'summary.json', {'accuracy': metrics, 'comparisons': comparisons, 'interactions': interactions,
               'primary_endpoint': 'fixed_two_extra_epochs', 'train_images': 19758, 'extra_visits_per_arm': 62258,
               'conditions': list(CONDITIONS), 'questions_per_arm': 5000, 'images': 3166,
               'bootstrap_unit': 'image_id', 'bootstrap_resamples': 10000, 'seed': 42,
               'note': 'Q-LRG adaptations, not paper reproductions. No teacher/TextOCR/labels at inference. Fixed checkpoints; no seed uncertainty.',
               'cost_policy': 'Reused encode/restore timings; registered operator FLOPs with fixed decode8 on 3 images, excludes unsupported ops.'})
    write_json(output / 'COMPLETE.json', {'stage': 'semantic_restoration', 'binding': object_hash(binding),
               'artifacts': {p.name: sha256(p) for p in output.iterdir() if p.suffix in ('.csv', '.json') and p.name != 'COMPLETE.json'}})


def run(output, phase=None, method=None):
    if phase == 'probe':
        probe(output)
    elif phase == 'train':
        train_method(output, method)
    elif phase == 'eval':
        evaluate_method(output, method)
    else:
        with FileLock(str(output / '.run.lock'), timeout=0):
            setup(output)
            if not (output / 'gradient_probe.json').exists():
                subprocess.run([sys.executable, '-B', '-m', 'scripts.semantic_restoration', str(output), '--phase', 'probe'], check=True)
            for selected in METHODS:
                write_json(output / 'progress.json', {'phase': 'train', 'method': selected})
                subprocess.run([sys.executable, '-B', '-m', 'scripts.semantic_restoration', str(output),
                                '--phase', 'train', '--method', selected], check=True)
            for selected in ('baseline', *METHODS):
                write_json(output / 'progress.json', {'phase': 'eval', 'method': selected})
                subprocess.run([sys.executable, '-B', '-m', 'scripts.semantic_restoration', str(output),
                                '--phase', 'eval', '--method', selected], check=True)
            summarize(output)
            write_json(output / 'progress.json', {'phase': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--phase', choices=('probe', 'train', 'eval'))
    parser.add_argument('--method', choices=('baseline', *METHODS))
    args = parser.parse_args()
    try:
        run(args.output, args.phase, args.method)
    except BaseException as error:
        import json
        frames, tb = [], error.__traceback__
        while tb is not None:
            frames.append({'file': tb.tb_frame.f_code.co_filename, 'line': tb.tb_lineno, 'function': tb.tb_frame.f_code.co_name})
            tb = tb.tb_next
        directory = Path.home() / '.qlrg_recovery' / args.output.name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'error-{os.getpid()}.json').write_text(json.dumps({
            'type': type(error).__name__, 'message': str(error), 'frames': frames}), encoding='utf-8')
        raise
