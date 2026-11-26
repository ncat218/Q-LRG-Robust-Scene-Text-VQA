"""Q-LRG semantic supervision of a residual visual restorer, not RGB restoration.

Clean/noisy supervision is inspired by NOPD (2607.23125), but this uses
gold-prefix forward KL, not their on-policy reverse-KL algorithm. Patch
InfoNCE adapts Eq. 11 of Parikh et al. (2604.21343v2) at the post-Merger site.
See docs/semantic_restoration_study.md for exact departures and citations.
"""

import torch
from torch.nn import functional as F


METHODS = ('continued', 'answer_kd', 'answer_kd_patch')


def valid_resume_offset(offset, examples_per_epoch, accumulate, epochs=2):
    # Each epoch may end in a shorter accumulation group (31,129 is not
    # divisible by four); validation/checkpoints must still be resumable.
    return (isinstance(offset, int) and 0 <= offset <= examples_per_epoch * epochs
            and (offset % examples_per_epoch) % accumulate == 0)


def causal_answer_logits(qwen, prompt, feature, target_ids):
    """Only answer-prefix positions, including EOS; differentiable in feature."""
    ids = prompt['input_ids']
    if ids.shape[0] != 1 or not target_ids:
        raise ValueError('Expected one prompt and nonempty answer including EOS')
    answer = ids.new_tensor([target_ids])
    positions = torch.arange(ids.shape[1] - 1, ids.shape[1] + len(target_ids) - 1, device=ids.device)
    batch = {**prompt, 'input_ids': torch.cat((ids, answer), dim=1),
             'attention_mask': torch.cat((prompt['attention_mask'], prompt['attention_mask'].new_ones(answer.shape)), dim=1)}
    with qwen.replacement(batch, feature):
        logits = qwen.model(**batch, use_cache=False, logits_to_keep=positions).logits[0].float()
    if logits.shape[0] != len(target_ids):
        raise ValueError('Answer-position logits/targets do not align')
    return logits


def clean_answer_kd(student_logits, teacher_logits, targets, temperature):
    """KL(clean teacher || restored student), only GT-consistent teacher positions.

    The GT gate is a training-only Q-LRG choice. It avoids explicitly copying a
    known wrong teacher token; no label/teacher is used at inference.
    """
    if student_logits.shape != teacher_logits.shape or temperature <= 0:
        raise ValueError('KD requires aligned logits and positive temperature')
    teacher_logits = teacher_logits.detach().float()
    targets = torch.as_tensor(targets, device=student_logits.device)
    gate = teacher_logits.argmax(-1).eq(targets)
    student_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher_logits / temperature, dim=-1)
    values = F.kl_div(student_log, teacher_log, reduction='none', log_target=True).sum(-1)
    loss = (values * gate).sum() / gate.sum().clamp_min(1) * temperature ** 2
    return loss, {'kd_positions': int(gate.sum()), 'answer_positions': len(targets),
                  'kd_eos_eligible': bool(gate[-1])}


def patch_infonce(restored, clean, mask, max_anchors, temperature, equivalent_cosine):
    """Paired spatial targets with bounded memory and masked near-identical negatives.

    Choose text anchors when training annotations exist; otherwise all cells.
    Never drop an image or depend on TextOCR at inference. Selection is
    deterministic so adding this loss cannot change the main training RNG.
    """
    if restored.shape != clean.shape or restored.ndim != 2:
        raise ValueError('Patch features must be spatially aligned matrices')
    if max_anchors < 1 or temperature <= 0 or not 0 < equivalent_cosine <= 1:
        raise ValueError('Invalid patch objective configuration')
    count = len(restored)
    if mask is not None and (mask.ndim != 1 or len(mask) != count):
        raise ValueError('Text mask differs from the feature grid')
    candidates = torch.nonzero(mask, as_tuple=False).flatten() if mask is not None else torch.arange(count, device=restored.device)
    text_anchors = bool(len(candidates)) and mask is not None
    if not len(candidates):
        candidates = torch.arange(count, device=restored.device)
    if len(candidates) > max_anchors:
        candidates = candidates[torch.linspace(0, len(candidates) - 1, max_anchors, device=restored.device).long()]
    if count < 2:
        return restored.sum() * 0, {'contrast_anchors': len(candidates), 'contrast_text_anchors': text_anchors,
                                    'contrast_valid_negatives': 0}
    teacher = F.normalize(clean.detach().float(), dim=-1)
    student = F.normalize(restored[candidates].float(), dim=-1)
    logits = student @ teacher.T / temperature
    # Repeated/near-identical teacher patches are not reliable negatives.
    excluded = teacher[candidates] @ teacher.T >= equivalent_cosine
    excluded[torch.arange(len(candidates), device=restored.device), candidates] = False
    logits = logits.masked_fill(excluded, torch.finfo(logits.dtype).min)
    loss = F.cross_entropy(logits, candidates)
    return loss, {'contrast_anchors': len(candidates), 'contrast_text_anchors': text_anchors,
                  'contrast_valid_negatives': int((~excluded).sum()) - len(candidates)}


def semantic_objective(qwen, adapter, cache, question, condition, answer, config, method):
    if method not in METHODS:
        raise ValueError('Unknown semantic-restoration method')
    base, clean = cache.features[condition], cache.features['CLEAN'].float()
    restored, delta = adapter(base, cache.grid)
    tokenizer = qwen.processor.tokenizer
    targets = tokenizer.encode(answer, add_special_tokens=False)[:config['max_answer_tokens']]
    targets.append(tokenizer.eos_token_id)
    student = causal_answer_logits(qwen, cache.prompt(condition, question['question']), restored.to(qwen.dtype), targets)
    task = F.cross_entropy(student, student.new_tensor(targets, dtype=torch.long))
    match = F.mse_loss(restored, clean)
    residual = delta.square().mean()
    residual_weight = config['residual_weight'] + (config['clean_residual_weight'] if condition == 'CLEAN' else 0)
    terms = {'task': task, 'feature': config['feature_weight'] * match,
             'regularization': residual_weight * residual}
    info = {'feature_mse': float(match.detach()), 'residual_mse': float(residual.detach()),
            'kd_positions': 0, 'answer_positions': len(targets), 'kd_eos_eligible': False,
            'contrast_anchors': 0, 'contrast_text_anchors': False, 'contrast_valid_negatives': 0}
    if method != 'continued':
        with torch.no_grad():
            teacher = causal_answer_logits(qwen, cache.prompt('CLEAN', question['question']),
                                           clean.to(qwen.dtype), targets)
        kd, statistics = clean_answer_kd(student, teacher, targets, config['kd_temperature'])
        terms['answer_kd'] = config['kd_weight'] * kd
        info.update(statistics)
    if method == 'answer_kd_patch':
        contrast, statistics = patch_infonce(restored, clean, cache.mask, config['contrast_anchors'],
                                             config['contrast_temperature'], config['equivalent_cosine'])
        terms['patch'] = config['contrast_weight'] * contrast
        info.update(statistics)
    return sum(terms.values()), terms, info
