"""Controlled post-Merger restoration ablations for Q-LRG."""

import torch
from torch import nn
from torch.nn import functional as F

from qlrg.data import get_words
from qlrg.grid_alignment import project
from qlrg.s1_model import VisualResidual, teacher_forced_loss


METHODS = ("s1b", "text_weighted", "text_relational", "scale_aware", "frequency", "bounded", "text_gate")


class FrequencyResidual(VisualResidual):
    """Adapt FMVR's spatial pooling to the S1b bottleneck on a fixed token grid."""

    def __init__(self, width, hidden, spatial_blocks):
        super().__init__(width, hidden, spatial_blocks)
        self.avg_gain = nn.Parameter(torch.ones(1, hidden, 1, 1))
        self.max_gain = nn.Parameter(torch.ones(1, hidden, 1, 1))

    def forward(self, base, grid):
        h, w = grid
        x = self.input(base.float()).T.reshape(1, -1, h, w)
        avg = x - F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
        anti = x - F.max_pool2d(x, 3, stride=1, padding=1)
        x = x + self.avg_gain * avg + self.max_gain * anti
        delta = self.output(self.spatial(x).flatten(2)[0].T)
        return base.float() + delta, delta


class BoundedResidual(VisualResidual):
    """Bound correction in fixed clean-training per-channel standard deviations."""

    def __init__(self, width, hidden, spatial_blocks, channel_scale):
        super().__init__(width, hidden, spatial_blocks)
        self.register_buffer("channel_scale", channel_scale)

    def forward(self, base, grid):
        _, delta = super().forward(base, grid)
        bound = 3 * self.channel_scale
        delta = bound * torch.tanh(delta / bound)
        return base.float() + delta, delta


class TextGatedResidual(VisualResidual):
    """Predict a per-token textness gate using degraded visual features."""

    def __init__(self, width, hidden, spatial_blocks):
        super().__init__(width, hidden, spatial_blocks)
        self.gate = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def gate_logits(self, base):
        return self.gate(base.float()).squeeze(-1)

    def forward(self, base, grid):
        _, delta = super().forward(base, grid)
        delta = delta * self.gate_logits(base).sigmoid()[:, None]
        return base.float() + delta, delta


def make_restorer(width, config, method, device, channel_scale=None):
    torch.manual_seed(config["seed"])
    args = width, config["hidden"], config["spatial_blocks"]
    if method == "frequency":
        adapter = FrequencyResidual(*args)
    elif method == "bounded":
        adapter = BoundedResidual(*args, channel_scale)
    elif method == "text_gate":
        adapter = TextGatedResidual(*args)
    elif method in METHODS:
        adapter = VisualResidual(*args)
    else:
        raise ValueError(f"Unknown latent restoration method: {method}")
    return adapter.to(device)


def text_mask(ocr, image, grid, device):
    meta = ocr["imgs"].get(image["image_id"])
    if meta is None or image["size"] != [meta["width"], meta["height"]]:
        return None
    words = get_words(ocr, image["image_id"])
    return torch.as_tensor(project(words, image["size"], grid, "overlap", 0.05).ravel(), device=device)


def local_relations(features, grid):
    x = F.normalize(features, dim=-1).reshape(*grid, -1)
    return (x[:, 1:] * x[:, :-1]).sum(-1), (x[1:] * x[:-1]).sum(-1)


def relational_loss(restored, clean, mask, grid):
    m = mask.reshape(grid)
    edges = (m[:, 1:] & m[:, :-1], m[1:] & m[:-1])
    terms = [(a - b).square()[valid] for a, b, valid in
             zip(local_relations(restored, grid), local_relations(clean, grid), edges)]
    distances = torch.cat(terms)
    return distances.sum() / max(1, distances.numel())


def feature_loss(method, restored, clean, mask, grid):
    if method == "scale_aware":
        s = restored.std(dim=-1, unbiased=False).clamp_min(1e-5)
        t = clean.std(dim=-1, unbiased=False).clamp_min(1e-5)
        return (F.mse_loss(F.layer_norm(restored, (restored.shape[-1],)),
                           F.layer_norm(clean, (clean.shape[-1],))) +
                (s.log() - t.log()).square().mean())
    if method in ("text_weighted", "text_relational", "text_gate") and mask is not None:
        weight = 1 + 4 * mask.float()
        match = ((restored - clean).square().mean(dim=-1) * weight).sum() / weight.sum()
        return match + (relational_loss(restored, clean, mask, grid) if method == "text_relational" else 0)
    return F.mse_loss(restored, clean)


def study_loss(qwen, method, adapter, cache, question, condition, answer, config):
    base, clean = cache.features[condition], cache.features["CLEAN"].float()
    restored, delta = adapter(base, cache.grid)
    mask = cache.mask if method in ("text_weighted", "text_relational", "text_gate") else None
    match = feature_loss(method, restored, clean, mask, cache.grid)
    tokenizer = qwen.processor.tokenizer
    target = tokenizer.encode(answer, add_special_tokens=False)[:config["max_answer_tokens"]]
    target.append(tokenizer.eos_token_id)
    vqa = teacher_forced_loss(qwen, cache.prompt(condition, question), restored.to(qwen.dtype), target)
    residual_weight = config["residual_weight"]
    if condition == "CLEAN":
        residual_weight += config["clean_residual_weight"]
    loss = vqa + config["feature_weight"] * match + residual_weight * delta.square().mean()
    if method == "text_gate" and mask is not None:
        loss = loss + 0.1 * F.binary_cross_entropy_with_logits(adapter.gate_logits(base), mask.float())
    return loss
