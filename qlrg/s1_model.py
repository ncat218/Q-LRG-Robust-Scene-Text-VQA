"""Residual restoration, parameter-matched controls and answer-only training loss."""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class SpatialBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.pointwise = nn.Conv2d(width, width, 1)

    def forward(self, features):
        return features + self.pointwise(F.gelu(self.depthwise(features)))


class VisualResidual(nn.Module):
    """A token-local visual adapter when spatial_blocks is zero."""

    def __init__(self, width, hidden, spatial_blocks=0):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU())
        self.spatial = nn.Sequential(*(SpatialBlock(hidden) for _ in range(spatial_blocks)))
        self.output = nn.Linear(hidden, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, base, grid):
        height, width = grid
        if base.ndim != 2 or base.shape[0] != height * width:
            raise ValueError("Visual residual expects one aligned merged image grid")
        hidden = self.input(base.float())
        if len(self.spatial):
            hidden = hidden.transpose(0, 1).reshape(1, -1, height, width)
            hidden = self.spatial(hidden).flatten(2)[0].transpose(0, 1)
        delta = self.output(hidden)
        return base.float() + delta, delta


class LowRankDelta(nn.Module):
    def __init__(self, input_width, output_width, rank, alpha):
        super().__init__()
        self.down = nn.Linear(input_width, rank, bias=False)
        self.up = nn.Linear(rank, output_width, bias=False)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return (self.up(self.down(x.float())) * self.scale).to(x.dtype)


class LoRALinear(nn.Module):
    def __init__(self, base, delta):
        super().__init__()
        self.base = base
        self.delta = delta

    def forward(self, x):
        return self.base(x) + self.delta(x)


class LanguageLoRA(nn.Module):
    """Own only the trainable deltas; attach them to text Q/V during a run."""

    def __init__(self, language_model, rank, alpha):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.sites = [(layer.self_attn, name) for layer in language_model.layers
                      for name in ("q_proj", "v_proj")]
        if not self.sites:
            raise ValueError("No text attention projections for LoRA")
        self.deltas = nn.ModuleList()
        for owner, name in self.sites:
            base = getattr(owner, name)
            if not isinstance(base, nn.Linear) or any(p.requires_grad for p in base.parameters()):
                raise ValueError(f"LoRA expects a frozen linear projection: {name}")
            self.deltas.append(LowRankDelta(base.in_features, base.out_features, rank, alpha))

    @contextmanager
    def attached(self):
        bases = [getattr(owner, name) for owner, name in self.sites]
        if not all(isinstance(base, nn.Linear) for base in bases):
            raise RuntimeError("LoRA projections are already patched")
        try:
            for (owner, name), base, delta in zip(self.sites, bases, self.deltas):
                setattr(owner, name, LoRALinear(base, delta))
            yield
        finally:
            for (owner, name), base in zip(self.sites, bases):
                setattr(owner, name, base)


def teacher_forced_loss(qwen, prompt, feature, target_ids):
    """Compute causal answer NLL without allocating prompt-length vocabulary logits."""
    ids = prompt["input_ids"]
    if ids.shape[0] != 1 or not target_ids:
        raise ValueError("Teacher forcing expects a single prompt and nonempty answer")
    answer = ids.new_tensor([target_ids])
    positions = torch.arange(ids.shape[1] - 1, ids.shape[1] + answer.shape[1] - 1, device=ids.device)
    full = dict(prompt)
    full["input_ids"] = torch.cat((ids, answer), dim=1)
    mask = prompt["attention_mask"]
    full["attention_mask"] = torch.cat((mask, mask.new_ones(answer.shape)), dim=1)
    # Frozen LLM weights still need a differentiable forward for gradients to the adapter.
    with qwen.replacement(full, feature):
        logits = qwen.model(**full, use_cache=False, logits_to_keep=positions).logits
    return F.cross_entropy(logits[0].float(), answer[0])


@contextmanager
def text_training(model, enable):
    """Use non-reentrant checkpointing so gradients reach injected/LoRA features."""
    if not enable:
        yield
        return
    language = model.model.language_model
    if getattr(model.config.text_config, "attention_dropout", 0.0):
        raise ValueError("S1 expects the pinned Qwen text model with zero attention dropout")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    language.train()
    try:
        yield
    finally:
        model.eval()
        model.gradient_checkpointing_disable()
