"""Scoped substitution at Qwen2.5-VL's post-Merger image feature boundary."""

from contextlib import contextmanager

import torch

from .grid_alignment import image_grid


@contextmanager
def inject_features(backbone, grid, features):
    grid_values = grid.tolist()
    if len(features) != len(grid_values):
        raise ValueError("Feature count does not match image count")
    width = backbone.config.text_config.hidden_size
    for thw, feature in zip(grid_values, features):
        h, w = image_grid(thw, backbone.visual.spatial_merge_size)
        if feature.shape != (h * w, width):
            raise ValueError("Replacement shape does not match the merged image grid")
        if feature.device != grid.device or feature.dtype != backbone.visual.dtype:
            raise ValueError("Replacement device and dtype must match the vision encoder")

    name = "get_image_features"
    if name in vars(backbone):
        raise RuntimeError("Nested image feature injection is unsupported")
    calls = 0

    def substitute(pixel_values, image_grid_thw=None):
        nonlocal calls
        if image_grid_thw is None or not torch.equal(grid, image_grid_thw):
            raise RuntimeError("Image grid changed during feature injection")
        calls += 1
        if calls != 1:
            raise RuntimeError("Image features were requested more than once per forward/generation")
        return tuple(features)

    setattr(backbone, name, substitute)
    try:
        yield
        if calls != 1:
            raise RuntimeError("Qwen bypassed the image feature injection point")
    finally:
        delattr(backbone, name)


def replace_region(degraded, clean, mask):
    if clean.shape != degraded.shape or mask.size != degraded.shape[0]:
        raise ValueError("Cannot substitute misaligned visual features")
    mask = torch.as_tensor(mask, device=degraded.device).reshape(-1, 1)
    # Direct selection preserves the exact BF16 value on both sides.
    return torch.where(mask, clean, degraded)
