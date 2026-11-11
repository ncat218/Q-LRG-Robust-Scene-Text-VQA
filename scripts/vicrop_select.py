"""ViCrop relative attention adapted to the pinned Qwen2.5-VL-3B reader."""

import time

import numpy as np
import torch
from transformers import AutoProcessor

from qlrg.data import rank
from qlrg.experiment import paired_images
from qlrg.grid_alignment import image_grid
from scripts.crop_benchmark import fraction_retained


UPSTREAM_REVISION = "a64c5d5a0ece793a2f7dc96e926d283a199033c1"
LAYERS = (7, 15, 22, 30)
GENERIC_QUESTION = "Write a general description of the image."
WINDOW_PX = 224


def select_box(attention, size):
    """Maximize peak-to-neighbor contrast per cell over the six ViCrop windows."""
    height, width = attention.shape
    cell_w, cell_h = size[0] / width, size[1] / height
    integral = np.pad(attention, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    best, box = -np.inf, None
    for ratio in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
        side = WINDOW_PX * ratio
        w = min(width, max(1, int(side / cell_w)))
        h = min(height, max(1, int(side / cell_h)))
        if (h, w) == attention.shape:
            break
        sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
        y, x = np.unravel_index(sums.argmax(), sums.shape)
        neighbors = [sums[a, b] for a, b in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                     if 0 <= a < sums.shape[0] and 0 <= b < sums.shape[1]]
        score = (sums[y, x] - np.mean(neighbors)) / (h * w)
        if score > best:
            crop_w, crop_h = min(size[0], round(side)), min(size[1], round(side))
            left = min(size[0] - crop_w, max(0, round((x + w / 2) * cell_w - crop_w / 2)))
            top = min(size[1] - crop_h, max(0, round((y + h / 2) * cell_h - crop_h / 2)))
            box = (left, top, left + crop_w, top + crop_h)
            best = score
    return box


class ViCrop:
    def __init__(self, model):
        self.reader = model
        self.processor = AutoProcessor.from_pretrained(
            model.snapshot, local_files_only=True, use_fast=True, min_pixels=3136, max_pixels=288 * 28 * 28)

    def prepare(self, image, question):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": f"{question} Answer the question using a single word or phrase."},
        ]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.reader.device)

    def attention(self, batch, features, layers, grid):
        model = self.reader.model
        mask = batch["input_ids"][0] == model.config.image_token_id
        if int(mask.sum()) != grid[0] * grid[1]:
            raise ValueError("ViCrop attention does not align with the visual grid")
        language = model.model.language_model.config
        previous = language._attn_implementation
        language._attn_implementation = "eager"
        try:
            with self.reader.replacement(batch, features):
                result = model(**batch, use_cache=False, logits_to_keep=1, output_attentions=True)
            return {layer: result.attentions[layer][0, :, -1, mask].float().mean(dim=0).reshape(grid).cpu().numpy()
                    for layer in layers}
        finally:
            language._attn_implementation = previous

    @torch.inference_mode()
    def start_image(self, image, layers):
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = self.prepare(image, GENERIC_QUESTION)
        grid = image_grid(batch["image_grid_thw"][0].tolist())
        features = self.reader.encode_views(batch)
        generic = self.attention(batch, features, layers, grid)
        torch.cuda.synchronize()
        return features, grid, generic, time.perf_counter() - start

    @torch.inference_mode()
    def locate(self, image, question, state):
        features, grid, generic, shared_seconds = state
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = self.prepare(image, question)
        if image_grid(batch["image_grid_thw"][0].tolist()) != grid:
            raise ValueError("ViCrop query changed the image grid")
        maps = self.attention(batch, features, tuple(generic), grid)
        boxes = {layer: select_box(scores / np.maximum(generic[layer], 1e-12), image.size)
                 for layer, scores in maps.items()}
        torch.cuda.synchronize()
        return boxes, shared_seconds, time.perf_counter() - start


def calibrate(selector, manifest, polygons, root, seed, pad):
    candidates = [(record, q) for record in manifest["images"] for q in record["questions"]
                  if q["question_id"] in polygons]
    candidates.sort(key=lambda row: rank(seed, row[1]["question_id"]))
    if not candidates:
        raise ValueError("No TextOCR-grounded val_inner cases for ViCrop calibration")
    hits, retained = dict.fromkeys(LAYERS, 0), dict.fromkeys(LAYERS, 0.0)
    sample = candidates[:128]
    for record, q in sample:
        _, image = paired_images(root, record)
        boxes, _, _ = selector.locate(image, q["question"], selector.start_image(image, LAYERS))
        for layer, box in boxes.items():
            overlap = fraction_retained(pad(box, image.size), polygons[q["question_id"]]) if box else 0.0
            hits[layer] += int(overlap >= 0.90)
            retained[layer] += overlap
    layer = max(LAYERS, key=lambda layer: (hits[layer], retained[layer], -layer))
    return layer, {"split": manifest["split"], "questions": len(sample), "layer_zero_based": layer,
                   "layers": {str(i): {"retains_90pct": hits[i], "mean_retained": retained[i] / len(sample)}
                              for i in LAYERS}}
