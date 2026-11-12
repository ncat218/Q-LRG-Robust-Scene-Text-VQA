"""Evaluate full-image and crop features at an exact, shared visual token budget."""

import io
import math
import time
from itertools import groupby
from pathlib import Path
from statistics import mean

import torch
from PIL import Image

from qlrg.common import atomic_write, object_hash
from qlrg.evaluation import SCORER_REVISION, paired_bootstrap
from qlrg.experiment import paired_images
from qlrg.feature_injection import inject_features
from qlrg.grid_alignment import image_grid
from qlrg.qwen_wrapper import Qwen
from scripts.query_crop import PADDING, padded_box


TOTAL_TOKENS = 576
VIEW_TOKENS = TOTAL_TOKENS // 2
PIXELS_PER_TOKEN = 28
FUSION_LAYOUTS = (
    (("full",), TOTAL_TOKENS, (("full", (False,)), ("full_S1b", (True,)))),
    (("crop",), TOTAL_TOKENS, (("crop", (False,)), ("crop_S1b", (True,)))),
    (("full", "crop"), VIEW_TOKENS,
     (("fusion", (False, False)), ("fusion_full_S1b", (True, False)),
      ("fusion_crop_S1b", (False, True)), ("fusion_both_S1b", (True, True)))),
)
FUSION_LABELS = {
    "full": "toàn ảnh · Qwen gốc",
    "full_S1b": "toàn ảnh · phục hồi S1b",
    "crop": "crop nới 20% · Qwen gốc",
    "crop_S1b": "crop nới 20% · phục hồi S1b",
    "fusion": "toàn ảnh + crop · giữ nguyên cả hai",
    "fusion_full_S1b": "toàn ảnh + crop · chỉ phục hồi toàn ảnh",
    "fusion_crop_S1b": "toàn ảnh + crop · chỉ phục hồi crop",
    "fusion_both_S1b": "toàn ảnh + crop · phục hồi cả hai",
}
FUSION_ALL_LABELS = {**FUSION_LABELS, "full_A2": "toàn ảnh · LoRA A2"}
FALLBACK_ARM = {
    "crop": "full", "crop_S1b": "full_S1b", "fusion": "full", "fusion_full_S1b": "full_S1b",
    "fusion_crop_S1b": "full", "fusion_both_S1b": "full_S1b",
}


class FusionQwen(Qwen):
    """Use native ordered image placeholders and preserve a grid for each image."""

    def prepare_views(self, views, question, *, exact_budget=True):
        messages = [{"role": "user", "content": [
            *({"type": "image"} for _ in views),
            {"type": "text", "text": self.config["prompt"].format(question=question)},
        ]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt], images=views, return_tensors="pt")
        merge = self.model.model.visual.spatial_merge_size
        expected = [(im.height // PIXELS_PER_TOKEN, im.width // PIXELS_PER_TOKEN) for im in views]
        grids = [image_grid(thw, merge) for thw in batch["image_grid_thw"].tolist()]
        image_id = self.model.config.image_token_id
        spans = [sum(1 for _ in span) for visual, span in groupby(batch["input_ids"][0].tolist(),
                                                                key=lambda token: token == image_id) if visual]
        if spans != [h * w for h, w in grids]:
            raise RuntimeError("Qwen processor changed the ordered view grids")
        if exact_budget and (grids != expected or sum(spans) != TOTAL_TOKENS):
            raise RuntimeError("Qwen processor changed the ordered view grids or exact visual token budget")
        return {name: tensor.to(self.device) for name, tensor in batch.items()}, grids

    def encode_views(self, batch):
        return self.model.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"])

    def replacement(self, batch, feature):
        if feature is None:
            return super().replacement(batch, None)
        return inject_features(self.model.model, batch["image_grid_thw"], feature)


def fit_token_canvas(image, tokens):
    """Fit inside an exact-area canvas without stretching the source aspect ratio."""
    ratio = image.width / image.height
    widths = [n for n in range(1, tokens + 1) if tokens % n == 0 and 1/200 <= n / (tokens // n) <= 200]
    w = min(widths,
            key=lambda n: abs(math.log((n / (tokens // n)) / ratio)))
    height, width = tokens // w * PIXELS_PER_TOKEN, w * PIXELS_PER_TOKEN
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    scaled = image.resize(size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), (127, 127, 127))
    canvas.paste(scaled, ((width - size[0]) // 2, (height - size[1]) // 2))
    return canvas, size[0] * size[1] / (width * height)


def select_fusion_cohort(images, crops):
    selected = []
    for image in images:
        proposals = crops[image["image_id"]]["questions"]
        questions = [q for q in image["questions"] if proposals[str(q["question_id"])]["query"]["box"] is not None]
        if questions:
            selected.append({**image, "questions": questions})
    if not selected:
        raise ValueError("No real SD-RPN crop available for the fusion comparison")
    return selected


def save_views(output, key, views):
    path = Path("review-images") / f"{object_hash(key)[:24]}.jpg"
    panels = [view.copy() for view in views]
    for panel in panels:
        panel.thumbnail((330, 280))
    preview = Image.new("RGB", (sum(p.width for p in panels) + 8 * (len(panels) - 1),
                                max(p.height for p in panels)), (32, 32, 32))
    offset = 0
    for panel in panels:
        preview.paste(panel, (offset, 0))
        offset += panel.width + 8
    encoded = io.BytesIO()
    preview.save(encoded, "JPEG", quality=82)
    atomic_write(output / path, encoded.getvalue())
    return path.as_posix()


@torch.inference_mode()
def predict_fusion_image(model, adapter, scorer, image, proposals, root, output, check_identity=False, lora=None):
    _, lr4 = paired_images(root, image)
    rows, identity = [], None
    iid = image["image_id"]
    full_canvases = {tokens: fit_token_canvas(lr4, tokens) for tokens in (TOTAL_TOKENS, VIEW_TOKENS)}
    for question in image["questions"]:
        qid = question["question_id"]
        original_box = proposals["questions"][str(qid)]["query"]["box"]
        box = padded_box(original_box, lr4.size, PADDING["pad20"]) if original_box is not None else None
        crop = lr4.crop(box) if box is not None else None
        question_rows = {}
        references = scorer._compute_answer_scores(question["answers"])
        layouts = FUSION_LAYOUTS if crop is not None else FUSION_LAYOUTS[:1]
        for names, tokens, arms in layouts:
            fitted = [full_canvases[tokens] if view == "full" else fit_token_canvas(crop, tokens) for view in names]
            views, occupancy = zip(*fitted)
            key = [iid, "fusion-budget-full"] if names == ("full",) else [iid, qid, *names]
            preview = save_views(output, key, views)
            batch, grids = model.prepare_views(list(views), question["question"])
            torch.cuda.synchronize()
            start = time.perf_counter()
            base = model.encode_views(batch)
            torch.cuda.synchronize()
            vision_seconds = time.perf_counter() - start
            if len(base) != len(grids):
                raise RuntimeError("Vision encoder returned the wrong number of views")
            if check_identity and len(names) == 2 and identity is None:
                native = model.prefill_logits(batch)
                injected = model.prefill_logits(batch, base)
                if not torch.equal(native, injected):
                    raise RuntimeError("Two-image identity injection changed the native Qwen logits")
                identity = {"question_id": qid, "visual_tokens": TOTAL_TOKENS, "logits_exact": True}
            restored, restore_seconds = [], []
            for feature, grid in zip(base, grids):
                start = time.perf_counter()
                replacement, _ = adapter(feature, grid)
                if replacement.shape != feature.shape:
                    raise RuntimeError("S1b changed the shape of a view")
                restored.append(replacement.to(model.dtype))
                torch.cuda.synchronize()
                restore_seconds.append(time.perf_counter() - start)
            for arm, mask in arms:
                selected = tuple(r if restore else b for b, r, restore in zip(base, restored, mask))
                result = model.generate(batch, selected)
                question_rows[arm] = {
                    "image_id": iid, "question_id": qid, "arm": arm, "crop_xyxy": box,
                    "roi_selected": crop is not None,
                    "views": list(names), "view_grids": dict(zip(names, grids)),
                    "view_tokens": dict(zip(names, (h * w for h, w in grids))),
                    "restored_views": [name for name, restore in zip(names, mask) if restore],
                    "canvas_content_fraction": dict(zip(names, occupancy)),
                    "visual_tokens": TOTAL_TOKENS, "prompt_tokens": batch["input_ids"].shape[1],
                    "preview": preview, "vision_seconds": vision_seconds,
                    "restore_seconds": sum(s for s, restore in zip(restore_seconds, mask) if restore),
                    **result, "score": references.get(scorer.answer_processor(result["answer"]), 0.0),
                }
            if lora is not None and names == ("full",):
                with lora.attached():
                    result = model.generate(batch, base)
                question_rows["full_A2"] = {**question_rows["full"], **result, "arm": "full_A2",
                                            "score": references.get(scorer.answer_processor(result["answer"]), 0.0)}
        if crop is None:
            for arm, source in FALLBACK_ARM.items():
                question_rows[arm] = {**question_rows[source], "arm": arm, "fallback_reason": "no_roi"}
        rows.extend(question_rows.values())
    return {"rows": rows, "identity": identity, "selector_seconds": 0.0}


def summarize_fusion(rows, manifest, seed, identity, omitted, include_a2=False):
    images = manifest["images"]
    expected = [(im["image_id"], q["question_id"]) for im in images for q in im["questions"]]
    labels = FUSION_ALL_LABELS if include_a2 else FUSION_LABELS
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in labels}
    if (len(rows) != len(expected) * len(by_arm) or any(r["visual_tokens"] != TOTAL_TOKENS for r in rows)
        or any([(r["image_id"], r["question_id"]) for r in rs] != expected for rs in by_arm.values())):
        raise ValueError("Fusion arms differ in question cohort or visual token budget")
    if not identity or not identity["logits_exact"]:
        raise ValueError("Missing two-image identity injection gate")
    groups = [iid for iid, _ in expected]
    scores = {arm: [r["score"] for r in rs] for arm, rs in by_arm.items()}
    pairs = (
        ("full", "crop"), ("full", "fusion"), ("crop", "fusion"),
        ("full", "full_S1b"), ("crop", "crop_S1b"),
        ("fusion", "fusion_full_S1b"), ("fusion", "fusion_crop_S1b"), ("fusion", "fusion_both_S1b"),
        ("fusion_full_S1b", "fusion_both_S1b"), ("fusion_crop_S1b", "fusion_both_S1b"),
        ("fusion_full_S1b", "fusion_crop_S1b"),
        ("full_S1b", "fusion_both_S1b"), ("crop_S1b", "fusion_both_S1b"),
    )
    if include_a2:
        pairs += (("full", "full_A2"), ("full_A2", "fusion"), ("full_A2", "fusion_full_S1b"),
                  ("full_A2", "fusion_crop_S1b"), ("full_A2", "fusion_both_S1b"))

    def comparisons(cohort_scores, cohort_groups):
        return [{"reference": a, "target": b,
                 **paired_bootstrap(cohort_scores[a], cohort_scores[b], cohort_groups, 10000, seed),
                 "score_improved": sum(y > x for x, y in zip(cohort_scores[a], cohort_scores[b])),
                 "score_worsened": sum(y < x for x, y in zip(cohort_scores[a], cohort_scores[b]))} for a, b in pairs]

    report = {
        "split": manifest["split"], "images": len(images), "questions": len(expected),
        "excluded_no_roi_questions": omitted, "scorer_revision": SCORER_REVISION,
        "bootstrap_unit": "image_id", "bootstrap_resamples": 10000, "identity": identity,
        "visual_budget": {"total": TOTAL_TOKENS, "fusion_full": VIEW_TOKENS, "fusion_crop": VIEW_TOKENS,
                          "padding_per_side": PADDING["pad20"], "padding_counts_toward_budget": True},
        "arms": {
            arm: {"label": labels[arm], "accuracy_pct": 100 * mean(scores[arm]),
                  "mean_visual_tokens": TOTAL_TOKENS, "mean_prompt_tokens": mean(r["prompt_tokens"] for r in rs),
                  "mean_canvas_content_fraction": {
                      name: mean(r["canvas_content_fraction"][name] for r in rs if name in r["views"])
                      for name in sorted({view for r in rs for view in r["views"]})
                  },
                  "mean_vision_seconds": mean(r["vision_seconds"] for r in rs),
                  "mean_generation_seconds": mean(r["generation_seconds"] for r in rs),
                  "mean_restore_seconds": mean(r["restore_seconds"] for r in rs)}
            for arm, rs in by_arm.items()
        },
        "comparisons": comparisons(scores, groups),
    }
    if include_a2:
        indices = [i for i, row in enumerate(by_arm["full"]) if row["roi_selected"]]
        roi_groups = [groups[i] for i in indices]
        roi_scores = {arm: [values[i] for i in indices] for arm, values in scores.items()}
        report["fallback_questions"] = len(expected) - len(indices)
        report["roi_only"] = {
            "questions": len(indices), "images": len(set(roi_groups)),
            "accuracy_pct": {arm: 100 * mean(values) for arm, values in roi_scores.items()},
            "comparisons": comparisons(roi_scores, roi_groups),
        }
    return report
