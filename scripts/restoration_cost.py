"""Compare pixel and latent restoration with the same LR4 questions and reader budget."""

import csv
import hashlib
import io
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image

from qlrg.common import atomic_write, sha256
from qlrg.evaluation import SCORER_REVISION, paired_bootstrap
from qlrg.experiment import paired_images
from scripts.compute_cost import measure
from scripts.crop_fusion import FALLBACK_ARM, FUSION_LABELS, FUSION_LAYOUTS, TOTAL_TOKENS, fit_token_canvas, save_views
from scripts.query_crop import PADDING, padded_box


PIXEL_SHA256 = "5c68d18db561aed8ead4ffedf1b897ea615baaf60ebf6c35f8e641f8fa4a21bf"
LABELS = {**FUSION_LABELS, "pixel_x4": "toàn ảnh · phục hồi pixel FSRCNN ×4"}
PIXEL_LAYOUT = (("pixel",), TOTAL_TOKENS, (("pixel_x4", (False,)),))
TIMED_STAGES = ("selector", "pixel", "prepare", "vision", "latent", "generate")
FLOP_STAGES = ("selector", "pixel", "vision", "latent", "generate")
PAIRS = (
    ("full", "pixel_x4"), ("full", "full_S1b"), ("pixel_x4", "full_S1b"),
    ("crop", "crop_S1b"), ("full_S1b", "crop_S1b"), ("full", "fusion"),
    ("fusion", "fusion_full_S1b"), ("fusion", "fusion_crop_S1b"), ("fusion", "fusion_both_S1b"),
    ("full_S1b", "fusion_both_S1b"), ("pixel_x4", "fusion_both_S1b"),
)


class PixelRestorer:
    def __init__(self, weights, runtime, threads):
        if sha256(weights) != PIXEL_SHA256:
            raise ValueError("Expected pinned public FSRCNN ×4 weights")
        runtime = Path(runtime).resolve()
        sys.path.insert(0, str(runtime))
        try:
            import cv2
        finally:
            sys.path.pop(0)
        if cv2.__version__ != "4.10.0" or not Path(cv2.__file__).resolve().is_relative_to(runtime):
            raise RuntimeError("Expected the isolated OpenCV contrib 4.10.0 runtime")
        cv2.setNumThreads(threads)
        self.model = cv2.dnn_superres.DnnSuperResImpl_create()
        self.model.readModel(str(weights))
        self.model.setModel("fsrcnn", 4)
        self.net = cv2.dnn.readNetFromTensorflow(str(weights))
        self.flops = {}

    def restore(self, low, original_size):
        start = time.perf_counter()
        bgr = np.ascontiguousarray(np.asarray(low)[:, :, ::-1])
        result = self.model.upsample(bgr)
        result = Image.fromarray(np.ascontiguousarray(result[:, :, ::-1]))
        if result.size != (low.width * 4, low.height * 4):
            raise ValueError("FSRCNN did not upscale both axes by four")
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.BICUBIC)
        elapsed = time.perf_counter() - start
        if low.size not in self.flops:
            self.flops[low.size] = int(self.net.getFLOPS([1, 1, low.height, low.width]))
        return result, elapsed, self.flops[low.size]


@torch.inference_mode()
def predict_image(model, adapter, pixel, scorer, image, proposals, root, output, profile_ids, check_identity):
    clean, lr4 = paired_images(root, image)
    iid = image["image_id"]
    if (proposals["image_id"] != iid or hashlib.sha256(lr4.tobytes()).hexdigest() != proposals["lr4_sha256"]
        or set(proposals["questions"]) != {str(q["question_id"]) for q in image["questions"]}):
        raise ValueError(f"SD-RPN used different LR4 pixels or questions for {iid}")
    low = clean.resize((max(1, clean.width // 4), max(1, clean.height // 4)), Image.Resampling.BICUBIC)
    if low.resize(lr4.size, Image.Resampling.BICUBIC).tobytes() != lr4.tobytes():
        raise ValueError("Pixel restoration input does not reproduce the audited LR4 control")
    restored_pixels, pixel_seconds, pixel_flops = pixel.restore(low, lr4.size)
    rows, identity = [], None
    for question in image["questions"]:
        qid = question["question_id"]
        profile = str(qid) in profile_ids
        region = proposals["questions"][str(qid)]["query"]
        box = padded_box(region["box"], lr4.size, PADDING["pad20"]) if region["box"] is not None else None
        question_rows = {}
        refs = scorer._compute_answer_scores(question["answers"])
        layouts = (*FUSION_LAYOUTS, PIXEL_LAYOUT) if box is not None else (FUSION_LAYOUTS[0], PIXEL_LAYOUT)
        for names, tokens, arms in layouts:
            def prepare():
                sources = {"full": lr4, "pixel": restored_pixels}
                if "crop" in names:
                    sources["crop"] = lr4.crop(box)
                views = [fit_token_canvas(sources[name], tokens)[0] for name in names]
                batch, grids = model.prepare_views(views, question["question"])
                return views, batch, grids

            (views, batch, grids), prepare_seconds, _ = measure(prepare)
            base, vision_seconds, (_, vision_flops) = measure(lambda: model.encode_views(batch), profile)
            if check_identity and len(names) == 2 and identity is None:
                native = model.prefill_logits(batch)
                injected = model.prefill_logits(batch, base)
                if not torch.equal(native, injected):
                    raise RuntimeError("Two-view identity injection changed Qwen logits")
                identity = {"question_id": qid, "visual_tokens": TOTAL_TOKENS, "logits_exact": True}
            restored, latent_costs = [], []
            if names != ("pixel",):
                for feature, grid in zip(base, grids, strict=True):
                    def restore():
                        replacement, _ = adapter(feature, grid)
                        if replacement.shape != feature.shape:
                            raise RuntimeError("S1b changed the shape of a view")
                        return replacement.to(model.dtype)

                    replacement, seconds, (_, flops) = measure(restore, profile)
                    restored.append(replacement)
                    latent_costs.append((seconds, flops))
            preview_key = [iid, *names] if "crop" not in names else [iid, qid, *names]
            preview = save_views(output, preview_key, views)
            for arm, mask in arms:
                selected = tuple(restored[i] if flag else base[i] for i, flag in enumerate(mask))
                result, generation_seconds, (profiled_result, generation_flops) = measure(
                    lambda: model.generate(batch, selected), profile)
                timings = {
                    "selector": region["selector_seconds"] if "crop" in names else 0.0,
                    "pixel": pixel_seconds if arm == "pixel_x4" else 0.0,
                    "prepare": prepare_seconds, "vision": vision_seconds,
                    "latent": sum(latent_costs[i][0] for i, flag in enumerate(mask) if flag),
                    "generate": generation_seconds,
                }
                flops = None
                if profile:
                    flops = {
                        "selector": region["selector_flops"] if "crop" in names else 0,
                        "pixel": pixel_flops if arm == "pixel_x4" else 0,
                        "vision": vision_flops,
                        "latent": sum(latent_costs[i][1] for i, flag in enumerate(mask) if flag),
                        "generate": generation_flops,
                    }
                question_rows[arm] = {
                    **result, "arm": arm, "image_id": iid, "question_id": qid, "preview": preview,
                    "roi_selected": "crop" in names, "crop_xyxy": box if "crop" in names else None,
                    "view_tokens": [h * w for h, w in grids], "visual_tokens": sum(h * w for h, w in grids),
                    "prompt_tokens": batch["input_ids"].shape[1], "generated_tokens": len(result["token_ids"]),
                    "profile_generated_tokens": len(profiled_result["token_ids"]) if profile else None,
                    "time_seconds": timings, "pipeline_seconds": sum(timings.values()),
                    "flops": flops, "total_flops": sum(flops.values()) if flops is not None else None,
                    "score": refs.get(scorer.answer_processor(result["answer"]), 0.0),
                }
        if box is None:
            for arm, fallback in FALLBACK_ARM.items():
                source = question_rows[fallback]
                timings = {**source["time_seconds"], "selector": region["selector_seconds"]}
                flops = {**source["flops"], "selector": region["selector_flops"]} if profile else None
                question_rows[arm] = {
                    **source, "arm": arm, "time_seconds": timings, "pipeline_seconds": sum(timings.values()),
                    "flops": flops, "total_flops": sum(flops.values()) if profile else None, "fallback_reason": "no_roi",
                }
        rows.extend(question_rows.values())
    return {"rows": rows, "identity": identity, "selector_seconds": proposals["selector_seconds"]}


def summarize(rows, manifest, seed, identity, profile_ids):
    expected = [(im["image_id"], q["question_id"]) for im in manifest["images"] for q in im["questions"]]
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in LABELS}
    if not identity or not identity["logits_exact"]:
        raise ValueError("Missing two-view identity gate")
    if (len(rows) != len(expected) * len(by_arm) or any(r["visual_tokens"] != TOTAL_TOKENS for r in rows)
        or any([(r["image_id"], r["question_id"]) for r in rs] != expected for rs in by_arm.values())):
        raise ValueError("Restoration cost benchmark changed the cohort or visual token budget")
    arms = {}
    for arm, rs in by_arm.items():
        measured = [r for r in rs if r["flops"] is not None]
        if {str(r["question_id"]) for r in measured} != profile_ids or any(
            r["flops"]["vision"] <= 0 or r["flops"]["generate"] <= 0 for r in measured
        ):
            raise ValueError(f"Missing registered FLOPs in {arm}")
        arms[arm] = {
            "label": LABELS[arm], "questions": len(rs), "profile_questions": len(measured),
            "accuracy_pct": 100 * mean(r["score"] for r in rs), "visual_tokens": TOTAL_TOKENS,
            "mean_prompt_tokens": mean(r["prompt_tokens"] for r in rs),
            "mean_generated_tokens": mean(r["generated_tokens"] for r in rs),
            "mean_profile_generated_tokens": mean(r["profile_generated_tokens"] for r in measured),
            "mean_pipeline_ms": 1000 * mean(r["pipeline_seconds"] for r in rs),
            "p50_pipeline_ms": 1000 * float(np.quantile([r["pipeline_seconds"] for r in rs], 0.50)),
            "p95_pipeline_ms": 1000 * float(np.quantile([r["pipeline_seconds"] for r in rs], 0.95)),
            "mean_estimated_GFLOPs": mean(r["total_flops"] for r in measured) / 1e9,
            **{f"mean_{stage}_ms": 1000 * mean(r["time_seconds"][stage] for r in rs) for stage in TIMED_STAGES},
            **{f"mean_{stage}_GFLOPs": mean(r["flops"][stage] for r in measured) / 1e9 for stage in FLOP_STAGES},
            "fallback_questions": sum("fallback_reason" in r for r in rs),
        }
    groups = [iid for iid, _ in expected]
    scores = {arm: [r["score"] for r in rs] for arm, rs in by_arm.items()}
    return {
        "split": manifest["split"], "questions": len(expected), "images": len(manifest["images"]),
        "profile_question_ids": sorted(profile_ids), "seed": seed, "identity": identity,
        "scorer_revision": SCORER_REVISION, "bootstrap_unit": "image_id", "bootstrap_resamples": 10000,
        "flop_method": "PyTorch FlopCounterMode registered ops and OpenCV Net.getFLOPS for FSRCNN",
        "latency_method": "sum of synchronized unprofiled stages, excluding loading and report I/O",
        "arms": arms, "comparisons": [
            {"reference": a, "target": b, **paired_bootstrap(scores[a], scores[b], groups, 10000, seed)}
            for a, b in PAIRS
        ],
    }


def write_cost_csv(path, arms):
    rows = [{"arm": arm, **metrics} for arm, metrics in arms.items()]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, output.getvalue().encode("utf-8"))
