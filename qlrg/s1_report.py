"""S1 paired comparison and feature-change diagnostics."""

import csv
import io
import re
from collections import Counter
from statistics import mean

import numpy as np
from PIL import ImageDraw

from .case_review import write_case_review
from .common import atomic_write, write_json, write_jsonl
from .evaluation import SCORER_REVISION, SCORER_SHA256, paired_bootstrap


CONDITIONS = ("CLEAN", "LR4")
LABELS = {
    "A0": "Qwen đóng băng, giữ feature đầu vào",
    "S1a": "phục hồi feature, học khớp feature sạch",
    "S1b": "phục hồi feature, học từ đáp án và feature sạch",
    "A1": "visual adapter theo từng token, học từ đáp án",
    "A2": "LoRA trên mô hình ngôn ngữ, học từ đáp án",
}
EXTRA_LABELS = {
    "A1clean": "visual adapter chỉ học trên ảnh sạch",
    "A2clean": "LoRA chỉ học trên ảnh sạch",
}
PAIRS = [("A0", arm) for arm in LABELS if arm != "A0"] + [
    (arm, "S1b") for arm in ("S1a", "A1", "A2")
]


def write_visuals(stage, condition, image_id, image, delta, grid):
    """Draw the per-token relative RMS residual on top of the input RGB image."""
    magnitude = delta.square().mean(dim=-1).sqrt().reshape(grid).cpu().numpy()
    peak = float(magnitude.max())
    picture = image.copy()
    picture.thumbnail((640, 480))
    draw = ImageDraw.Draw(picture, "RGBA")
    height, width = grid
    for y in range(height):
        for x in range(width):
            x0, x1 = int(x * picture.width / width), int((x + 1) * picture.width / width)
            y0, y1 = int(y * picture.height / height), int((y + 1) * picture.height / height)
            if x1 > x0 and y1 > y0 and peak > 0:
                alpha = round(155 * float(magnitude[y, x]) / peak)
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(255, 99, 40, alpha))
    name = stage / "heatmaps" / condition / f"{image_id}-{stage.name}"
    buf = io.BytesIO()
    picture.save(buf, format="PNG")
    atomic_write(name.with_suffix(".png"), buf.getvalue())
    write_json(name.with_suffix(".json"), {"grid": grid, "rms": magnitude.tolist(), "max_rms": peak})


def write_report(output, manifest, rows, stats, examples, config):
    labels = {**LABELS, **{k: v for k, v in EXTRA_LABELS.items() if any(r["arm"] == k for r in rows)}}
    pairs_to_report = [*PAIRS, *[("A0", arm) for arm in EXTRA_LABELS if arm in labels],
                       *[(arm, "S1b") for arm in EXTRA_LABELS if arm in labels]]
    numeric = {
        q["question_id"] for image in manifest["images"] for q in image["questions"]
        if re.search(r"\d", Counter(q["answers"]).most_common(1)[0][0])
    }
    comparisons, accuracies, feature_regressions = [], [], []
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        arms = {arm: [r for r in subset if r["arm"] == arm] for arm in labels}
        for arm, predictions in arms.items():
            images = {r["image_id"]: r for r in predictions}
            accuracies.append({
                "condition": condition, "arm": arm, "questions": len(predictions),
                "accuracy_pct": 100 * mean(r["score"] for r in predictions),
                "mean_generation_seconds_per_question": mean(r["generation_seconds"] for r in predictions),
                "mean_adapter_seconds_per_image": mean(r["adapter_seconds_per_image"] for r in images.values()),
                "median_generation_seconds_per_question": float(
                    np.median([r["generation_seconds"] for r in predictions])
                ),
                "p95_generation_seconds_per_question": float(
                    np.percentile([r["generation_seconds"] for r in predictions], 95)
                ),
                "mean_feature_mse_to_clean": mean(r["feature_mse_to_clean"] for r in images.values()),
                "mean_feature_cosine_to_clean": mean(r["feature_cosine_to_clean"] for r in images.values()),
            })
        for left, right in pairs_to_report:
            reference, target = arms[left], arms[right]
            pairs = list(zip(reference, target, strict=True))
            if any((a["image_id"], a["question_id"]) != (b["image_id"], b["question_id"]) for a, b in pairs):
                raise ValueError(f"Unaligned S1 predictions: {condition}/{left}/{right}")
            if left == "A0" and right in ("S1a", "S1b", "A1"):
                feature_regressions.extend({
                    "condition": condition, "arm": right, "image_id": a["image_id"], "question_id": a["question_id"],
                    "baseline_answer": a["answer"], "target_answer": b["answer"],
                    "baseline_score": a["score"], "target_score": b["score"],
                    "baseline_mse": a["feature_mse_to_clean"], "target_mse": b["feature_mse_to_clean"],
                    "baseline_cosine": a["feature_cosine_to_clean"], "target_cosine": b["feature_cosine_to_clean"],
                } for a, b in pairs if b["score"] < a["score"] and b["feature_mse_to_clean"] < a["feature_mse_to_clean"])
            for slice_name in ("all", "numeric_or_date"):
                selected = pairs if slice_name == "all" else [(a, b) for a, b in pairs if a["question_id"] in numeric]
                if not selected:
                    continue
                a = [ref["score"] for ref, _ in selected]
                b = [pred["score"] for _, pred in selected]
                groups = [ref["image_id"] for ref, _ in selected]
                comparisons.append({
                    "condition": condition, "slice": slice_name, "reference": left, "target": right, "questions": len(a),
                    "images": len(set(groups)), "reference_accuracy_pct": 100 * mean(a),
                    "target_accuracy_pct": 100 * mean(b),
                    **paired_bootstrap(a, b, groups, config["bootstrap_resamples"], config["seed"]),
                    "score_improved": sum(y > x for x, y in zip(a, b)),
                    "score_worsened": sum(y < x for x, y in zip(a, b)),
                    "fully_correct_to_lower": sum(x == 1 and y < 1 for x, y in zip(a, b)),
                    "lower_to_fully_correct": sum(x < 1 and y == 1 for x, y in zip(a, b)),
                })
        write_case_review(
            output / ("review.html" if condition == "LR4" else "review-clean.html"),
            subset, labels, title=f"S1 · {condition} · {output.parent.name}", manifest=manifest,
            repeats=config["bootstrap_resamples"], seed=config["seed"],
            image_overlays=(p for arm in labels for p in
                            (output / "variants" / arm / "heatmaps" / condition).glob("*.png")),
        )
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)
    atomic_write(output / "summary.csv", table.getvalue().encode("utf-8"))
    write_json(output / "summary.json", {
        "metric": "TextVQA MMF soft accuracy", "scorer_revision": SCORER_REVISION, "scorer_sha256": SCORER_SHA256,
        "bootstrap_cluster": "image_id", "bootstrap_resamples": config["bootstrap_resamples"],
        "arm_labels": labels, "comparisons": comparisons, "accuracies": accuracies, "runtime": stats,
        "training_questions_by_condition": dict(Counter(e[2] for e in examples)),
    })
    write_interactions(output, rows, pairs_to_report, config)
    history = io.StringIO()
    writer = csv.DictWriter(history, fieldnames=["arm", "step", "examples", "loss", "grad_norm", "seconds"],
                            extrasaction="ignore")
    writer.writeheader()
    for arm, runtime in stats.items():
        writer.writerows({"arm": arm, "step": step, **entry}
                         for step, entry in enumerate(runtime.get("history", []), 1))
    atomic_write(output / "training_history.csv", history.getvalue().encode("utf-8"))
    write_jsonl(output / "predictions.jsonl", rows)
    write_json(output / "feature_regressions.json", feature_regressions)



def degradation_interactions(rows, pairs, repeats, seed):
    indexed = {}
    for row in rows:
        key = row["condition"], row["arm"], row["image_id"], row["question_id"]
        if key in indexed:
            raise ValueError(f"Duplicate degradation-interaction row: {key}")
        indexed[key] = row["score"]
    cases = sorted({(r["image_id"], r["question_id"]) for r in rows})
    results = []
    for reference, target in pairs:
        deltas = []
        for image_id, qid in cases:
            scores = {(c, a): indexed[c, a, image_id, qid] for c in CONDITIONS for a in (reference, target)}
            deltas.append((scores["LR4", target] - scores["LR4", reference]) -
                          (scores["CLEAN", target] - scores["CLEAN", reference]))
        results.append({"reference": reference, "target": target, "questions": len(cases),
                        **paired_bootstrap(np.zeros(len(deltas)), deltas, [c[0] for c in cases], repeats, seed)})
    return results


def write_interactions(output, rows, pairs, config):
    values = degradation_interactions(rows, pairs, config["bootstrap_resamples"], config["seed"])
    write_json(output / "degradation_interaction.json", values)
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(values[0]))
    writer.writeheader()
    writer.writerows(values)
    atomic_write(output / "degradation_interaction.csv", table.getvalue().encode("utf-8"))
