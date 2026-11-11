"""Evaluate locked full-image S1 checkpoints on LR4, Blur, and LR4 then Blur."""

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from filelock import FileLock
from PIL import ImageFilter

from qlrg.common import read_json, sha256, write_json
from qlrg.evaluation import SCORER_REVISION, evaluator, paired_bootstrap
from qlrg.experiment import paired_images
from qlrg.grid_alignment import image_grid
from qlrg.qwen_wrapper import Qwen
from qlrg.s1_report import EXTRA_LABELS, LABELS as S1_LABELS
from qlrg.s1_run import make_adapter
from scripts.followup_common import bind, finish, load_study, write_table


CONDITIONS = ("LR4", "BLUR", "LR4_BLUR")
BLUR_RADIUS = 4.0
LABELS = {**S1_LABELS, **EXTRA_LABELS, "S1b_A2": "S1b phục hồi visual feature + A2 LoRA ở LLM"}


def conditions_for(clean, lr4):
    blur = ImageFilter.GaussianBlur(radius=BLUR_RADIUS)
    return {"LR4": lr4, "BLUR": clean.filter(blur), "LR4_BLUR": lr4.filter(blur)}


@torch.inference_mode()
def predict(model, adapters, scorer, record, root):
    clean, lr4 = paired_images(root, record)
    rows = []
    for condition, image in conditions_for(clean, lr4).items():
        first = model.prepare(image, record["questions"][0]["question"])
        grid = image_grid(first["image_grid_thw"][0].tolist(), model.model.model.visual.spatial_merge_size)
        base = model.encode(first)
        features, seconds = {"base": base}, {"base": 0.0}
        for arm in ("S1a", "S1b", "A1", "A1clean"):
            torch.cuda.synchronize()
            start = time.perf_counter()
            restored, _ = adapters[arm](base, grid)
            features[arm] = restored.to(model.dtype)
            torch.cuda.synchronize()
            seconds[arm] = time.perf_counter() - start
        for index, question in enumerate(record["questions"]):
            batch = first if index == 0 else model.prepare(image, question["question"])
            if not torch.equal(batch["image_grid_thw"], first["image_grid_thw"]):
                raise ValueError("Question changed the visual grid")
            refs = scorer._compute_answer_scores(question["answers"])
            for arm in LABELS:
                feature_arm = "S1b" if arm == "S1b_A2" else arm if arm in features else "base"
                lora_arm = "A2" if arm == "S1b_A2" else arm
                attach = adapters[lora_arm].attached() if lora_arm in ("A2", "A2clean") else nullcontext()
                with attach:
                    result = model.generate(batch, features[feature_arm])
                rows.append({**result, "condition": condition, "arm": arm,
                             "image_id": record["image_id"], "question_id": question["question_id"],
                             "visual_tokens": grid[0] * grid[1], "adapter_seconds_per_image": seconds[feature_arm],
                             "score": refs.get(scorer.answer_processor(result["answer"]), 0.0)})
    return rows


def report(output, rows, manifest, seed):
    cases = [(image["image_id"], question["question_id"]) for image in manifest["images"]
             for question in image["questions"]]
    groups = [image for image, _ in cases]
    pairs = [("A0", arm) for arm in LABELS if arm != "A0"] + [
        ("A1", "S1b"), ("A2", "S1b"), ("A2", "S1b_A2")]
    indexed = {(c, a): [] for c in CONDITIONS for a in LABELS}
    for row in rows:
        indexed[row["condition"], row["arm"]].append(row)
    values, tables = {}, {}
    for condition in CONDITIONS:
        accuracies, comparisons = [], []
        for arm, label in LABELS.items():
            predictions = indexed[condition, arm]
            if [(r["image_id"], r["question_id"]) for r in predictions] != cases:
                raise ValueError(f"Mismatched official validation cases: {condition}/{arm}")
            values[condition, arm] = np.array([row["score"] for row in predictions])
            accuracy = {"arm": arm, "label": label, "questions": len(cases),
                        "accuracy_pct": 100 * mean(values[condition, arm]),
                        "mean_visual_tokens": mean(r["visual_tokens"] for r in predictions),
                        "mean_generation_ms": 1000 * mean(r["generation_seconds"] for r in predictions)}
            accuracies.append(accuracy)
        for left, right in pairs:
            a, b = values[condition, left], values[condition, right]
            comparisons.append({"reference": left, "target": right, "questions": len(cases),
                                **paired_bootstrap(a, b, groups, 10000, seed),
                                "improved": int(sum(b > a)), "worsened": int(sum(b < a))})
        write_table(output / f"accuracy_{condition.lower()}.csv", accuracies)
        write_table(output / f"paired_{condition.lower()}.csv", comparisons)
        tables[condition] = {"accuracies": accuracies, "comparisons": comparisons}
    return {"split": manifest["split"], "questions": len(cases), "images": len(manifest["images"]),
            "bootstrap_unit": "image_id", "bootstrap_resamples": 10000, "scorer_revision": SCORER_REVISION,
            "conditions": tables, "blur_radius_px": BLUR_RADIUS, "combined_order": "LR4 then GaussianBlur"}


def run(output):
    with FileLock(str(output / ".run.lock"), timeout=0):
        launch, paths, config, manifest, checkpoints, binding = load_study(
            output, "restoration", (arm for arm in LABELS if arm not in ("A0", "S1b_A2")))
        binding.update({"conditions": CONDITIONS, "blur_radius_px": BLUR_RADIUS,
                        "combined_order": "LR4 then GaussianBlur", "worker_sha256": sha256(Path(__file__))})
        binding["conditions"] = list(CONDITIONS)
        bind(output, binding)
        model = Qwen(config, paths["hf_hub"])
        adapters = {}
        for arm, record in checkpoints.items():
            adapter = make_adapter(model, arm, record["config"]).eval()
            adapter.load_state_dict(record["state"])
            adapters[arm] = adapter
        scorer = evaluator()
        rows = []
        for index, image in enumerate(manifest["images"], 1):
            shard = output / "shards" / f"{image['image_id']}.json"
            if shard.is_file():
                image_rows = read_json(shard)
            else:
                image_rows = predict(model, adapters, scorer, image, paths["textvqa"])
                write_json(shard, image_rows)
            rows.extend(image_rows)
            if index % 50 == 0 or index == len(manifest["images"]):
                print(f"Restoration evaluation: {index}/{len(manifest['images'])} images", flush=True)
        finish(output, binding, report(output, rows, manifest, launch["seed"]), rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
