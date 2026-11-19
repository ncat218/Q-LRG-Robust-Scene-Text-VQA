"""Verify a completed four-arm evaluation using checksums and per-question scores."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(output):
    binding = json.loads((output / "run.json").read_text(encoding="utf-8"))
    marker = json.loads((output / "COMPLETE.json").read_text(encoding="utf-8"))
    canonical = json.dumps(binding, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    if marker["stage"] != "final_study" or hashlib.sha256(canonical).hexdigest() != marker["binding"]:
        raise ValueError("Invalid final evaluation binding")
    required = {"accuracy.csv", "paired.csv", "cost_summary.csv", "degradation_interaction.csv",
                "predictions.jsonl", "eval_manifest.json", "summary.json", "run.json", "identity.json",
                "error_cases.csv", "demo.html", "demo_cases.json", "review_clean.html", "review_lr4.html"}
    if not required <= marker["artifacts"].keys():
        raise ValueError("Completion marker is missing evaluation artifacts")
    for name, expected in marker["artifacts"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    manifest = json.loads((output / "eval_manifest.json").read_text(encoding="utf-8"))
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    if hashlib.sha256(manifest_bytes).hexdigest() != binding["manifest_hash"]:
        raise ValueError("Evaluation manifest differs from binding")
    cases = {(str(image["image_id"]), str(q["question_id"]))
             for image in manifest["images"] for q in image["questions"]}
    scores = defaultdict(dict)
    for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        group = row["condition"], row["arm"]
        key = str(row["image_id"]), str(row["question_id"])
        if key in scores[group] or not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
            raise ValueError(f"Invalid or duplicate prediction: {group}/{key}")
        scores[group][key] = row["score"]
    groups = {(condition, arm) for condition in ("CLEAN", "LR4") for arm in ("A0", "S1b", "Pre", "A2")}
    if scores.keys() != groups or any(values.keys() != cases for values in scores.values()):
        raise ValueError("Prediction cohort differs between conditions or arms")
    with (output / "accuracy.csv").open(encoding="utf-8", newline="") as handle:
        accuracy = list(csv.DictReader(handle))
    seen = set()
    for row in accuracy:
        group = row["condition"], row["arm"]
        if group not in groups or group in seen:
            raise ValueError("Invalid accuracy table groups")
        seen.add(group)
        measured = 100 * sum(scores[group].values()) / len(cases)
        if int(row["questions"]) != len(cases) or abs(float(row["accuracy_pct"]) - measured) > 1e-8:
            raise ValueError(f"Accuracy table does not match predictions: {group}")
    if seen != groups:
        raise ValueError("Incomplete accuracy table")
    print(f"Verified {len(cases)} questions, {len(manifest['images'])} images, 4 arms, 2 conditions")
    print(f"Verified {len(marker['artifacts'])} artifact checksums and re-aggregated accuracy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.output.resolve())
