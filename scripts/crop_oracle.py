"""Score matched full-image and oracle crops on the S0 development cohort."""

import math
import os
import random
from pathlib import Path
from statistics import mean

from filelock import FileLock
from PIL import Image

from qlrg.case_review import write_case_review
from qlrg.common import dataset_paths, object_hash, read_config, read_json, read_jsonl, sha256, write_json, write_jsonl
from qlrg.data import rank, verify_pillow
from qlrg.evaluation import SCORER_REVISION, evaluator, paired_bootstrap
from qlrg.experiment import paired_images, validate_manifest, verified_complete
from qlrg.grid_alignment import image_grid
from qlrg.grounding import provisional_grounding, read_review


LABELS = {
    "C0": "ảnh LR4 toàn cảnh",
    "C1": "crop vùng đáp án theo nhãn, zoom bicubic",
    "random": "crop ngẫu nhiên cùng kích thước và mức zoom",
}


def word_box(word, size):
    xs, ys = zip(*word["polygon"])
    return fit_box((min(xs), min(ys), max(xs), max(ys)), size)


def fit_box(box, size):
    width, height = size
    x0, y0, x1, y1 = box
    bounded = (max(0, math.floor(x0)), max(0, math.floor(y0)),
               min(width, math.ceil(x1)), min(height, math.ceil(y1)))
    if bounded[0] >= bounded[2] or bounded[1] >= bounded[3]:
        raise ValueError("Grounding has no pixels in the raw image canvas")
    return bounded


def answer_crop(words, size, padding):
    x0, y0, x1, y1 = zip(*words)
    left, top, right, bottom = min(x0), min(y0), max(x1), max(y1)
    dx, dy = (right - left) * padding, (bottom - top) * padding
    return fit_box((left - dx, top - dy, right + dx, bottom + dy), size)


def random_crop(box, size, seed):
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    columns, rows = size[0] - width + 1, size[1] - height + 1
    if columns * rows == 1:
        return box
    oracle_position = y0 * columns + x0
    position = random.Random(seed).randrange(columns * rows - 1)
    if position >= oracle_position:
        position += 1
    top, left = divmod(position, columns)
    return (left, top, left + width, top + height)


def covered_fraction(target, box):
    x0, y0, x1, y1 = target
    width = max(0, min(x1, box[2]) - max(x0, box[0]))
    height = max(0, min(y1, box[3]) - max(y0, box[1]))
    return width * height / ((x1 - x0) * (y1 - y0))


def read_detections(path):
    detections = {}
    for row in read_jsonl(path):
        image_id = row["image_id"]
        if image_id in detections or row["condition"] != "LR4":
            raise ValueError(f"Expected one LR4 detector record for {image_id}")
        detections[image_id] = row
    return detections


def recall_for_case(detections, image, targets):
    row = detections.get(image["image_id"])
    if row is None or row["image_sha256"] != image["image_sha256"]:
        raise ValueError(f"Missing or mismatched detector record for {image['image_id']}")
    width, height = image["size"]
    boxes = row["boxes"]
    for box in boxes:
        if (
            len(box) != 4 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in box)
            or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height
        ):
            raise ValueError(f"Invalid detector box for {image['image_id']}: {box}")
    return sum(any(covered_fraction(target, box) >= 0.5 for box in boxes) for target in targets)


def score_case(model, scorer, root, image, question, targets, padding, zoom, seed):
    _, degraded = paired_images(root, image)
    oracle_box = answer_crop(targets, degraded.size, padding)
    control_box = random_crop(oracle_box, degraded.size, int(rank(seed, question["question_id"])[:16], 16))
    boxes = {"C0": (0, 0, *degraded.size), "C1": oracle_box, "random": control_box}
    refs = scorer._compute_answer_scores(question["answers"])
    results = []
    for arm, box in boxes.items():
        rgb = degraded if arm == "C0" else degraded.crop(box)
        if arm != "C0" and zoom != 1:
            rgb = rgb.resize(tuple(max(1, round(n * zoom)) for n in rgb.size), Image.Resampling.BICUBIC)
        batch = model.prepare(rgb, question["question"])
        result = model.generate(batch)
        grid_thw = batch["image_grid_thw"][0].tolist()
        height, width = image_grid(grid_thw)
        results.append({
            "image_id": image["image_id"], "question_id": question["question_id"], "arm": arm,
            "crop_xyxy": box, "input_size": rgb.size, "grid_thw": grid_thw, "visual_tokens": height * width,
            "answer_region_coverage": mean(covered_fraction(target, box) for target in targets),
            "score": refs.get(scorer.answer_processor(result["answer"]), 0.0), **result,
        })
    return results


def build_report(rows, cases, repeats, seed):
    if len(rows) != len(cases) * len(LABELS):
        raise ValueError("Unexpected number of crop predictions")
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in LABELS}
    expected = [(image["image_id"], question["question_id"]) for image, question, _ in cases]
    for arm, arm_rows in by_arm.items():
        if [(r["image_id"], r["question_id"]) for r in arm_rows] != expected:
            raise ValueError(f"Incomplete or unpaired predictions for {arm}")
    scores = {arm: [row["score"] for row in arm_rows] for arm, arm_rows in by_arm.items()}
    groups = [image_id for image_id, _ in expected]
    comparisons = []
    for reference, target in (("C0", "C1"), ("C0", "random"), ("random", "C1")):
        a, b = scores[reference], scores[target]
        comparisons.append({
            "reference": reference, "target": target,
            **paired_bootstrap(a, b, groups, repeats, seed),
            "score_improved": sum(y > x for x, y in zip(a, b)),
            "score_worsened": sum(y < x for x, y in zip(a, b)),
        })
    return {
        "questions": len(expected), "images": len(set(groups)), "scorer_revision": SCORER_REVISION,
        "arms": {arm: {"accuracy_pct": 100 * mean(scores[arm]),
                       "mean_visual_tokens": mean(r["visual_tokens"] for r in arm_rows),
                       "mean_generation_seconds": mean(r["generation_seconds"] for r in arm_rows),
                       "mean_answer_region_coverage": mean(r["answer_region_coverage"] for r in arm_rows)}
                 for arm, arm_rows in by_arm.items()},
        "comparisons": comparisons,
    }


def select_cases(source, manifest, s0, review_path, limit):
    if review_path:
        review = read_review(review_path, manifest)
        selected = {qid for qid, row in review.items() if row["status"] == "accepted"}
        policy = "human_reviewed"
    else:
        if s0["grounding_policy"] != "provisional_ocr_150":
            raise ValueError("Provide --review for a source without the provisional 150-question policy")
        review = provisional_grounding(manifest)
        coverage = read_json(source / "s0" / "oracle_coverage.json")
        selected = {row["question_id"] for row in coverage["records"] if "mask_tokens" in row}
        if not selected <= review.keys():
            raise ValueError("S0 grounded questions do not match the provisional OCR policy")
        policy = "provisional_ocr_150"
    cases = []
    for image in manifest["images"]:
        for question in image["questions"]:
            qid = question["question_id"]
            if qid in selected:
                ids = set(review[qid]["word_ids"])
                words = [word_box(w, image["size"]) for w in image["words"] if w["id"] in ids]
                if len(words) != len(ids) or not words:
                    raise ValueError(f"Missing grounding words for question {qid}")
                cases.append((image, question, words))
    if not cases or len(cases) != len(selected):
        raise ValueError("Missing grounded cases in the source manifest")
    if limit:
        cases = sorted(cases, key=lambda case: rank(manifest["seed"], case[1]["question_id"]))[:limit]
    return cases, policy


def run(args):
    from qlrg.qwen_wrapper import Qwen
    from qlrg.s1_run import runtime_shared

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Crop evaluation must run inside a Slurm GPU allocation")
    verify_pillow()
    source = args.from_run
    manifest_path = source / "manifest.json"
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    s0 = verified_complete(source / "s0", "s0")
    paths = dataset_paths(read_config(args.paths))
    config = read_config(args.model_config)
    manifest_sha = sha256(manifest_path)
    if manifest_sha != s0["shared"]["manifest_sha256"] or config != s0["shared"]["model"]:
        raise ValueError("Crop run requires the same manifest and pinned Qwen model as the S0 source")
    cases, policy = select_cases(source, manifest, s0, args.review, args.limit)
    detection_rows = read_detections(args.detections) if args.detections else None
    recall = None
    if detection_rows is not None:
        hits = [recall_for_case(detection_rows, image, targets) for image, _, targets in cases]
        totals = [len(targets) for _, _, targets in cases]
        recall = {"word_hits": sum(hits), "word_total": sum(totals),
                  "all_words_covered_questions": sum(h == n for h, n in zip(hits, totals)),
                  "questions": len(cases), "minimum_word_box_coverage": 0.5}
    binding = {
        "stage": "crop_oracle", "s0_binding": object_hash(s0), "manifest_sha256": manifest_sha,
        "shared": runtime_shared(paths, config), "crop_source_sha256": sha256(Path(__file__)),
        "grounding_policy": policy, "seed": args.seed, "padding_fraction": args.padding, "zoom": args.zoom,
        "review_sha256": sha256(args.review) if args.review else None,
        "detections_sha256": sha256(args.detections) if args.detections else None,
        "question_ids": [question["question_id"] for _, question, _ in cases],
    }
    output = args.run_dir
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".crop.lock"), timeout=0):
        state = output / "run.json"
        if state.exists() and read_json(state) != binding:
            raise ValueError("Crop experiment changed; choose a new output directory")
        complete = output / "COMPLETE.json"
        artifacts = ("summary.json", "predictions.jsonl", "review.html")
        if complete.exists():
            record = read_json(complete)
            if record["binding"] != object_hash(binding) or any(
                sha256(output / name) != record["artifacts"].get(name) for name in artifacts
            ):
                raise ValueError("Completed crop experiment failed artifact verification")
            print(f"Verified completed crop oracle: {output}", flush=True)
            return
        write_json(state, binding)
        scorer = evaluator()
        model = Qwen(config, paths["hf_hub"])
        rows = []
        for index, (image, question, targets) in enumerate(cases):
            shard = output / "shards" / f"{index:06d}.json"
            if not shard.exists():
                write_json(shard, score_case(model, scorer, paths["textvqa"], image, question,
                                             targets, args.padding, args.zoom, args.seed))
            rows.extend(read_json(shard))
            if (index + 1) % 25 == 0 or index + 1 == len(cases):
                print(f"Crop oracle {index + 1}/{len(cases)} questions", flush=True)
        report = build_report(rows, cases, repeats=10000, seed=args.seed)
        report.update({"grounding_policy": policy, "detector_recall": recall, "padding_fraction": args.padding,
                       "bicubic_zoom": args.zoom, "oracle_uses_answer_region_labels": True})
        write_json(output / "summary.json", report)
        write_jsonl(output / "predictions.jsonl", rows)
        write_case_review(output / "review.html", rows, LABELS, title="Crop oracle · LR4",
                          manifest=manifest, grounding_policy=policy, seed=args.seed)
        write_json(complete, {"stage": "crop_oracle", "binding": object_hash(binding),
                              "artifacts": {name: sha256(output / name) for name in artifacts}})
    print(f"Crop oracle complete: {output / 'summary.json'}", flush=True)
