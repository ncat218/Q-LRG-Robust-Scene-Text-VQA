"""Recover fixed SD-RPN crops from verified, completed predictions."""

import os
from pathlib import Path

from qlrg.common import object_hash, read_json, read_jsonl, sha256


def read_crops(source, output, settings, shared, hashes, images):
    source_binding = read_json(source / "run.json")
    complete = read_json(source / "COMPLETE.json")
    if complete["stage"] != "query_crop" or complete["binding"] != object_hash(source_binding):
        raise ValueError("SD-RPN source completion does not match its run binding")
    predictions = source / "predictions.jsonl"
    digest = sha256(predictions)
    if digest != complete["artifacts"]["predictions.jsonl"]:
        raise ValueError("SD-RPN source predictions failed checksum verification")
    original = source_binding["settings"]
    if original.get("selector") != "sdrpn" or original["split"] != "val_inner" or source_binding["checkpoint"] is not None:
        raise ValueError("Expected an unrestored SD-RPN development baseline")
    if any(original[k] != settings[k] for k in ("split", "zoom", "seed", "images")):
        raise ValueError("SD-RPN replay changed the original crop settings")
    if source_binding["dataset_hashes"] != hashes or source_binding["selected_images"] != [im["image_id"] for im in images]:
        raise ValueError("SD-RPN replay requires the original image cohort and manifests")
    if any(source_binding["shared"].get(key) != value for key, value in shared.items() if key != "source_hash"):
        raise ValueError("SD-RPN replay requires the original Qwen runtime and LR4 preprocessing")

    expected = {
        (im["image_id"], str(q["question_id"])): tuple(im["size"])
        for im in images for q in im["questions"]
    }
    seen = set()
    crops = {im["image_id"]: {"image_id": im["image_id"], "questions": {}, "selector_seconds": 0.0} for im in images}
    for row in read_jsonl(predictions):
        arm = row["arm"]
        if arm not in ("full", "query"):
            continue
        iid, qid = row["image_id"], str(row["question_id"])
        if (iid, qid) not in expected or (iid, qid, arm) in seen:
            raise ValueError(f"Unexpected or duplicate SD-RPN prediction for {iid}/{qid}/{arm}")
        seen.add((iid, qid, arm))
        path = Path(row["preview"])
        if path.parent != Path("review-images") or path.suffix != ".jpg":
            raise ValueError(f"Unexpected SD-RPN preview path: {path}")
        preview = Path(os.path.relpath(source / path, output)).as_posix()
        if arm == "full":
            prior = crops[iid].setdefault("full_preview", preview)
            if prior != preview:
                raise ValueError(f"Inconsistent full preview for {iid}")
            continue
        width, height = expected[iid, qid]
        x0, y0, x1, y1 = row["crop_xyxy"]
        selected = row["roi_selected"]
        if type(selected) is not bool or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Invalid archived crop for {iid}/{qid}")
        if not selected and row["crop_xyxy"] != [0, 0, width, height]:
            raise ValueError(f"Invalid archived full-image fallback for {iid}/{qid}")
        crops[iid]["questions"][qid] = {"query": {
            "box": row["crop_xyxy"] if selected else None, "preview": preview,
            "foreground_tokens": row["foreground_tokens"], "input_size": row["input_size"], "grid_thw": row["grid_thw"],
        }}
    if len(seen) != 2 * len(expected):
        raise ValueError("SD-RPN predictions do not cover the replay cohort")
    return crops, source_binding["selector"], {
        "run": str(source), "binding": complete["binding"], "predictions_sha256": digest,
        "replay_code_sha256": sha256(Path(__file__)),
    }
