"""Compare SD-RPN and DBNet crops on the same LR4 questions with a locked A2 reader."""

import hashlib
import time
from collections import Counter
from statistics import mean

import torch

from qlrg.common import read_json
from qlrg.data import candidate, get_words
from qlrg.evaluation import SCORER_REVISION, paired_bootstrap
from qlrg.experiment import paired_images
from qlrg.grid_alignment import overlap_area
from scripts.crop_fusion import TOTAL_TOKENS, VIEW_TOKENS
from scripts.dbnet_crop import area, best_text_box, hybrid_boxes
from scripts.fusion_confirm import random_crop_box, read_a2_views
from scripts.query_crop import PADDING, padded_box


LABELS = {
    "full_A2": "A2 · Toàn ảnh",
    "sdrpn": "A2 · Toàn ảnh + crop SD-RPN theo query",
    "dbnet": "A2 · Toàn ảnh + crop DBNet theo độ tin cậy",
    "hybrid_union": "A2 · Toàn ảnh + box bao SD-RPN và DBNet",
    "hybrid_text": "A2 · Toàn ảnh + trọn box DBNet do SD-RPN chọn",
    "hybrid_intersection": "A2 · Toàn ảnh + crop phần giao SD-RPN và DBNet",
    "hybrid_random": "A2 · Toàn ảnh + crop ngẫu nhiên cùng kích thước box bao",
    "hybrid_shuffled": "A2 · Toàn ảnh + box bao với query tráo",
}
PAIRS = (
    *(("full_A2", arm) for arm in LABELS if arm != "full_A2"),
    ("sdrpn", "dbnet"), ("sdrpn", "hybrid_union"), ("dbnet", "hybrid_union"),
    ("hybrid_text", "hybrid_union"), ("hybrid_intersection", "hybrid_text"),
    ("hybrid_intersection", "hybrid_union"),
    ("hybrid_random", "hybrid_union"), ("hybrid_shuffled", "hybrid_union"),
)


def crop_boxes(size, iid, qid, seed, vlm, shuffled, candidates):
    def pad(box):
        return padded_box(box, size, PADDING["pad20"]) if box is not None else None

    match, fraction, union, text, shared = hybrid_boxes(vlm, candidates)
    donor_match, donor_fraction, donor_union, _, _ = hybrid_boxes(shuffled, candidates)
    union_box = pad(union)
    random_box = random_crop_box(union_box, size, seed, iid, qid) if union_box is not None else None
    boxes = {
        "full_A2": None, "sdrpn": pad(vlm), "dbnet": pad(best_text_box(candidates)),
        "hybrid_union": union_box, "hybrid_text": pad(text), "hybrid_intersection": pad(shared),
        "hybrid_random": random_box, "hybrid_shuffled": pad(donor_union),
    }
    return boxes, (match, fraction), (donor_match, donor_fraction)


@torch.inference_mode()
def predict_image(model, lora, detector, scorer, image, proposals, donors, root, output, seed, check_identity):
    _, lr4 = paired_images(root, image)
    iid = image["image_id"]
    if (proposals["image_id"] != iid or hashlib.sha256(lr4.tobytes()).hexdigest() != proposals["lr4_sha256"]
        or set(proposals["questions"]) != {str(q["question_id"]) for q in image["questions"]}):
        raise ValueError(f"SD-RPN used different LR4 pixels or questions for {iid}")

    start = time.perf_counter()
    candidates = detector.detect(lr4)
    dbnet_seconds = time.perf_counter() - start
    rows, identity = [], None
    for question in image["questions"]:
        qid = question["question_id"]
        query, shuffled = (proposals["questions"][str(qid)][route] for route in ("query", "shuffled"))
        start = time.perf_counter()
        boxes, (match, fraction), (donor_match, donor_fraction) = crop_boxes(
            lr4.size, iid, qid, seed, query["box"], shuffled["box"], candidates)
        routing_seconds = time.perf_counter() - start
        refs = scorer._compute_answer_scores(question["answers"])
        readers = {}
        for arm, box in boxes.items():
            donor = arm == "hybrid_shuffled"
            hybrid = arm.startswith("hybrid_")
            region = shuffled if donor else query
            chosen, overlap = (donor_match, donor_fraction) if donor else (match, fraction)
            if box not in readers:
                key = [iid, "full_A2"] if box is None else [iid, qid, box]
                readers[box], gate = read_a2_views(model, lora, scorer, question, refs, lr4, box, output, key,
                                                   check_identity and identity is None)
                identity = identity or gate
            reader = readers[box]
            sd_seconds = region["selector_seconds"] if hybrid or arm == "sdrpn" else 0.0
            det_seconds = dbnet_seconds if hybrid or arm == "dbnet" else 0.0
            selector_seconds = sd_seconds + det_seconds + (routing_seconds if hybrid else 0.0)
            rows.append({
                **reader, "arm": arm, "image_id": iid, "question_id": qid, "roi_selected": box is not None,
                "crop_area_fraction": area(box) / (lr4.width * lr4.height) if box is not None else None,
                "sdrpn_xyxy": region["box"] if hybrid or arm == "sdrpn" else None,
                "dbnet_xyxy": chosen if hybrid else best_text_box(candidates) if arm == "dbnet" else None,
                "dbnet_overlap_fraction": overlap if hybrid else None,
                "dbnet_candidates": len(candidates), "hybrid_matched": chosen is not None if hybrid else None,
                "selector_donor_qid": donors[qid][0] if donor else None,
                "random_box_moved": box != boxes["hybrid_union"] if arm == "hybrid_random" else None,
                "selector_seconds": selector_seconds, "pipeline_seconds": selector_seconds + reader["reader_seconds"],
            })
    return {"rows": rows, "identity": identity, "selector_seconds": proposals["selector_seconds"],
            "dbnet_seconds": dbnet_seconds}


def answer_polygons(manifest, path):
    """Evaluation-only grounding with an independent cohort of unique literal matches."""
    ocr = read_json(path)
    polygons = {}
    for image in manifest["images"]:
        iid = image["image_id"]
        meta = ocr["imgs"].get(iid)
        if meta is None or image["size"] != [meta["width"], meta["height"]]:
            continue
        words = get_words(ocr, iid)
        by_id = {word["id"]: word["polygon"] for word in words}
        for question in image["questions"]:
            match = candidate(question, words, minimum_votes=8)
            if match:
                polygon = by_id[match["word_ids"][0]]
                if polygon_area(polygon) > 0:
                    polygons[question["question_id"]] = polygon
    return polygons


def polygon_area(points):
    return abs(sum(x * v - u * y for (x, y), (u, v) in zip(points, points[1:] + points[:1]))) / 2


def fraction_retained(box, polygon):
    left, top, right, bottom = box
    points = [((x - left) / (right - left), (y - top) / (bottom - top)) for x, y in polygon]
    return min(1.0, overlap_area(points, 0, 0) / polygon_area(points))


def summarize(rows, manifest, seed, identity, selector_seconds, dbnet_seconds, textocr):
    expected = [(im["image_id"], q["question_id"]) for im in manifest["images"] for q in im["questions"]]
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in LABELS}
    if not identity or not identity["logits_exact"]:
        raise ValueError("Missing two-image A2 identity check")
    if (len(rows) != len(expected) * len(by_arm)
        or any(r["visual_tokens"] != TOTAL_TOKENS
               or r["view_tokens"] != ([VIEW_TOKENS] * 2 if r["roi_selected"] else [TOTAL_TOKENS]) for r in rows)
        or any([(r["image_id"], r["question_id"]) for r in rs] != expected for rs in by_arm.values())):
        raise ValueError("Crop benchmark changed the cohort or visual token budget")

    def comparisons(pairs, indices):
        groups = [expected[i][0] for i in indices]
        scores = {arm: [rs[i]["score"] for i in indices] for arm, rs in by_arm.items()}
        return [{"reference": a, "target": b, "questions": len(indices), "images": len(set(groups)),
                 **paired_bootstrap(scores[a], scores[b], groups, 10000, seed)} for a, b in pairs]

    polygons = answer_polygons(manifest, textocr)
    for row in rows:
        polygon = polygons.get(row["question_id"])
        if polygon is not None and row["roi_selected"]:
            row["answer_region_retained"] = fraction_retained(row["crop_xyxy"], polygon)
    arms = {}
    for arm, rs in by_arm.items():
        crops = [r for r in rs if r["roi_selected"]]
        retained = [r["answer_region_retained"] for r in crops if "answer_region_retained" in r]
        arms[arm] = {
            "label": LABELS[arm], "accuracy_pct": 100 * mean(r["score"] for r in rs),
            "mean_visual_tokens": TOTAL_TOKENS, "mean_prompt_tokens": mean(r["prompt_tokens"] for r in rs),
            "crop_questions": len(crops), "full_fallback_questions": len(rs) - len(crops) if arm != "full_A2" else 0,
            "hybrid_match_questions": sum(r["hybrid_matched"] is True for r in rs),
            "mean_crop_area_fraction": mean(r["crop_area_fraction"] for r in crops) if crops else None,
            "mean_reader_seconds": mean(r["reader_seconds"] for r in rs),
            "mean_selector_seconds": mean(r["selector_seconds"] for r in rs),
            "mean_pipeline_seconds": mean(r["pipeline_seconds"] for r in rs),
            "grounded_crop_questions": len(retained), "grounded_crop_retains_90pct": sum(x >= 0.90 for x in retained),
            "mean_answer_region_retained": mean(retained) if retained else None,
        }
    moved = [i for i, r in enumerate(by_arm["hybrid_random"]) if r["random_box_moved"]]
    matched = [i for i, r in enumerate(by_arm["hybrid_union"]) if r["hybrid_matched"]]
    jointly_grounded_crops = [i for i, (_, qid) in enumerate(expected)
                              if qid in polygons and all(by_arm[arm][i]["roi_selected"] for arm in LABELS if arm != "full_A2")]
    return {
        "split": manifest["split"], "images": len(manifest["images"]), "questions": len(expected),
        "visual_tokens": TOTAL_TOKENS, "fusion_view_tokens": [VIEW_TOKENS, VIEW_TOKENS],
        "crop_padding_per_side": PADDING["pad20"], "seed": seed, "scorer_revision": SCORER_REVISION,
        "bootstrap_unit": "image_id", "bootstrap_resamples": 10000, "identity": identity,
        "sdrpn_seconds_total": selector_seconds, "dbnet_seconds_total": dbnet_seconds,
        "grounding": {"policy": "provisional_unique_TextOCR_literal_at_least_8_of_10", "questions": len(polygons),
                      "common_crop_questions": len(jointly_grounded_crops),
                      "common_retains_90pct": {arm: sum(by_arm[arm][i]["answer_region_retained"] >= 0.90
                                                        for i in jointly_grounded_crops)
                                               for arm in LABELS if arm != "full_A2"}},
        "arms": arms, "comparisons": comparisons(PAIRS, list(range(len(expected)))),
        "matched_hybrid_comparisons": comparisons(
            (("hybrid_text", "hybrid_union"), ("hybrid_intersection", "hybrid_union")), matched) if matched else [],
        "random_moved_comparisons": comparisons((("hybrid_random", "hybrid_union"),), moved) if moved else [],
        "sdrpn_and_dbnet_matches": dict(Counter(
            "both" if u["hybrid_matched"] and s["hybrid_matched"] else "union_only" if u["hybrid_matched"]
            else "shuffled_only" if s["hybrid_matched"] else "neither"
            for u, s in zip(by_arm["hybrid_union"], by_arm["hybrid_shuffled"]))),
    }
