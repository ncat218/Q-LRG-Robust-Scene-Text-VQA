"""Locked official validation of A2 with query and size-matched random crops."""

import hashlib
import random
import time
from statistics import mean

import torch

from qlrg.common import object_hash, read_json
from qlrg.evaluation import SCORER_REVISION, paired_bootstrap
from qlrg.experiment import paired_images
from scripts.crop_fusion import TOTAL_TOKENS, VIEW_TOKENS, fit_token_canvas, save_views
from scripts.query_crop import PADDING, padded_box


CONFIRM_LABELS = {
    "full_A2": "A2 · toàn ảnh",
    "fusion_A2": "A2 · toàn ảnh + crop SD-RPN theo câu hỏi",
    "random_fusion_A2": "A2 · toàn ảnh + crop ngẫu nhiên cùng kích thước",
}


def validate_selector_source(source, current):
    done = read_json(source / "COMPLETE.json")
    original = read_json(source / "run.json")
    if done["stage"] != "query_crop" or done["binding"] != object_hash(original):
        raise ValueError("Invalid SD-RPN source binding")
    if original["settings"]["split"] != "val_inner" or original["settings"]["selector"] != "sdrpn":
        raise ValueError("Expected the locked SD-RPN selector from the development run")
    for key, val in current.items():
        if key not in ("backend_sha256", "worker_sha256") and original["selector"].get(key) != val:
            raise ValueError(f"SD-RPN checkpoint or runtime changed: {key}")
    return {"run": str(source), "binding": done["binding"]}


def random_crop_box(box, size, seed, image_id, question_id):
    """Uniform over all same-size placements except the original when possible."""
    x, y, right, bottom = box
    w, h = right - x, bottom - y
    across, down = size[0] - w + 1, size[1] - h + 1
    count = across * down
    if count == 1:
        return box
    source = y * across + x
    index = random.Random(object_hash([seed, image_id, str(question_id)])).randrange(count - 1)
    if index >= source:
        index += 1
    left, top = index % across, index // across
    return left, top, left + w, top + h


@torch.inference_mode()
def predict_confirmation_image(model, lora, scorer, image, proposals, root, output, seed, check_identity):
    _, lr4 = paired_images(root, image)
    iid = image["image_id"]
    if proposals["image_id"] != iid or hashlib.sha256(lr4.tobytes()).hexdigest() != proposals["lr4_sha256"]:
        raise ValueError(f"SD-RPN used different LR4 pixels for {iid}")
    if set(proposals["questions"]) != {str(q["question_id"]) for q in image["questions"]}:
        raise ValueError(f"SD-RPN question coverage changed for {iid}")
    rows, identity = [], None
    for question in image["questions"]:
        qid = question["question_id"]
        region = proposals["questions"][str(qid)]["query"]
        raw_box = region["box"]
        query_box = padded_box(raw_box, lr4.size, PADDING["pad20"]) if raw_box is not None else None
        random_box = random_crop_box(query_box, lr4.size, seed, iid, qid) if query_box is not None else None
        reader_rows = {}
        refs = scorer._compute_answer_scores(question["answers"])
        for arm, box in (("full_A2", None), ("fusion_A2", query_box), ("random_fusion_A2", random_box)):
            includes_selector = arm != "full_A2"
            if includes_selector and (box is None or (arm == "random_fusion_A2" and random_box == query_box)):
                source = reader_rows["full_A2" if box is None else "fusion_A2"]
                result = {**source, "arm": arm}
            else:
                start = time.perf_counter()
                budget = TOTAL_TOKENS if box is None else VIEW_TOKENS
                images = (lr4,) if box is None else (lr4, lr4.crop(box))
                views = [fit_token_canvas(im, budget)[0] for im in images]
                batch, grids = model.prepare_views(views, question["question"])
                features = model.encode_views(batch)
                # Logit parity runs once and is excluded from the reader latency.
                gate_seconds = 0.0
                if check_identity and box is not None and identity is None:
                    torch.cuda.synchronize()
                    gate_start = time.perf_counter()
                    with lora.attached():
                        native = model.prefill_logits(batch)
                        injected = model.prefill_logits(batch, features)
                    if not torch.equal(native, injected):
                        raise RuntimeError("Two-view A2 identity injection changed the logits")
                    identity = {"question_id": qid, "logits_exact": True, "visual_tokens": TOTAL_TOKENS}
                    gate_seconds = time.perf_counter() - gate_start
                with lora.attached():
                    result = model.generate(batch, features)
                reader_seconds = time.perf_counter() - start - gate_seconds
                preview = save_views(output, [iid, "full_A2"] if box is None else [iid, qid, arm], views)
                result = {
                    **result, "arm": arm, "image_id": iid, "question_id": qid, "crop_xyxy": box,
                    "visual_tokens": sum(h * w for h, w in grids), "prompt_tokens": batch["input_ids"].shape[1],
                    "view_tokens": [h * w for h, w in grids], "reader_seconds": reader_seconds, "preview": preview,
                    "score": refs.get(scorer.answer_processor(result["answer"]), 0.0),
                }
            selector_seconds = region["selector_seconds"] if includes_selector else 0.0
            result.update({
                "roi_selected": query_box is not None, "random_box_moved": random_box != query_box,
                "selector_seconds": selector_seconds, "pipeline_seconds": selector_seconds + result["reader_seconds"],
            })
            reader_rows[arm] = result
        rows.extend(reader_rows.values())
    return {"rows": rows, "identity": identity, "selector_seconds": proposals["selector_seconds"]}


def summarize_confirmation(rows, manifest, seed, identity, selector_seconds):
    images = manifest["images"]
    expected = [(im["image_id"], q["question_id"]) for im in images for q in im["questions"]]
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in CONFIRM_LABELS}
    if not identity or not identity["logits_exact"]:
        raise ValueError("Missing two-image A2 identity check")
    if (len(rows) != len(expected) * len(by_arm) or any(r["visual_tokens"] != TOTAL_TOKENS for r in rows)
        or any([(r["image_id"], r["question_id"]) for r in rs] != expected for rs in by_arm.values())):
        raise ValueError("Official confirmation changed the cohort or visual token budget")

    def compare(indices):
        groups = [expected[i][0] for i in indices]
        scores = {arm: [rs[i]["score"] for i in indices] for arm, rs in by_arm.items()}
        pairs = (("full_A2", "fusion_A2"), ("full_A2", "random_fusion_A2"), ("random_fusion_A2", "fusion_A2"))
        return {"questions": len(indices), "images": len(set(groups)),
                "accuracy_pct": {arm: 100 * mean(values) for arm, values in scores.items()},
                "comparisons": [{"reference": a, "target": b, **paired_bootstrap(scores[a], scores[b], groups, 10000, seed)}
                                for a, b in pairs]}

    report = compare(list(range(len(expected))))
    roi = [i for i, r in enumerate(by_arm["fusion_A2"]) if r["roi_selected"]]
    moved = [i for i in roi if by_arm["random_fusion_A2"][i]["random_box_moved"]]
    report.update({
        "split": manifest["split"], "visual_tokens": TOTAL_TOKENS, "fusion_view_tokens": [VIEW_TOKENS, VIEW_TOKENS],
        "crop_padding_per_side": PADDING["pad20"], "seed": seed, "scorer_revision": SCORER_REVISION,
        "identity": identity, "fallback_questions": len(expected) - len(roi), "random_box_unmoved": len(roi) - len(moved),
        "selector_seconds_total": selector_seconds, "bootstrap_unit": "image_id", "bootstrap_resamples": 10000,
        "arms": {
            arm: {"label": CONFIRM_LABELS[arm], "accuracy_pct": report["accuracy_pct"][arm],
                  "mean_visual_tokens": TOTAL_TOKENS, "mean_prompt_tokens": mean(r["prompt_tokens"] for r in rs),
                  "mean_reader_seconds": mean(r["reader_seconds"] for r in rs),
                  "mean_selector_seconds": mean(r["selector_seconds"] for r in rs),
                  "mean_pipeline_seconds": mean(r["pipeline_seconds"] for r in rs)}
            for arm, rs in by_arm.items()
        },
    })
    if roi:
        report["roi_only"] = compare(roi)
    if moved:
        report["random_moved_only"] = compare(moved)
    return report
