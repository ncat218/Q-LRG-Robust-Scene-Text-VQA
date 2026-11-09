"""Evaluate question-selected LR4 crops with a frozen image/text embedding model."""

import hashlib
import io
import math
import os
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean

import torch
from filelock import FileLock
from PIL import Image, ImageDraw, ImageOps
from torch.nn.functional import normalize

from qlrg.case_review import write_case_review
from qlrg.common import atomic_write, dataset_paths, object_hash, read_config, read_json, sha256, write_json, write_jsonl
from qlrg.data import rank, verify_pillow
from qlrg.evaluation import SCORER_REVISION, evaluator, paired_bootstrap
from qlrg.experiment import paired_images, verified_complete
from qlrg.grid_alignment import image_grid
from qlrg.s1_manifests import read_s1_manifests
from scripts.crop_replay import read_crops
from scripts.sdrpn_backend import run_selector, selector_provenance


SELECTOR_ID = "openai/clip-vit-base-patch32"
SELECTOR_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
SELECTOR_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
LABELS = {
    "full": "toàn ảnh LR4, Qwen gốc",
    "query": "crop theo query, Qwen gốc",
    "shuffled": "crop theo query tráo, trả lời query gốc",
    "full_restore": "toàn ảnh LR4 + phục hồi latent",
    "query_restore": "cùng crop theo query + phục hồi latent",
}
VARIANT_LABELS = {
    "S1a": "S1a (khớp feature sạch)",
    "S1b": "S1b (khớp feature sạch và học từ đáp án)",
    "A1": "A1 (visual adapter học từ đáp án)",
    "A2": "A2 (LoRA Q/V trong LLM, học từ đáp án)",
}
ABLATION_LABELS = {
    "full": "A2 trên toàn ảnh LR4",
    "query": "A2 trên crop SD-RPN, phóng theo cấu hình gốc",
    "nozoom": "A2 trên cùng crop SD-RPN, không phóng thêm",
    "pad10": "A2 trên crop SD-RPN nới 10% mỗi phía",
    "pad20": "A2 trên crop SD-RPN nới 20% mỗi phía",
    "fallback": "A2 với crop gốc hoặc toàn ảnh theo ngưỡng token",
}
PADDING = {"pad10": 0.1, "pad20": 0.2}
MIN_CROP_TOKENS = 25
STACK_ARM = "S1b_A2"


def candidate_boxes(size, fraction):
    """Generate the same set of equal-size windows for either query."""
    width, height = size
    cw, ch = max(1, round(width * fraction)), max(1, round(height * fraction))
    xs = sorted({round(i * (width - cw) / 3) for i in range(4)})
    ys = sorted({round(i * (height - ch) / 3) for i in range(4)})
    return [(x, y, x + cw, y + ch) for y in ys for x in xs]


def donor_questions(images, selected, seed):
    questions = sorted(
        ((im["image_id"], q["question_id"], q["question"]) for im in images for q in im["questions"]),
        key=lambda row: rank(seed, row[1]),
    )
    selected_ids = {q["question_id"] for im in selected for q in im["questions"]}
    donors = {}
    for i, (iid, qid, text) in enumerate(questions):
        if qid not in selected_ids:
            continue
        normalized = " ".join(text.casefold().split())
        for offset in range(1, len(questions)):
            other_image, other_id, other_text = questions[(i + offset) % len(questions)]
            if other_image != iid and " ".join(other_text.casefold().split()) != normalized:
                donors[qid] = (other_id, other_text)
                break
        else:
            raise ValueError(f"No different query from a different image for question {qid}")
    return donors


def selector_snapshot(cache):
    from huggingface_hub import snapshot_download

    snapshot = Path(cache) / "models--openai--clip-vit-base-patch32" / "snapshots" / SELECTOR_REVISION
    if all((snapshot / name).is_file() for name in SELECTOR_FILES):
        return snapshot
    try:
        downloaded = snapshot_download(SELECTOR_ID, revision=SELECTOR_REVISION, cache_dir=cache,
                                       allow_patterns=SELECTOR_FILES, max_workers=2)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Could not cache {SELECTOR_ID} in {cache}. "
                           "Prepare this selector cache where Hugging Face is reachable and retry.") from exc
    snapshot = Path(downloaded)
    if not all((snapshot / name).is_file() for name in SELECTOR_FILES):
        raise FileNotFoundError(f"Incomplete CLIP snapshot: {snapshot}")
    return snapshot


class ClipSelector:
    def __init__(self, snapshot, device):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
        self.model = CLIPModel.from_pretrained(
            snapshot, local_files_only=True, use_safetensors=False, weights_only=True, dtype=torch.float32,
        ).to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def rank(self, image, boxes, queries):
        # Pad to a square so CLIP sees the whole proposed box without stretching.
        crops = [ImageOps.pad(image.crop(box), (224, 224), Image.Resampling.BICUBIC,
                              color=(123, 117, 104)) for box in boxes]
        pixels = self.processor.image_processor(images=crops, return_tensors="pt")["pixel_values"].to(self.device)
        tokens = self.processor.tokenizer(list(queries), padding=True, truncation=True, return_tensors="pt")
        text = self.model.get_text_features(**tokens.to(self.device))
        visual = self.model.get_image_features(pixel_values=pixels)
        scores = normalize(text, dim=-1) @ normalize(visual, dim=-1).T
        return scores.cpu().tolist()


def save_preview(output, key, image, box=None, crop=None):
    path = Path("review-images") / f"{object_hash(key)[:24]}.jpg"
    preview = image.copy()
    preview.thumbnail((420, 360))
    if box is not None:
        x0, y0, x1, y1 = box
        sx, sy = preview.width / image.width, preview.height / image.height
        ImageDraw.Draw(preview).rectangle((x0 * sx, y0 * sy, x1 * sx - sx, y1 * sy - sy),
                                          outline="#6effbc", width=3)
        patch = crop.copy()
        patch.thumbnail((360, 360))
        panel = Image.new("RGB", (preview.width + patch.width + 8, max(preview.height, patch.height)), "#222222")
        panel.paste(preview, (0, 0))
        panel.paste(patch, (preview.width + 8, 0))
        preview = panel
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=84)
    atomic_write(output / path, buf.getvalue())
    return path.as_posix()


def select_box(boxes, scores):
    if len(scores) != len(boxes) or not all(math.isfinite(score) for score in scores):
        raise ValueError("Expected a finite similarity for every candidate box")
    index = max(range(len(boxes)), key=scores.__getitem__)
    return boxes[index], scores[index]


def padded_box(box, size, fraction):
    x0, y0, x1, y1 = box
    dx = math.ceil(fraction * (x1 - x0))
    dy = math.ceil(fraction * (y1 - y0))
    return max(0, x0 - dx), max(0, y0 - dy), min(size[0], x1 + dx), min(size[1], y1 + dy)


def fallback_reason(crop, full):
    if not crop["roi_selected"]:
        return "no_roi"
    if crop["visual_tokens"] <= MIN_CROP_TOKENS:
        return "tiny_crop"
    if crop["visual_tokens"] >= full["visual_tokens"]:
        return "no_token_saving"
    return None


@torch.inference_mode()
def predict_image(model, selector, adapters, scorer, image, donors, root, output, settings, proposals=None):
    _, lr4 = paired_images(root, image)
    iid = image["image_id"]
    questions = image["questions"]
    replay = bool(settings.get("from_crop_run"))
    ablation = bool(settings.get("crop_ablation"))
    stack = bool(settings.get("stack_s1b_a2"))
    if proposals is None:
        queries = [text for q in questions for text in (q["question"], donors[q["question_id"]][1])]
        boxes = candidate_boxes(lr4.size, settings["crop_fraction"])
        start = time.perf_counter()
        similarities = selector.rank(lr4, boxes, queries)
        selector_seconds = time.perf_counter() - start
        if len(similarities) != len(queries):
            raise ValueError("Selector returned the wrong number of query rankings")
        regions = {
            str(q["question_id"]): {
                route: dict(zip(("box", "clip_cosine"), select_box(boxes, similarities[2 * index + j])))
                for j, route in enumerate(("query", "shuffled"))
            }
            for index, q in enumerate(questions)
        }
    else:
        if proposals["image_id"] != iid or (
            not replay and proposals["lr4_sha256"] != hashlib.sha256(lr4.tobytes()).hexdigest()
        ):
            raise ValueError(f"SD-RPN and Q-LRG did not use the same LR4 pixels for {iid}")
        regions, selector_seconds = proposals["questions"], proposals["selector_seconds"]
        if regions.keys() != {str(q["question_id"]) for q in questions}:
            raise ValueError(f"SD-RPN proposal coverage differs for {iid}")
    full_preview = proposals["full_preview"] if replay else save_preview(output, [iid, "full"], lr4)
    full_box = (0, 0, *lr4.size)
    routes = ("full", "query") if replay else ("full", "query", "shuffled")
    if ablation:
        routes += ("nozoom", *PADDING)
    rows = []
    for question in questions:
        qid, text = question["question_id"], question["question"]
        selections = regions[str(qid)]
        native_results = {}
        restored_results = {}
        seen_inputs = {}
        ablation_rows = {}
        for route in routes:
            selection = {} if route == "full" else selections["query" if ablation else route]
            selected_box = selection.get("box")
            box = tuple(selected_box) if selected_box is not None else full_box
            if selected_box is None:
                pixels, preview = lr4, full_preview
            else:
                x0, y0, x1, y1 = box
                if not (0 <= x0 < x1 <= lr4.width and 0 <= y0 < y1 <= lr4.height):
                    raise ValueError(f"Out-of-bounds crop for question {qid}: {box}")
                if route in PADDING:
                    box = padded_box(box, lr4.size, PADDING[route])
                elif stack and route == "query":
                    box = padded_box(box, lr4.size, PADDING["pad20"])
                crop = lr4.crop(box)
                zoom = 1.0 if route == "nozoom" else settings["zoom"]
                scale = max(zoom, 28 / min(crop.size))
                pixels = crop.resize(tuple(max(28, round(n * scale)) for n in crop.size), Image.Resampling.BICUBIC)
                if replay and route == "query" and not stack:
                    preview = selection["preview"]
                else:
                    preview = save_preview(output, [qid, "pad20" if stack else route], lr4, box, pixels)
            reuse = None
            if route != "full" and selected_box is None:
                reuse = "full"
            elif route == "shuffled" and selected_box == selections["query"]["box"]:
                reuse = "query"
            if ablation and reuse is None:
                geometry = (box, pixels.size)
                reuse = seen_inputs.get(geometry)
                if reuse is None:
                    seen_inputs[geometry] = route
            if reuse:
                result, grid_thw, vision_seconds = native_results[reuse]
            else:
                batch = model.prepare(pixels, text)
                grid_thw = batch["image_grid_thw"][0].tolist()
                torch.cuda.synchronize()
                start = time.perf_counter()
                base = model.encode(batch)
                torch.cuda.synchronize()
                vision_seconds = time.perf_counter() - start
                result = None if ablation else model.generate(batch, base)
                native_results[route] = (result, grid_thw, vision_seconds)

            if replay and route == "query" and not stack and (
                list(pixels.size) != selection["input_size"] or grid_thw != selection["grid_thw"]
            ):
                raise ValueError(f"Replayed SD-RPN input changed for question {qid}")
            grid = image_grid(grid_thw)
            metadata = {
                "image_id": iid, "question_id": qid, "crop_xyxy": box, "input_size": pixels.size,
                "grid_thw": grid_thw, "visual_tokens": math.prod(grid), "preview": preview,
                "clip_cosine": selection.get("clip_cosine"), "vision_seconds": vision_seconds,
                "roi_selected": selected_box is not None if route != "full" else None,
                "foreground_tokens": selection.get("foreground_tokens"),
                "selector_donor_qid": donors[qid][0] if route == "shuffled" else None,
            }
            if not ablation:
                rows.append({"arm": route, **metadata, **result})
            if route == "shuffled":
                continue
            for variant, adapter in adapters.items():
                if reuse:
                    result, restore_seconds = restored_results[reuse, variant]
                else:
                    if variant == "A2":
                        with adapter.attached():
                            result = model.generate(batch, base)
                        restore_seconds = 0.0
                    else:
                        start = time.perf_counter()
                        restored, _ = adapter(base, grid)
                        if restored.shape != base.shape:
                            raise ValueError(f"Restoration changed the visual token count for {qid}/{route}")
                        restored = restored.to(model.dtype)
                        torch.cuda.synchronize()
                        restore_seconds = time.perf_counter() - start
                        result = model.generate(batch, restored)
                    restored_results[route, variant] = (result, restore_seconds)
                arm = route if ablation else f"{route}_{variant}"
                row = {"arm": arm, **metadata, **result, "restore_seconds": restore_seconds}
                rows.append(row)
                if ablation:
                    ablation_rows[route] = row
                if stack and variant == "S1b":
                    if reuse:
                        combined, restore_seconds = restored_results[reuse, STACK_ARM]
                    else:
                        # Reuse the same restored visual feature from the S1b-only arm.
                        with adapters["A2"].attached():
                            combined = model.generate(batch, restored)
                        restored_results[route, STACK_ARM] = (combined, restore_seconds)
                    rows.append({"arm": f"{route}_{STACK_ARM}", **metadata, **combined,
                                 "restore_seconds": restore_seconds})
        if ablation:
            reason = fallback_reason(ablation_rows["query"], ablation_rows["full"])
            chosen = ablation_rows["full" if reason else "query"]
            rows.append({**chosen, "arm": "fallback", "fallback_reason": reason})
    references = {q["question_id"]: scorer._compute_answer_scores(q["answers"]) for q in questions}
    for row in rows:
        row["score"] = references[row["question_id"]].get(scorer.answer_processor(row["answer"]), 0.0)
    return {"rows": rows, "selector_seconds": selector_seconds}


def summarize(rows, labels, manifest, selector_seconds, seed, ablation=False):
    images = manifest["images"]
    expected = [(im["image_id"], q["question_id"]) for im in images for q in im["questions"]]
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in labels}
    for arm, predictions in by_arm.items():
        if [(r["image_id"], r["question_id"]) for r in predictions] != expected:
            raise ValueError(f"Missing or mismatched predictions for {arm}")
    if len(rows) != len(expected) * len(labels):
        raise ValueError("Unexpected prediction arms")
    groups = [iid for iid, _ in expected]
    scores = {arm: [row["score"] for row in predictions] for arm, predictions in by_arm.items()}
    variants = [arm.removeprefix("query_") for arm in labels if arm.startswith("query_")]
    pairs = [("full", "query"), ("shuffled", "query")]
    if ablation:
        for target in ("nozoom", *PADDING, "fallback"):
            pairs.extend((("query", target), ("full", target)))
        pairs.append(("pad10", "pad20"))
        if any(a["crop_xyxy"] != b["crop_xyxy"] for a, b in zip(by_arm["query"], by_arm["nozoom"])):
            raise ValueError("The zoom control must use the original crop box")
    for variant in variants:
        full, crop = f"full_{variant}", f"query_{variant}"
        pairs.extend((("full", full), ("query", crop), (full, crop)))
        for raw, adapted in (("full", full), ("query", crop)):
            if any((x["crop_xyxy"], x["grid_thw"], x["visual_tokens"]) !=
                   (y["crop_xyxy"], y["grid_thw"], y["visual_tokens"])
                   for x, y in zip(by_arm[raw], by_arm[adapted])):
                raise ValueError(f"Input changed between {raw} and {adapted}")
    for route in ("full", "query"):
        pairs.extend(combinations((f"{route}_{variant}" for variant in variants), 2))
    pairs = [(a, b) for a, b in pairs if a in labels and b in labels]
    comparisons = [
        {"reference": a, "target": b, **paired_bootstrap(scores[a], scores[b], groups, 10000, seed),
         "score_improved": sum(y > x for x, y in zip(scores[a], scores[b])),
         "score_worsened": sum(y < x for x, y in zip(scores[a], scores[b]))}
        for a, b in pairs
    ]
    summary = {
        "split": manifest["split"], "images": len(images), "questions": len(expected),
        "scorer_revision": SCORER_REVISION, "selector_seconds_total": selector_seconds,
        "bootstrap_unit": "image_id", "bootstrap_resamples": 10000,
        "arms": {
            arm: {"label": labels[arm], "accuracy_pct": 100 * mean(scores[arm]),
                  "mean_visual_tokens": mean(row["visual_tokens"] for row in predictions),
                  "mean_vision_seconds": mean(row["vision_seconds"] for row in predictions),
                  "mean_generation_seconds": mean(row["generation_seconds"] for row in predictions),
                  "mean_restore_seconds": mean(row.get("restore_seconds", 0.0) for row in predictions)}
            for arm, predictions in by_arm.items()
        },
        "comparisons": comparisons,
    }
    if "shuffled" in by_arm:
        summary["same_query_and_shuffled_boxes"] = sum(
            a["crop_xyxy"] == b["crop_xyxy"] for a, b in zip(by_arm["query"], by_arm["shuffled"])
        )
    if variants or ablation:
        def gain_interactions(cohort_scores, cohort_groups):
            if not variants:
                return {}
            results = {}
            for variant in variants:
                full_gain = [y - x for x, y in zip(cohort_scores["full"], cohort_scores[f"full_{variant}"])]
                crop_gain = [y - x for x, y in zip(cohort_scores["query"], cohort_scores[f"query_{variant}"])]
                results[variant] = paired_bootstrap(full_gain, crop_gain, cohort_groups, 10000, seed)
            if variants == ["restore"]:
                return {"restoration_gain_crop_minus_full": results["restore"]}
            interactions = {"gain_crop_minus_full_by_checkpoint": results}
            if STACK_ARM in variants:
                interactions["s1b_a2_interaction_by_input"] = {}
                for route in ("full", "query"):
                    s1b_gain = [y - x for x, y in zip(cohort_scores[route], cohort_scores[f"{route}_S1b"])]
                    stacked_gain = [y - x for x, y in zip(cohort_scores[f"{route}_A2"],
                                                         cohort_scores[f"{route}_{STACK_ARM}"])]
                    interactions["s1b_a2_interaction_by_input"][route] = paired_bootstrap(
                        s1b_gain, stacked_gain, cohort_groups, 10000, seed)
            return interactions

        summary.update(gain_interactions(scores, groups))
        selected = [i for i, row in enumerate(by_arm["query"]) if row.get("roi_selected") is True]
        if selected:
            roi_groups = [groups[i] for i in selected]
            roi_scores = {arm: [values[i] for i in selected] for arm, values in scores.items()}
            summary["roi_only"] = {
                "questions": len(selected), "images": len(set(roi_groups)),
                "accuracy_pct": {arm: 100 * mean(values) for arm, values in roi_scores.items()},
                "mean_visual_tokens": {
                    arm: mean(values[i]["visual_tokens"] for i in selected) for arm, values in by_arm.items()
                },
                "comparisons": [
                    {"reference": a, "target": b, **paired_bootstrap(roi_scores[a], roi_scores[b], roi_groups, 10000, seed)}
                    for a, b in pairs
                ],
                **gain_interactions(roi_scores, roi_groups),
            }
    if ablation:
        summary["ablation"] = {
            "padding_per_side": PADDING, "fallback_tiny_max_tokens": MIN_CROP_TOKENS,
            "fallback_full_at_or_above_full_tokens": True,
            "fallback_reasons": dict(Counter(r["fallback_reason"] or "use_crop" for r in by_arm["fallback"])),
            "zoom_changed_input_size": sum(a["input_size"] != b["input_size"]
                                           for a, b in zip(by_arm["query"], by_arm["nozoom"])),
            "zoom_changed_grid": sum(a["grid_thw"] != b["grid_thw"]
                                     for a, b in zip(by_arm["query"], by_arm["nozoom"])),
        }
        whole_box = [
            i for i, (full, query) in enumerate(zip(by_arm["full"], by_arm["query"]))
            if query["roi_selected"] and full["crop_xyxy"] == query["crop_xyxy"]
        ]
        if whole_box:
            summary["whole_image_box_zoom_control"] = {
                "questions": len(whole_box),
                "accuracy_pct": {arm: 100 * mean(scores[arm][i] for i in whole_box)
                                 for arm in ("full", "query", "nozoom")},
                "query_vs_nozoom": paired_bootstrap(
                    [scores["query"][i] for i in whole_box], [scores["nozoom"][i] for i in whole_box],
                    [groups[i] for i in whole_box], 10000, seed,
                ),
            }
    return summary


def checkpoints_for(settings, shared, dataset_hashes):
    if settings["s1_run"] is None:
        return {}
    output = Path(settings["s1_run"]) / "s1"
    replay = bool(settings.get("from_crop_run"))
    if not replay or (output / "COMPLETE.json").is_file():
        trained = verified_complete(output, "s1")
    else:
        if settings["split"] != "val_inner":
            raise ValueError("A finalized arm before S1 completion may only be explored on val_inner")
        trained = read_json(output / "run.json")
        if trained["stage"] != "s1":
            raise ValueError("Expected an S1 training binding")
    # The crop checkout extends the HTML report. The audited inference and adapter
    # source, preprocessing, packages and model still have to match the S1 run.
    if any(trained["shared"].get(key) != value for key, value in shared.items() if key != "source_hash"):
        raise ValueError("S1 checkpoint model or audited inference environment differs")
    if trained.get("dataset_hashes") != dataset_hashes:
        raise ValueError("Use the expanded S1 run trained with the same split manifests")
    if replay:
        plan = read_json(output / "train_manifest.json")
        if object_hash(plan) != trained["train_manifest"]:
            raise ValueError("S1 training manifest does not match its binding")
        examples = sum(len(image["questions"]) for image in plan) * trained["config"]["epochs"]
        steps = math.ceil(examples / trained["config"]["accumulate"])
    fingerprint = object_hash(trained)
    checkpoints = {}
    for arm in settings.get("arms") or [settings["arm"]]:
        path = output / "variants" / arm / "checkpoint.pt"
        digest = sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["arm"] != arm or checkpoint["binding"] != fingerprint or sha256(path) != digest:
            raise ValueError(f"Checkpoint does not belong to the selected S1 run: {path}")
        if replay:
            finished = read_json(path.with_name("training.json"))
            history = checkpoint["history"]
            if (checkpoint["examples"] != examples or finished["examples"] != examples or finished["steps"] != steps
                or len(history) != steps or not history or history[-1]["examples"] != examples
                or finished["history"] != history):
                raise ValueError(f"{arm} does not have a verified final checkpoint")
        checkpoints[arm] = {"state": checkpoint["adapter"], "config": trained["config"],
                            "provenance": {"arm": arm, "sha256": digest, "s1_binding": fingerprint}}
    return checkpoints


def run(output):
    from qlrg.qwen_wrapper import Qwen
    from qlrg.s1_run import make_adapter, runtime_shared

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Query crop inference must run in a Slurm GPU allocation")
    verify_pillow()
    settings = read_json(output / "launch.json")
    repo = Path(settings["repo"])
    paths = dataset_paths(read_config(settings["paths"]))
    config = read_config(repo / "configs" / "model_qwen.yaml")
    datasets, hashes = read_s1_manifests(Path(settings["data_dir"]), paths["textvqa"])
    manifest = datasets[settings["split"]]
    images = manifest["images"]
    selected = images
    if settings["images"] is not None:
        selected = sorted(images, key=lambda row: rank(settings["seed"], row["image_id"]))[:settings["images"]]
    replay = settings.get("from_crop_run")
    ablation = bool(settings.get("crop_ablation"))
    stack = bool(settings.get("stack_s1b_a2"))
    fusion = bool(settings.get("fair_fusion"))
    donors = {} if replay else donor_questions(images, selected, settings["seed"])
    manifest = {**manifest, "images": selected}
    shared = runtime_shared(paths, config)
    checkpoints = checkpoints_for(settings, shared, hashes)
    multiple = bool(settings.get("arms"))
    if stack and (not replay or ablation or not multiple or set(checkpoints) != {"S1b", "A2"}):
        raise ValueError("Stack evaluation requires SD-RPN replay with S1b and A2")
    labels = {arm: LABELS[arm] for arm in ("full", "query")}
    if not replay:
        labels["shuffled"] = LABELS["shuffled"]
    if fusion:
        from scripts import crop_fusion

        if not replay or ablation or stack or not multiple or set(checkpoints) != {"S1b"}:
            raise ValueError("Fair fusion requires SD-RPN replay with S1b only")
        labels = crop_fusion.FUSION_LABELS
    elif ablation:
        if not replay or set(checkpoints) != {"A2"}:
            raise ValueError("Crop ablation requires SD-RPN replay with A2 only")
        labels = ABLATION_LABELS
    elif multiple:
        crop_label = "crop SD-RPN nới 20%" if stack else "cùng crop theo query"
        if stack:
            labels["query"] = f"{crop_label}, Qwen gốc"
        for arm in checkpoints:
            labels[f"full_{arm}"] = f"toàn ảnh LR4 + {VARIANT_LABELS[arm]}"
            labels[f"query_{arm}"] = f"{crop_label} + {VARIANT_LABELS[arm]}"
        if stack:
            labels[f"full_{STACK_ARM}"] = "toàn ảnh LR4 + S1b phục hồi latent + LoRA A2"
            labels[f"query_{STACK_ARM}"] = f"{crop_label} + S1b phục hồi latent + LoRA A2"
    elif checkpoints:
        labels.update({arm: LABELS[arm] for arm in ("full_restore", "query_restore")})
    use_sdrpn = settings.get("selector", "clip") == "sdrpn"
    if replay:
        if not checkpoints or not use_sdrpn:
            raise ValueError("SD-RPN crop replay requires a trained checkpoint")
        crops, selector_binding, replay_binding = read_crops(Path(replay), output, settings, shared, hashes, selected)
    elif use_sdrpn:
        selector_binding = selector_provenance(settings, repo)
    else:
        selector_binding = {"id": SELECTOR_ID, "revision": SELECTOR_REVISION}
    if fusion:
        original_question_count = sum(len(image["questions"]) for image in selected)
        selected = crop_fusion.select_fusion_cohort(selected, crops)
        omitted = original_question_count - sum(len(image["questions"]) for image in selected)
        manifest = {**manifest, "images": selected}
    binding = {
        "stage": "query_crop", "settings": settings, "shared": shared, "dataset_hashes": hashes,
        "selected_images": [image["image_id"] for image in selected],
        "selector": selector_binding,
        "crop_code_sha256": sha256(Path(__file__)),
        "report_template_sha256": sha256(repo / "qlrg" / "case_review.html"),
        "checkpoint": checkpoints[settings["arm"]]["provenance"] if checkpoints and not multiple else None,
    }
    if multiple:
        binding["checkpoints"] = {arm: checkpoint["provenance"] for arm, checkpoint in checkpoints.items()}
    if replay:
        binding["replay"] = replay_binding
    if fusion:
        binding["fusion_code_sha256"] = sha256(Path(crop_fusion.__file__))
    fingerprint = object_hash(binding)
    with FileLock(str(output / ".query-crop.lock"), timeout=0):
        state = output / "run.json"
        if state.exists() and read_json(state) != binding:
            raise ValueError("Run inputs changed. Resume with the original code and inputs.")
        complete = output / "COMPLETE.json"
        if complete.exists():
            raise ValueError(f"Run is already complete: {output}")
        write_json(state, binding)
        print(f"Query crop on {manifest['split']}: {len(selected)} images, {len(labels)} arms", flush=True)
        proposals = None
        if replay:
            print(f"Replaying locked SD-RPN crops from {replay}", flush=True)
        elif use_sdrpn:
            print(f"Generating boxes with {selector_binding['id']}", flush=True)
            proposals = run_selector(settings, repo, output, paths, config, selected, donors)
        else:
            print(f"Loading selector {SELECTOR_ID} at {SELECTOR_REVISION} from {settings['selector_cache']}", flush=True)
            snapshot = selector_snapshot(settings["selector_cache"])
        model_class = crop_fusion.FusionQwen if fusion else Qwen
        model = model_class(config, paths["hf_hub"])
        selector = None if use_sdrpn else ClipSelector(snapshot, model.device)
        adapters = {}
        for arm, checkpoint in checkpoints.items():
            adapter = make_adapter(model, arm, checkpoint["config"])
            adapter.load_state_dict(checkpoint["state"])
            adapter.eval()
            adapters[arm if multiple else "restore"] = adapter
        scorer = evaluator()
        rows, selector_seconds, identity = [], 0.0, None
        for index, image in enumerate(selected, start=1):
            shard = output / "shards" / f"{object_hash(image['image_id'])[:24]}.json"
            if shard.is_file():
                batch = read_json(shard)
                if any(not (output / row["preview"]).is_file() for row in batch["rows"]
                       if not replay or Path(row["preview"]).parent == Path("review-images")):
                    raise ValueError(f"Missing review preview for completed shard: {shard}")
            else:
                roi = crops[image["image_id"]] if replay else (read_json(proposals / shard.name) if proposals else None)
                if fusion:
                    batch = crop_fusion.predict_fusion_image(model, adapters["S1b"], scorer, image, roi,
                                                            paths["textvqa"], output, check_identity=index == 1)
                else:
                    batch = predict_image(model, selector, adapters, scorer, image, donors, paths["textvqa"], output,
                                          settings, proposals=roi)
                write_json(shard, batch)
            if batch.get("identity") is not None:
                identity = batch["identity"]
            rows.extend(batch["rows"])
            selector_seconds += batch["selector_seconds"]
            if index % 50 == 0 or index == len(selected):
                print(f"Query crop: {index}/{len(selected)} images", flush=True)
        if fusion:
            summary = crop_fusion.summarize_fusion(rows, manifest, settings["seed"], identity, omitted)
        else:
            summary = summarize(rows, labels, manifest, selector_seconds, settings["seed"], ablation=ablation)
        summary["checkpoint"] = binding["checkpoint"]
        if multiple:
            summary["checkpoints"] = binding["checkpoints"]
        summary["selector"] = selector_binding["id"]
        if replay:
            summary["replay"] = replay_binding
        if stack:
            summary["stack"] = {"restorer": "S1b", "language_lora": "A2",
                                "crop_padding_per_side": PADDING["pad20"], "crop_zoom": settings["zoom"]}
        summary["no_roi_fallbacks"] = {
            arm: sum(r["arm"] == arm and r["roi_selected"] is False for r in rows)
            for arm in ("query", "shuffled") if arm in labels
        }
        summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json(output / "summary.json", summary)
        write_jsonl(output / "predictions.jsonl", rows)
        previews = {f"{r['question_id']}-{r['arm']}": r["preview"] for r in rows}
        if replay:
            available = {path for path in set(previews.values()) if (output / path).is_file()}
            previews = {key: path for key, path in previews.items() if path in available}
        title = "A2 · Zoom, padding và fallback · LR4" if ablation else "Crop theo query · LR4"
        if stack:
            title = "S1b + A2 · Toàn ảnh và crop nới 20% · LR4"
        if fusion:
            title = "Toàn ảnh + crop · Ngân sách 576 visual token · LR4"
        write_case_review(output / "review.html", rows, labels, title=title,
                          manifest=manifest, seed=settings["seed"], crop_previews=previews)
        artifacts = ("summary.json", "predictions.jsonl", "review.html")
        write_json(complete, {"stage": "query_crop", "binding": fingerprint,
                              "artifacts": {name: sha256(output / name) for name in artifacts}})
    print(f"Query crop complete: {output / 'summary.json'}", flush=True)
