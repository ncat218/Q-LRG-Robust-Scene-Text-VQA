"""S0 feature substitutions and coverage reports."""

import io
import os
from collections import Counter
from pathlib import Path

from PIL import ImageDraw
from filelock import FileLock
from safetensors import safe_open
from safetensors.torch import save
import torch

from .common import atomic_write, object_hash, write_json
from .data import rank
from .evaluation import oracle_decision_memo
from .feature_injection import replace_region
from .grid_alignment import dilate, image_grid, project, random_region
from .qwen_wrapper import assert_aligned
from .s1_storage import feature_cache_mode


def feature_pair(model, image, reference, base, output, shared):
    # Keep cache keys independent of subset size so matching prefixes can reuse features.
    key = object_hash({
        "image_sha256": image["image_sha256"], "grid": base["image_grid_thw"].tolist(),
        "protocol": "qlrg-post-reverse-indices-lr4-bicubic-v1",
        "model": shared["model"], "source_hash": shared["source_hash"],
        "packages": shared["packages"], "torch_build": shared["torch_build"],
        "gpu": torch.cuda.get_device_name(),
    })
    cache = Path(os.environ.get("QLRG_FEATURE_CACHE", output / "features"))
    path = cache / f"{key}.safetensors"
    h, w = image_grid(base["image_grid_thw"][0].tolist())

    def read_pair():
        with safe_open(path, framework="pt", device="cpu") as handle:
            if handle.metadata() != {"key": key}:
                raise ValueError(f"Feature cache metadata differs from the requested key: {path}")
            return {name: handle.get_tensor(name) for name in ("clean", "degraded")}

    if feature_cache_mode() == "read-only":
        try:
            loaded = read_pair()
        except FileNotFoundError:
            loaded = {"clean": model.encode(reference), "degraded": model.encode(base)}
    else:
        cache.mkdir(parents=True, exist_ok=True)
        with FileLock(str(cache / f"{key}.lock")):
            if not path.exists():
                features = {
                    "clean": model.encode(reference).cpu().contiguous(),
                    "degraded": model.encode(base).cpu().contiguous(),
                }
                atomic_write(path, save(features, metadata={"key": key}))
            loaded = read_pair()
    width = model.model.config.text_config.hidden_size
    for name in ("clean", "degraded"):
        if loaded[name].shape != (h * w, width) or loaded[name].dtype != model.dtype:
            raise ValueError(f"Invalid cached {name} features in {path}")
    return loaded["clean"].to(model.device), loaded["degraded"].to(model.device)


def overlay(path, image, mask):
    picture = image.copy()
    picture.thumbnail((800, 650))
    draw = ImageDraw.Draw(picture, "RGBA")
    height, width = mask.shape
    for y, x in zip(*mask.nonzero()):
        draw.rectangle((
            int(x * picture.width / width), int(y * picture.height / height),
            int((x + 1) * picture.width / width) - 1, int((y + 1) * picture.height / height) - 1,
        ), fill=(50, 225, 90, 85))
    buf = io.BytesIO()
    picture.save(buf, format="PNG")
    atomic_write(path, buf.getvalue())


def grounding_status(question, mode, review):
    qid = question["question_id"]
    if qid in review:
        row = review[qid]
        return row["status"], row["word_ids"], row.get("context_dependent", False)
    if mode == "smoke" and not review and question["candidate"]:
        return "automatic_smoke_only", question["candidate"]["word_ids"], False
    return ("unreviewed_candidate" if question["candidate"] else "no_literal_candidate"), [], False


def oracle_image(model, image, root, output, config, mode, review, shared):
    from .experiment import paired_images

    clean, degraded = paired_images(root, image)
    first = image["questions"][0]["question"]
    reference = model.prepare(clean, first)
    base = model.prepare(degraded, first)
    assert_aligned(reference, base)
    clean_features, degraded_features = feature_pair(model, image, reference, base, output, shared)
    grid_thw = base["image_grid_thw"][0].tolist()
    grid = image_grid(grid_thw)
    rule, threshold = config["mask_rule"], config["cell_overlap_threshold"]
    all_text = project(image["words"], clean.size, grid, rule, threshold)
    other_rule = "center" if rule == "overlap" else "overlap"
    alternative_text = project(image["words"], clean.size, grid, other_rule, threshold)
    small_word_ids = {
        word["id"] for word in image["words"]
        if max(y for _, y in word["polygon"]) - min(y for _, y in word["polygon"]) <= clean.height / grid[0]
    }
    small_word_fraction = len(small_word_ids) / len(image["words"])
    rows, coverage = [], []
    for q in image["questions"]:
        qid = q["question_id"]
        batch = model.prepare(degraded, q["question"])
        if batch["image_grid_thw"][0].tolist() != grid_thw:
            raise RuntimeError(f"Image grid changed across questions for {image['image_id']}")
        status, word_ids, context = grounding_status(q, mode, review)
        arms = [
            ("O0", degraded_features, 0.0),
            ("O1", clean_features, 1.0),
            ("O2", replace_region(degraded_features, clean_features, all_text), float(all_text.mean())),
        ]
        small_answer_text = None
        record = {
            "question_id": qid, "image_id": image["image_id"], "grounding": status,
            "literal_candidate": bool(q["candidate"]), "small_ocr_text": bool(small_word_ids),
            "small_ocr_word_fraction": small_word_fraction,
            "text_mask_tokens": int(all_text.sum()),
            "alternative_text_mask_tokens": int(alternative_text.sum()),
            "grid_tokens": all_text.size,
        }
        if status in {"accepted", "automatic_smoke_only", "automatic_provisional"}:
            selected = [word for word in image["words"] if word["id"] in word_ids]
            answer_mask = project(selected, clean.size, grid, rule, threshold)
            if not answer_mask.any():
                record["grounding"] = "empty_projected_answer_mask"
            else:
                small_answer_text = bool(small_word_ids.intersection(word_ids))
                record["small_answer_text"] = small_answer_text
                matched, random_mode = random_region(answer_mask, int(rank(config["random_seed"], qid)[:16], 16))
                expanded = dilate(answer_mask, config["dilation_radius"])
                masks = {"O3": answer_mask, "O4": matched, "O5": expanded}
                record["random_mode"] = random_mode
                record["o3_o4_shared_tokens"] = int((answer_mask & matched).sum())
                record["mask_tokens"] = {arm: int(mask.sum()) for arm, mask in masks.items()}
                for arm, mask in masks.items():
                    arms.append((arm, replace_region(degraded_features, clean_features, mask), float(mask.mean())))
                # Store a deterministic sample across IDs instead of every image.
                if config["overlay_questions"] > 0:
                    for arm, mask in {"O2": all_text, **masks}.items():
                        destination = output / "overlay_candidates" / f"{rank(config['random_seed'], qid)}-{arm}.png"
                        overlay(destination, degraded, mask)
        coverage.append(record)
        for arm, features, area in arms:
            rows.append({
                "question_id": qid, "image_id": image["image_id"], "arm": arm,
                "grid_thw": grid_thw, "mask_fraction": area, "small_ocr_text": bool(small_word_ids),
                "small_answer_text": small_answer_text,
                "context_dependent": context, **model.generate(batch, features),
            })
    return {"rows": rows, "coverage": coverage}


def oracle_report(output, manifest, mode, config, details, comparisons, grounding_policy):
    by_status = Counter(row["grounding"] for row in details)
    grounded = [row for row in details if "mask_tokens" in row]
    write_json(output / "oracle_coverage.json", {
        "mode": mode, "images": len(manifest["images"]), "questions": len(details),
        "grounding_policy": grounding_policy,
        "mask_rule": config["mask_rule"], "cell_overlap_threshold": config["cell_overlap_threshold"],
        "literal_candidates": sum(row["literal_candidate"] for row in details),
        "grounding_status": dict(by_status), "grounded_questions": len(grounded),
        "grounded_images": len({row["image_id"] for row in grounded}),
        "grounded_fraction": len(grounded) / len(details),
        "random_mode": dict(Counter(row["random_mode"] for row in grounded)),
        "random_with_answer_overlap": sum(row["o3_o4_shared_tokens"] > 0 for row in grounded),
        "selection_bias": (
            "Candidate packs favor unique literal OCR matches with high annotator agreement; "
            "multi-word and context-dependent answers may be underrepresented."
        ),
        "records": details,
    })
    ranked = sorted(grounded, key=lambda row: rank(config["random_seed"], row["question_id"]))
    for row in ranked[:config["overlay_questions"]]:
        prefix = rank(config["random_seed"], row["question_id"])
        for arm in ("O2", "O3", "O4", "O5"):
            source = output / "overlay_candidates" / f"{prefix}-{arm}.png"
            atomic_write(output / "overlays" / f"{row['question_id']}-{arm}.png", source.read_bytes())
    memo = oracle_decision_memo(mode, len(grounded), len(details), grounding_policy, comparisons)
    atomic_write(output / "decision.md", memo.encode("utf-8"))
