"""Run and resume D0 or S0 on a Slurm GPU allocation."""

import importlib.metadata
import os
from collections import Counter
from pathlib import Path

from filelock import FileLock

from .case_review import S0_LABELS, write_case_review

from .common import (
    code_hash, object_hash, read_json, read_jsonl, sha256, write_json, write_jsonl,
)
from .data import PROTOCOL, bicubic_control, load_clean, rank, verify_pillow
from .evaluation import evaluator, summarize
from .grounding import provisional_grounding, read_review

REQUIRED_D0_ARMS = {"CLEAN", "LR4-BIC-CONTROL"}
REQUIRED_S0_ARMS = {"O0", "O1", "O2"}
GROUNDED_ARMS = {"O3", "O4", "O5"}


def validate_manifest(manifest):
    if (
        manifest["schema"] != 1 or manifest["protocol"] != PROTOCOL
        or manifest["split"] != "official_train/val_inner"
        or not manifest["images"] or manifest["requested_images"] != len(manifest["images"])
    ):
        raise ValueError("Expected a nonempty Q-LRG val_inner manifest")
    ids = [image["image_id"] for image in manifest["images"]]
    qids = [q["question_id"] for image in manifest["images"] for q in image["questions"]]
    if len(ids) != len(set(ids)) or len(qids) != len(set(qids)) or not set(ids) <= set(manifest["val_inner_image_ids"]):
        raise ValueError("Manifest has duplicate IDs or images outside val_inner")
    for image in manifest["images"]:
        iid = image["image_id"]
        if image["image_rel"] != f"train_images/{iid}.jpg" or Path(iid).name != iid:
            raise ValueError(f"Manifest contains an invalid image path for {iid}")
        if not image["questions"] or not image["words"]:
            raise ValueError(f"No questions or OCR annotations for {iid}")


def validate_rows(image, rows, stage):
    questions = {q["question_id"] for q in image["questions"]}
    seen = {}
    required = REQUIRED_D0_ARMS if stage == "d0" else REQUIRED_S0_ARMS
    allowed = required if stage == "d0" else required | GROUNDED_ARMS
    for row in rows:
        qid, arm = row["question_id"], row["arm"]
        if row["image_id"] != image["image_id"] or qid not in questions or arm not in allowed:
            raise ValueError("Prediction row has an unexpected image, question or arm")
        arms = seen.setdefault(qid, set())
        if arm in arms:
            raise ValueError(f"Duplicate prediction for {qid}/{arm}")
        arms.add(arm)
    if questions != seen.keys() or any(arms not in (required, allowed) for arms in seen.values()):
        raise ValueError(f"Incomplete prediction coverage for {image['image_id']}")


def verified_complete(output, stage, fingerprint=None):
    complete = read_json(output / "COMPLETE.json")
    binding = read_json(output / "run.json")
    if complete["stage"] != stage or object_hash(binding) != complete["binding"]:
        raise ValueError(f"Completion marker mismatch in {output}")
    if fingerprint is not None and complete["binding"] != fingerprint:
        raise ValueError(f"Experiment binding changed in {output}; choose a new run directory")
    required = {"predictions.jsonl", "identity.json", "baseline_summary.json", "baseline_summary.csv"} if stage == "d0" else {
        "oracle_predictions.jsonl", "oracle_summary.json", "oracle_summary.csv",
        "oracle_coverage.json", "decision.md",
    }
    if not required <= complete["artifacts"].keys():
        raise ValueError(f"Missing required artifacts in completion marker {output}")
    for name, digest in complete["artifacts"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return binding


def paired_images(root, image):
    clean, orientation, _ = load_clean(root / image["image_rel"], image["image_sha256"])
    if list(clean.size) != image["size"] or orientation != image["exif_orientation"]:
        raise ValueError(f"Image geometry changed: {image['image_id']}")
    return clean, bicubic_control(clean)


def d0_identity(model, manifest, root, count, output):
    from .qwen_wrapper import assert_aligned

    if count < 1:
        raise ValueError("At least one D0 identity image is required")
    results = []
    grids = set()
    for image in manifest["images"]:
        clean, degraded = paired_images(root, image)
        q = image["questions"][0]
        base = model.prepare(degraded, q["question"])
        grid = tuple(base["image_grid_thw"][0].tolist())
        if grid in grids and len(results) < count:
            continue
        grids.add(grid)
        reference = model.prepare(clean, q["question"])
        assert_aligned(reference, base)
        results.append({
            "image_id": image["image_id"], "question_id": q["question_id"], "grid_thw": grid,
            "o0_vs_native_lr4": model.compare(base, base, model.encode(base)),
            "o1_vs_native_clean": model.compare(reference, base, model.encode(reference)),
        })
        if len(results) == count:
            break
    if len(results) < min(count, len(manifest["images"])):
        raise RuntimeError("Manifest has too few distinct grids for the requested D0 identity audit")
    write_json(output, {"passed": True, "images": results})


def d0_image(model, image, root):
    from .qwen_wrapper import assert_aligned

    clean, degraded = paired_images(root, image)
    rows = []
    for q in image["questions"]:
        reference = model.prepare(clean, q["question"])
        base = model.prepare(degraded, q["question"])
        assert_aligned(reference, base)
        grid = base["image_grid_thw"][0].tolist()
        for arm, batch in (("CLEAN", reference), ("LR4-BIC-CONTROL", base)):
            rows.append({
                "question_id": q["question_id"], "image_id": image["image_id"],
                "arm": arm, "grid_thw": grid, **model.generate(batch),
            })
    return {"rows": rows}


def execute(stage, mode, manifest_path, paths, model_config, config, output, review_path=None,
            d0_path=None, allow_provisional=False):
    import torch

    from .oracle import oracle_image, oracle_report
    from .qwen_wrapper import Qwen

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU inference must run in a Slurm allocation")
    verify_pillow()
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    evaluator()  # Fail before model loading if the scoring source is missing.
    if mode == "smoke" and len(manifest["images"]) > 20:
        raise ValueError("Smoke mode supports at most 20 images")
    runtime = {p: importlib.metadata.version(p) for p in (
        "torch", "torchvision", "transformers", "Pillow", "safetensors", "numpy",
        "tokenizers", "huggingface-hub",
    )}
    shared = {
        "manifest_sha256": sha256(manifest_path), "model": model_config,
        "seed": manifest["seed"], "preprocessing": PROTOCOL,
        "mode": mode, "packages": runtime, "source_hash": code_hash(),
        "paths": {key: str(value) for key, value in paths.items()},
        "torch_build": torch.__version__, "torch_cuda": torch.version.cuda,
    }
    review = {}
    grounding_policy = None
    binding = {"shared": shared, "stage": stage, "config": config}
    if stage == "s0":
        d0 = verified_complete(d0_path, "d0")
        if d0["shared"] != shared or not read_json(d0_path / "identity.json")["passed"]:
            raise ValueError("S0 requires a completed, matching D0 identity gate")
        if allow_provisional and (mode != "full" or review_path is not None):
            raise ValueError("Provisional grounding is a separate full S0 option")
        if review_path is not None:
            grounding_policy = "human_reviewed"
            review = read_review(review_path, manifest)
        elif allow_provisional:
            grounding_policy = "provisional_ocr_150"
            review = provisional_grounding(manifest)
            if not review:
                raise ValueError("No usable OCR candidates for provisional full S0")
        elif mode == "smoke":
            grounding_policy = "smoke_ocr"
        else:
            raise ValueError("Full S0 requires --review or explicit --provisional-grounding")
        binding["grounding_policy"] = grounding_policy
        binding["d0_binding"] = object_hash(d0)
        binding["review_sha256"] = sha256(review_path) if review_path is not None else None

    output.mkdir(parents=True, exist_ok=True)
    fingerprint = object_hash(binding)
    with FileLock(str(output / ".run.lock"), timeout=0):
        if (output / "COMPLETE.json").exists():
            verified_complete(output, stage, fingerprint)
            print(f"Verified completed {stage} at {output}")
            return
        if (output / "run.json").exists():
            if read_json(output / "run.json") != binding:
                raise ValueError(f"Experiment binding changed in {output}; choose a new run directory")
        elif any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError(f"Existing directory has no Q-LRG experiment binding: {output}")
        else:
            write_json(output / "run.json", binding)

        torch.manual_seed(manifest["seed"])
        model = Qwen(model_config, paths["hf_hub"])
        torch.cuda.reset_peak_memory_stats()
        identity_path = output / "identity.json"
        if stage == "d0":
            if not identity_path.exists():
                d0_identity(model, manifest, paths["textvqa"], config["identity_images"], identity_path)
            if not read_json(identity_path)["passed"]:
                raise RuntimeError("D0 identity parity failed")
        rows, details = [], []
        for index, image in enumerate(manifest["images"]):
            path = output / "shards" / f"{index:06d}.json"
            if path.exists():
                shard = read_json(path)
            else:
                if stage == "d0":
                    shard = d0_image(model, image, paths["textvqa"])
                else:
                    shard = oracle_image(model, image, paths["textvqa"], output, config, mode, review, shared)
                validate_rows(image, shard["rows"], stage)
                write_json(path, shard)
            validate_rows(image, shard["rows"], stage)
            rows.extend(shard["rows"])
            details.extend(shard.get("coverage", []))
            if (index + 1) % 25 == 0 or index + 1 == len(manifest["images"]):
                print(f"{stage}: {index + 1}/{len(manifest['images'])} images", flush=True)

        comparisons = summarize(
            output, rows, manifest, stage,
            config.get("bootstrap_resamples", 10000), config.get("bootstrap_seed", 42),
            grounding_policy=grounding_policy,
        )
        predictions = "oracle_predictions.jsonl" if stage == "s0" else "predictions.jsonl"
        write_jsonl(output / predictions, rows)
        if stage == "s0":
            oracle_report(output, manifest, mode, config, details, comparisons, grounding_policy)
            write_case_review(
                output / "review.html", rows, S0_LABELS, title=f"S0 · {output.parent.name}",
                manifest=manifest, overlays_dir=output / "overlays", grounding_policy=grounding_policy,
                repeats=config.get("bootstrap_resamples", 10000), seed=config.get("bootstrap_seed", 42),
            )
        write_json(output / "runtime.json", {
            "job_id": os.environ["SLURM_JOB_ID"], "device": torch.cuda.get_device_name(),
            "peak_allocated_bytes_this_job": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes_this_job": torch.cuda.max_memory_reserved(),
            "prediction_rows": len(rows), "rows_per_arm": dict(Counter(row["arm"] for row in rows)),
        })
        artifacts = {
            p.name: sha256(p) for p in output.iterdir()
            if p.is_file() and not p.name.startswith(".")
        }
        artifacts.update({
            p.relative_to(output).as_posix(): sha256(p)
            for p in (output / "overlays").glob("*.png")
        })
        write_json(output / "COMPLETE.json", {
            "stage": stage, "binding": fingerprint, "artifacts": artifacts,
        })
        print(f"Completed {stage}: {output}")
