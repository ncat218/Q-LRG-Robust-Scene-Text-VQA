"""Repeat native and cached-injection A0 paths on the same questions."""

import inspect
import os
from pathlib import Path

import torch
from filelock import FileLock

from .common import object_hash, read_json, read_jsonl, sha256, write_json, write_jsonl
from .data import annotation_file, verify_pillow
from .evaluation import evaluator, paired_bootstrap
from .experiment import paired_images, validate_manifest, verified_complete
from .qwen_wrapper import Qwen
from .s1_manifests import validate_s1_manifest
from .s1_storage import accepts_prior_audit



AUDIT_POLICY = "within-path-repeatability-v2"
LEGACY_SOURCE_HASH = "d22a29cf448d4d247de2c2302e09b23b6013cb49ad84a132d2a94e0e55c8c234"
LEGACY_A0_PATH_HASH = "786a7b319e54830c990f43e2c92309049b37c83aad15eb19a32f085eaeaebb2b"


def a0_path_hash():
    """Bind cache, preprocessing, scoring and prediction code independently of training/reporting."""
    from .s1_run import ImageFeatures, predict

    files = ("data.py", "evaluation.py", "feature_injection.py", "grid_alignment.py",
             "oracle.py", "qwen_wrapper.py", "s1_model.py")
    root = Path(__file__).parent
    signature = {name: sha256(root / name) for name in files}
    signature.update({part.__name__: inspect.getsource(part) for part in (ImageFeatures, paired_images, predict)})
    return object_hash(signature)

def audit_summary(rows, manifest, repeats=10000, seed=42, diagnostics=()):
    expected = {(im["image_id"], q["question_id"]) for im in manifest["images"] for q in im["questions"]}
    by_path = {}
    for row in rows:
        key = row["condition"], row["path"], row["repeat"]
        values = by_path.setdefault(key, {})
        case = row["image_id"], row["question_id"]
        if case in values:
            raise ValueError(f"Duplicate A0 audit prediction: {key}/{case}")
        values[case] = row
    keys = {(c, p, n) for c in ("CLEAN", "LR4") for p in ("native", "injected") for n in (1, 2)}
    if by_path.keys() != keys or any(v.keys() != expected for v in by_path.values()):
        raise ValueError("Incomplete A0 audit prediction coverage")
    contrasts = []
    ordered = sorted(expected)
    for condition in ("CLEAN", "LR4"):
        for left, right in (
            (("native", 1), ("native", 2)),
            (("injected", 1), ("injected", 2)),
            (("native", 1), ("injected", 1)),
            (("native", 2), ("injected", 2)),
        ):
            a = [by_path[(condition, *left)][key] for key in ordered]
            b = [by_path[(condition, *right)][key] for key in ordered]
            changed = [
                {"image_id": x["image_id"], "question_id": x["question_id"],
                 "left_answer": x["answer"], "right_answer": y["answer"],
                 "left_score": x["score"], "right_score": y["score"],
                 "tokens_equal": x["token_ids"] == y["token_ids"]}
                for x, y in zip(a, b) if x["token_ids"] != y["token_ids"] or x["score"] != y["score"]
            ]
            contrasts.append({
                "condition": condition, "left": list(left), "right": list(right), "questions": len(a),
                "different_token_sequences": sum(x["token_ids"] != y["token_ids"] for x, y in zip(a, b)),
                "different_scores": sum(x["score"] != y["score"] for x, y in zip(a, b)),
                "mean_absolute_score_change_pp": 100 * sum(
                    abs(row["left_score"] - row["right_score"]) for row in changed
                ) / len(a),
                "changed": changed,
                **paired_bootstrap([x["score"] for x in a], [y["score"] for y in b],
                                   [key[0] for key in ordered], repeats, seed),
            })
    metrics = ("different_token_sequences", "different_scores", "mean_absolute_score_change_pp")
    limits = {}
    for condition in ("CLEAN", "LR4"):
        same_path = [r for r in contrasts if r["condition"] == condition and r["left"][0] == r["right"][0]]
        cross_path = [r for r in contrasts if r["condition"] == condition and r["left"][0] != r["right"][0]]
        observed = {metric: max(r[metric] for r in cross_path) for metric in metrics}
        limit = {metric: max(r[metric] for r in same_path) for metric in metrics}
        limits[condition] = {
            "same_path_max": limit, "cross_path_max": observed,
            "passed": all(observed[metric] <= limit[metric] for metric in metrics),
        }
    feature_checks_passed = all(row["cached_equals_fresh"] and row["prefill_equal"] for row in diagnostics)
    return {
        "passed": feature_checks_passed and all(result["passed"] for result in limits.values()),
        "policy": AUDIT_POLICY,
        "exact_match": all(not row["changed"] for row in contrasts),
        "repeatability": limits,
        "feature_checks_passed": feature_checks_passed,
        "contrasts": contrasts,
        "scope": "Two full passes in one job, batch size one, native versus first-question cached image features.",
    }


def validate_audit(output, shared):
    binding = verified_complete(output, "audit")
    saved = binding["shared"]
    for name in ("paths", "model", "packages", "torch_build", "torch_cuda", "device", "preprocessing"):
        if saved[name] != shared[name]:
            raise ValueError(f"A0 audit environment differs in {name}")
    if saved["source_hash"] != shared["source_hash"]:
        audited_path = saved.get("a0_path_hash")
        if audited_path is None and saved["source_hash"] == LEGACY_SOURCE_HASH:
            audited_path = LEGACY_A0_PATH_HASH
        if not audited_path or (
            audited_path != shared.get("a0_path_hash") and not accepts_prior_audit(audited_path, shared)
        ):
            raise ValueError("A0 inference code changed; run a fresh audit")
    summary = audit_summary(read_jsonl(output / "predictions.jsonl"), read_json(output / "manifest.json"),
                            diagnostics=read_json(output / "diagnostics.json"))
    if not summary["passed"]:
        raise ValueError("A0 path differences exceed the measured repeatability or feature checks failed")
    return object_hash(binding), summary


def run_audit(source, output, paths, model_config, pilot=None):
    from .s1_run import ImageFeatures, runtime_shared

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("A0 audit must run in a Slurm allocation")
    verify_pillow()
    manifest_path = source / "manifest.json"
    manifest = read_json(manifest_path)
    (validate_s1_manifest if manifest["schema"] == 2 else validate_manifest)(manifest)
    for split, digest in manifest["annotation_sha256"].items():
        if sha256(annotation_file(paths["textvqa"], split)) != digest:
            raise ValueError(f"Source annotations changed for {split}")
    pilot_binding, pilot_plan = None, None
    if pilot is not None:
        from .s1_manifests import check_disjoint

        pilot_binding = verified_complete(pilot / "s1", "s1")
        pilot_plan = read_json(pilot / "s1" / "train_manifest.json")
        if object_hash(pilot_plan) != pilot_binding["train_manifest"]:
            raise ValueError("Pilot training manifest changed")
        if object_hash(manifest) != pilot_binding["eval_manifest"]:
            raise ValueError("Pilot and audit use different evaluation manifests")
        check_disjoint([{"split": "pilot_train", "images": pilot_plan},
                        {"split": "pilot_eval", "images": manifest["images"]}])
    shared = runtime_shared(paths, model_config)
    binding = {"stage": "audit", "manifest_sha256": sha256(manifest_path), "shared": shared,
               "protocol": "native-and-cached-injection-two-passes-v1",
               "pilot_binding": object_hash(pilot_binding) if pilot_binding else None}
    fingerprint = object_hash(binding)
    scorer = evaluator()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".run.lock"), timeout=0):
        binding_path = output / "run.json"
        if binding_path.exists() and read_json(binding_path) != binding:
            raise ValueError("Audit binding changed; use a new run directory")
        if (output / "COMPLETE.json").exists():
            verified_complete(output, "audit", fingerprint)
            return
        write_json(binding_path, binding)
        if pilot_plan is not None:
            write_json(output / "pilot_train_manifest.json", pilot_plan)
            write_json(output / "pilot_split_audit.json", {
                "passed": True, "train_images": len(pilot_plan),
                "train_questions": sum(len(im["questions"]) for im in pilot_plan),
                "eval_images": len(manifest["images"]),
                "train_manifest_hash": object_hash(pilot_plan),
                "checked": ["image_id", "question_id", "image_sha256"],
            })
        qwen = Qwen(model_config, paths["hf_hub"])
        cache = ImageFeatures(qwen, paths["textvqa"], output, shared)
        rows, diagnostics = [], []
        for repeat in (1, 2):
            # Recreate the per-image in-memory state for the second full pass.
            cache.current = None
            for index, image in enumerate(manifest["images"]):
                shard_path = output / "shards" / f"{repeat}-{image['image_id']}.json"
                if shard_path.exists():
                    shard = read_json(shard_path)
                else:
                    clean, lr4 = paired_images(paths["textvqa"], image)
                    cache.load(image)
                    shard = {"rows": [], "diagnostics": []}
                    for condition, rgb in (("CLEAN", clean), ("LR4", lr4)):
                        for question in image["questions"]:
                            batch = qwen.prepare(rgb, question["question"])
                            feature = cache.features[condition]
                            native = qwen.generate(batch)
                            injected = qwen.generate(batch, feature)
                            for path, result in (("native", native), ("injected", injected)):
                                refs = scorer._compute_answer_scores(question["answers"])
                                shard["rows"].append({
                                    "condition": condition, "path": path, "repeat": repeat,
                                    "image_id": image["image_id"], "question_id": question["question_id"],
                                    "grid_thw": cache.thw, **result,
                                    "score": refs.get(scorer.answer_processor(result["answer"]), 0.0),
                                })
                            if native["token_ids"] != injected["token_ids"]:
                                fresh = qwen.encode(batch)
                                left = qwen.prefill_logits(batch)
                                right = qwen.prefill_logits(batch, feature)
                                shard["diagnostics"].append({
                                    "repeat": repeat, "condition": condition,
                                    "image_id": image["image_id"], "question_id": question["question_id"],
                                    "cached_equals_fresh": torch.equal(feature, fresh),
                                    "max_feature_difference": float((feature.float() - fresh.float()).abs().max()),
                                    "prefill_equal": torch.equal(left, right),
                                    "max_prefill_difference": float((left.float() - right.float()).abs().max()),
                                    "fresh_injected": qwen.generate(batch, fresh),
                                })
                    write_json(shard_path, shard)
                rows.extend(shard["rows"])
                diagnostics.extend(shard["diagnostics"])
                if (index + 1) % 50 == 0:
                    print(f"A0 audit pass {repeat}: {index+1}/{len(manifest['images'])} images", flush=True)
        summary = audit_summary(rows, manifest, diagnostics=diagnostics)
        historical = {}
        old_sources = [("d0", source / "d0" / "predictions.jsonl",
                        {"CLEAN": "CLEAN", "LR4": "LR4-BIC-CONTROL"})]
        if pilot is not None:
            old_sources.append(("s1_pilot", pilot / "s1" / "predictions.jsonl",
                                {"CLEAN": "A0", "LR4": "A0"}))
        for name, path, arms in old_sources:
            old = read_jsonl(path)
            old_index = {}
            for row in old:
                condition = row.get("condition")
                old_index[row["image_id"], row["question_id"], row["arm"], condition] = row
            differences = []
            for new in rows:
                if new["repeat"] != 1 or new["path"] != "native":
                    continue
                key = (new["image_id"], new["question_id"], arms[new["condition"]],
                       new["condition"] if name == "s1_pilot" else None)
                prior = old_index.get(key)
                if prior is None:
                    raise ValueError("Historical predictions do not match the audit cohort")
                if prior["token_ids"] != new["token_ids"]:
                    differences.append({"condition": new["condition"], "question_id": new["question_id"],
                                        "old_answer": prior["answer"], "new_answer": new["answer"],
                                        "old_score": prior["score"], "new_score": new["score"]})
            historical[name] = differences
        write_json(output / "historical_comparisons.json", historical)
        write_json(output / "summary.json", summary)
        write_json(output / "diagnostics.json", diagnostics)
        write_jsonl(output / "predictions.jsonl", rows)
        write_json(output / "manifest.json", manifest)
        write_json(output / "COMPLETE.json", {"stage": "audit", "binding": fingerprint,
                   "artifacts": {p.name: sha256(p) for p in output.iterdir()
                                 if p.is_file() and not p.name.startswith(".")}})
        print(f"A0 audit passed: {summary['passed']}. Report: {output / 'summary.json'}", flush=True)
