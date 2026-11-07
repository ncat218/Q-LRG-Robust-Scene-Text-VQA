"""Evaluate a locked expanded S1 run on labeled official validation."""

import os
from pathlib import Path

import torch
from filelock import FileLock

from .common import object_hash, read_json, sha256, write_json
from .data import verify_pillow
from .evaluation import evaluator
from .experiment import verified_complete
from .qwen_wrapper import Qwen
from .s1_manifests import read_s1_manifests
from .s1_report import arm_labels, write_report
from .s1_run import ImageFeatures, make_adapter, predict, runtime_shared
from .s1_storage import resume_runtime


def evaluate_s1(source, output, paths, model_config):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Official validation must run in a Slurm GPU allocation")
    verify_pillow()
    trained = verified_complete(source / "s1", "s1")
    if "dataset_dir" not in trained:
        raise ValueError("Lock a run trained from the full S1 manifests for official validation")
    datasets, hashes = read_s1_manifests(Path(trained["dataset_dir"]), paths["textvqa"])
    if hashes != trained["dataset_hashes"]:
        raise ValueError("Dataset manifests changed since training")
    shared = runtime_shared(paths, model_config)
    cache_shared = resume_runtime(trained["shared"], shared)
    config = trained["config"]
    labels = arm_labels(config)
    checkpoint_hashes = {a: sha256(source / "s1" / "variants" / a / "checkpoint.pt") for a in labels if a != "A0"}
    binding = {"stage": "s1_eval", "shared": shared, "trained_binding": object_hash(trained),
               "checkpoints": checkpoint_hashes, "evaluation_role": "official_val_locked",
               "manifest_hash": hashes["official_val"], "source": str(source.resolve())}
    fingerprint = object_hash(binding)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".run.lock"), timeout=0):
        if (output / "run.json").exists():
            saved = read_json(output / "run.json")
            binding["shared"] = resume_runtime(saved["shared"], shared)
            fingerprint = object_hash(binding)
            if saved != binding:
                raise ValueError("Official evaluation binding changed")
        if (output / "COMPLETE.json").exists():
            verified_complete(output, "s1_eval", fingerprint)
            return
        write_json(output / "run.json", binding)
        manifest = datasets["official_val"]
        write_json(output / "eval_manifest.json", manifest)
        scorer = evaluator()
        qwen = Qwen(model_config, paths["hf_hub"])
        cache = ImageFeatures(qwen, paths["textvqa"], output, cache_shared)
        stats = read_json(source / "s1" / "summary.json")["runtime"]
        rows = []
        for arm in labels:
            torch.cuda.reset_peak_memory_stats()
            adapter = make_adapter(qwen, arm, config)
            if adapter is not None:
                state = torch.load(source / "s1" / "variants" / arm / "checkpoint.pt",
                                   map_location="cpu", weights_only=True)
                adapter.load_state_dict(state["adapter"])
                del state
            rows.extend(predict(qwen, arm, adapter, cache, manifest, output / "variants" / arm, scorer))
            stats[arm]["evaluation_peak_allocated_bytes_this_job"] = torch.cuda.max_memory_allocated()
            del adapter
        write_report(output, manifest, rows, stats, [], config)
        artifacts = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and not p.name.startswith(".")}
        write_json(output / "COMPLETE.json", {"stage": "s1_eval", "binding": fingerprint, "artifacts": artifacts})
        print(f"Official validation complete: {output}", flush=True)
