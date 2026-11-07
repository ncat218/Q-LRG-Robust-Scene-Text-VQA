"""Train and evaluate S1 variants in one Slurm allocation."""

import importlib.metadata
import io
import math
import os
import shutil
import time
from pathlib import Path
from contextlib import nullcontext
from itertools import islice

import torch
from filelock import FileLock
from torch.nn import functional as F

from .common import atomic_write, code_hash, object_hash, read_json, sha256, write_json
from .data import PROTOCOL, verify_pillow
from .evaluation import evaluator
from .experiment import d0_identity, paired_images, verified_complete
from .grid_alignment import image_grid
from .oracle import feature_pair
from .qwen_wrapper import Qwen, assert_aligned
from .s1_data import build_train_plan, compare_archived_baselines, read_eval_anchor, training_examples
from .s1_model import LanguageLoRA, VisualResidual, teacher_forced_loss, text_training
from .s1_report import CONDITIONS, EXTRA_LABELS, arm_labels, write_report, write_visuals
from .s1_storage import checkpoint_steps, feature_cache_mode, resume_runtime


def runtime_shared(paths, model_config):
    from .a0_audit import a0_path_hash

    packages = {name: importlib.metadata.version(name) for name in (
        "torch", "torchvision", "transformers", "Pillow", "safetensors", "numpy", "tokenizers", "huggingface-hub",
    )}
    return {"model": model_config, "source_hash": code_hash(), "a0_path_hash": a0_path_hash(), "packages": packages,
              "torch_build": torch.__version__, "torch_cuda": torch.version.cuda, "preprocessing": PROTOCOL,
              "device": torch.cuda.get_device_name(),
              "paths": {key: str(value) for key, value in paths.items()}}


def validate_config(config):
    if type(config["seed"]) is not int or type(config.get("clean_controls", False)) is not bool:
        raise ValueError("S1 seed must be an integer and clean_controls must be a boolean")
    if config["train_images"] != "all" and (type(config["train_images"]) is not int or config["train_images"] < 1):
        raise ValueError("train_images must be a positive integer or all")
    cadence = {"checkpoint_every", "checkpoint_saves"} & config.keys()
    if len(cadence) != 1:
        raise ValueError("Choose exactly one of checkpoint_every and checkpoint_saves")
    for key in ("epochs", "accumulate", *cadence, "hidden", "spatial_blocks",
                "lora_rank", "max_answer_tokens", "bootstrap_resamples"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"S1 requires a positive integer for {key}")
    clean_penalty = config.get("clean_residual_weight", 0)
    if not math.isfinite(clean_penalty) or clean_penalty < 0:
        raise ValueError("clean_residual_weight must be finite and nonnegative")
    for key in ("learning_rate", "max_grad_norm", "lora_alpha", "feature_weight",
                "residual_weight", "weight_decay", "clean_fraction"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"Invalid S1 setting: {key}")
    if not 0 <= config["clean_fraction"] < 1 or not all(
        config[key] > 0 for key in ("learning_rate", "max_grad_norm", "lora_alpha", "feature_weight")
    ):
        raise ValueError("S1 needs LR4 training examples and positive training scales")


class ImageFeatures:
    """Keep just one aligned CLEAN/LR4 pair in memory while reading the shared disk cache."""

    def __init__(self, qwen, root, output, shared):
        self.qwen = qwen
        self.root = root
        self.output = output
        self.shared = shared
        self.current = None

    def load(self, image):
        if self.current == image["image_id"]:
            return
        clean, lr4 = paired_images(self.root, image)
        question = image["questions"][0]["question"]
        reference, base = (self.qwen.prepare(im, question) for im in (clean, lr4))
        assert_aligned(reference, base)
        self.thw = base["image_grid_thw"][0].tolist()
        self.grid = image_grid(self.thw)
        clean_feature, lr4_feature = feature_pair(
            self.qwen, image, reference, base, self.output, self.shared,
        )
        self.images = dict(zip(CONDITIONS, (clean, lr4)))
        self.features = dict(zip(CONDITIONS, (clean_feature, lr4_feature)))
        self.current = image["image_id"]

    def prompt(self, condition, question):
        batch = self.qwen.prepare(self.images[condition], question)
        if batch["image_grid_thw"][0].tolist() != self.thw:
            raise RuntimeError("Evaluation prompt changed the cached image grid")
        return batch


def make_adapter(qwen, arm, config):
    torch.manual_seed(config["seed"])
    arm = arm.removesuffix("clean")
    if arm == "A0":
        return None
    language = qwen.model.model.language_model
    if arm == "A2":
        adapter = LanguageLoRA(language, config["lora_rank"], config["lora_alpha"])
    else:
        width = qwen.model.config.text_config.hidden_size
        adapter = VisualResidual(width, config["hidden"], 0 if arm == "A1" else config["spatial_blocks"])
    return adapter.to(qwen.device)


def loss_for(qwen, arm, adapter, cache, question, condition, answer, config):
    arm = arm.removesuffix("clean")
    base = cache.features[condition]
    if arm == "A2":
        replacement = base
    else:
        replacement, delta = adapter(base, cache.grid)
    if arm in ("S1a", "S1b"):
        match = F.mse_loss(replacement, cache.features["CLEAN"].float())
        if arm == "S1a":
            penalty = config.get("clean_residual_weight", 0) * delta.square().mean() if condition == "CLEAN" else 0
            return match + penalty
    tokenizer = qwen.processor.tokenizer
    tokens = tokenizer.encode(answer, add_special_tokens=False)[:config["max_answer_tokens"]]
    tokens.append(tokenizer.eos_token_id)
    prompt = cache.prompt(condition, question["question"])
    loss = teacher_forced_loss(qwen, prompt, replacement.to(qwen.dtype), tokens)
    if arm == "S1b":
        loss = loss + config["feature_weight"] * match + (config["residual_weight"] + (config.get("clean_residual_weight", 0) if condition == "CLEAN" else 0)) * delta.square().mean()
    return loss


def train(qwen, arm, adapter, cache, examples, config, path, fingerprint):
    save_steps = checkpoint_steps(config, len(examples))
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    offset, history = 0, []
    if path.is_file():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["binding"] != fingerprint or checkpoint["arm"] != arm:
            raise ValueError(f"Checkpoint belongs to a different run or variant: {path}")
        adapter.load_state_dict(checkpoint["adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        offset, history = checkpoint["examples"], checkpoint["history"]
        if not 0 <= offset <= len(examples) or (offset != len(examples) and offset % config["accumulate"]):
            raise ValueError(f"Checkpoint has an invalid example offset: {offset}")
        del checkpoint

    adapter.train()
    if offset == len(examples):
        if not any(step["grad_norm"] > 0 for step in history):
            raise RuntimeError(f"{arm} produced no nonzero adapter gradients")
        return history
    attach = adapter.attached() if arm.removesuffix("clean") == "A2" else nullcontext()
    is_cuda = torch.device(qwen.device).type == "cuda"
    remaining = iter(examples[offset:])
    accumulate = config["accumulate"]
    with attach, text_training(qwen.model, arm != "S1a"):
        while group := list(islice(remaining, accumulate)):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=qwen.device)
            for image, question, condition, answer in group:
                cache.load(image)
                loss = loss_for(qwen, arm, adapter, cache, question, condition, answer, config)
                (loss / len(group)).backward()
                total += loss.detach()
            norm = float(torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), config["max_grad_norm"], error_if_nonfinite=True,
            ))
            average = float(total / len(group))
            if not math.isfinite(average):
                raise RuntimeError(f"{arm}: nonfinite loss at example {offset}")
            optimizer.step()
            offset += len(group)
            history.append({"examples": offset, "loss": average, "grad_norm": norm,
                            "seconds": time.perf_counter() - start})
            if len(history) in save_steps:
                if is_cuda:
                    history[-1]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(qwen.device)
                buf = io.BytesIO()
                torch.save({"binding": fingerprint, "arm": arm, "examples": offset, "history": history,
                            "adapter": adapter.state_dict(), "optimizer": optimizer.state_dict()}, buf)
                atomic_write(path, buf.getvalue())
                print(f"{arm} training: {offset}/{len(examples)} questions, loss {average:.4f}", flush=True)
        if not any(step["grad_norm"] > 0 for step in history):
            raise RuntimeError(f"{arm} produced no nonzero adapter gradients")
    optimizer.zero_grad(set_to_none=True)
    return history


def probe_gradients(qwen, cache, examples, config):
    """Fail early in the allocation if BF16 teacher forcing cannot train the two injection paths."""
    image, question, condition, answer = next(example for example in examples if example[2] == "LR4")
    cache.load(image)
    results = {}
    for arm in ("S1b", "A2"):
        adapter = make_adapter(qwen, arm, config)
        attach = adapter.attached() if arm.removesuffix("clean") == "A2" else nullcontext()
        with attach, text_training(qwen.model, True):
            loss = loss_for(qwen, arm, adapter, cache, question, condition, answer, config)
            loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(
            adapter.parameters(), config["max_grad_norm"], error_if_nonfinite=True,
        ))
        if not math.isfinite(float(loss.detach())) or norm <= 0 or any(p.grad is not None for p in qwen.model.parameters()):
            raise RuntimeError(f"{arm} failed the frozen-backbone gradient gate")
        results[arm] = {"loss": float(loss.detach()), "grad_norm": norm}
        del loss, adapter
    return {"passed": True, "image_id": image["image_id"], "question_id": question["question_id"], "arms": results}


def predict(qwen, arm, adapter, cache, manifest, stage, scorer):
    rows = []
    qwen.model.eval()
    if adapter is not None:
        adapter.eval()
    attach = adapter.attached() if arm.removesuffix("clean") == "A2" else nullcontext()
    with attach, torch.inference_mode():
        for index, image in enumerate(manifest["images"]):
            iid = image["image_id"]
            shard = stage / "shards" / f"{iid}.json"
            if shard.is_file():
                prediction = read_json(shard)
            else:
                cache.load(image)
                prediction = []
                for condition in CONDITIONS:
                    base = cache.features[condition]
                    feature = base
                    adapter_seconds = 0.0
                    if arm.removesuffix("clean") in ("A1", "S1a", "S1b"):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        restored, _ = adapter(base, cache.grid)
                        feature = restored.to(qwen.dtype)
                        torch.cuda.synchronize()
                        adapter_seconds = time.perf_counter() - start
                        if index < 8:
                            write_visuals(stage, condition, iid, cache.images[condition],
                                          feature.float() - base.float(), cache.grid)
                    clean = cache.features["CLEAN"].float()
                    feature_mse = float(F.mse_loss(feature.float(), clean))
                    feature_cosine = float(F.cosine_similarity(feature.float(), clean, dim=-1).mean())
                    for question in image["questions"]:
                        qid = question["question_id"]
                        batch = cache.prompt(condition, question["question"])
                        refs = scorer._compute_answer_scores(question["answers"])
                        result = qwen.generate(batch, feature)
                        prediction.append({
                            "arm": arm, "condition": condition, "image_id": iid, "question_id": qid,
                            "grid_thw": cache.thw, **result, "adapter_seconds_per_image": adapter_seconds,
                            "feature_mse_to_clean": feature_mse, "feature_cosine_to_clean": feature_cosine,
                            "score": refs.get(scorer.answer_processor(result["answer"]), 0.0),
                        })
                write_json(shard, prediction)
            expected = {(qid["question_id"], condition) for qid in image["questions"] for condition in CONDITIONS}
            actual = {(r["question_id"], r["condition"]) for r in prediction if r["arm"] == arm and r["image_id"] == iid}
            if expected != actual or len(prediction) != len(expected):
                raise ValueError(f"Incomplete or duplicate S1 rows in {shard}")
            rows.extend(prediction)
            if (index + 1) % 50 == 0 or index + 1 == len(manifest["images"]):
                print(f"{arm} evaluation: {index + 1}/{len(manifest['images'])} images", flush=True)
    return rows


def run_s1(source, output, paths, model_config, config, dataset_dir=None, audit_path=None):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("S1 must run in a Slurm GPU allocation")
    verify_pillow()
    validate_config(config)
    scorer = evaluator()
    datasets, dataset_hashes = None, None
    if dataset_dir is None:
        manifest, d0_binding, s0_binding = read_eval_anchor(source, paths, model_config)
        if config["train_images"] == "all":
            raise ValueError("Expanded training needs independently validated S1 manifests and A0 audit")
    else:
        from .s1_manifests import read_s1_manifests

        datasets, dataset_hashes = read_s1_manifests(dataset_dir, paths["textvqa"])
        manifest = datasets["val_inner"]
        d0_binding, s0_binding = None, None
    shared = runtime_shared(paths, model_config)
    audit_binding, audit_assessment = None, None
    if dataset_dir is not None:
        from .a0_audit import validate_audit

        if audit_path is None:
            raise ValueError("Expanded S1 requires a passing A0 audit")
        audit_binding, audit_assessment = validate_audit(audit_path, shared)
    labels = arm_labels(config)
    binding = {"stage": "s1", "config": config, "shared": shared, "source": str(source.resolve()) if source else None,
               "d0_binding": d0_binding, "s0_binding": s0_binding, "eval_manifest": object_hash(manifest)}
    if dataset_dir is not None:
        binding.update({"dataset_hashes": dataset_hashes, "dataset_dir": str(dataset_dir.resolve()),
                        "audit_binding": audit_binding, "audit_policy": audit_assessment["policy"],
                        "evaluation_role": "hyperparameter_selection"})
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".run.lock"), timeout=0):
        plan_path = output / "train_manifest.json"
        binding_path = output / "run.json"
        saved = None
        if binding_path.is_file():
            saved = read_json(binding_path)
            binding["shared"] = resume_runtime(saved["shared"], shared)
            plan = read_json(plan_path)
        else:
            if datasets is None:
                plan = build_train_plan(paths["textvqa"], manifest, config["train_images"], config["seed"])
            else:
                all_train = datasets["train_inner"]["images"]
                count = len(all_train) if config["train_images"] == "all" else config["train_images"]
                if count > len(all_train):
                    raise ValueError("Requested more images than train_inner contains")
                plan = all_train[:count]
            write_json(plan_path, plan)
        binding["train_manifest"] = object_hash(plan)
        fingerprint = object_hash(binding)
        if saved is not None:
            if saved != binding:
                raise ValueError("S1 binding changed; use the original code/config or start a new run")
        else:
            write_json(binding_path, binding)
        if (output / "COMPLETE.json").exists():
            verified_complete(output, "s1", fingerprint)
            print(f"Verified completed S1 at {output}")
            return

        write_json(output / "baseline_audit.json",
                   compare_archived_baselines(source, manifest) if datasets is None else
                   {"audit_binding": audit_binding, "audit_run": str(audit_path.resolve()),
                    "assessment": audit_assessment})
        write_json(output / "eval_manifest.json", manifest)
        qwen = Qwen(model_config, paths["hf_hub"])
        identity = output / "identity.json"
        if not identity.exists():
            d0_identity(qwen, manifest, paths["textvqa"], 4, identity, require_decode=datasets is None)
        if not read_json(identity)["passed"]:
            raise RuntimeError("S1 identity gate failed")
        cache = ImageFeatures(qwen, paths["textvqa"], output, binding["shared"])
        if datasets is not None:
            cache_root = Path(os.environ.get("QLRG_FEATURE_CACHE", output / "features"))
            mode = feature_cache_mode()
            max_tokens = model_config["max_pixels"] // (28 * 28) + 128
            estimate = max_tokens * qwen.model.config.text_config.hidden_size * 4 * (len(plan) + len(manifest["images"]))
            if mode == "write":
                cache_root.mkdir(parents=True, exist_ok=True)
                free = shutil.disk_usage(cache_root).free
                next_pair_bound = max_tokens * qwen.model.config.text_config.hidden_size * 4
                if free < 2 * next_pair_bound:
                    raise RuntimeError("Insufficient free space to atomically write the next feature pair")
            else:
                estimate = 0
                free = shutil.disk_usage(output).free
            write_json(output / "storage.json", {
                "cache": str(cache_root), "feature_cache_mode": mode, "estimated_upper_bytes": estimate,
                "free_bytes": free, "runtime_source_hash": shared["source_hash"],
                "checkpoint_saves_override": os.environ.get("QLRG_CHECKPOINT_SAVES"),
            })
            print(f"S1 feature cache mode: {mode}", flush=True)
        examples = list(training_examples(plan, config["epochs"], config["seed"], config["clean_fraction"]))
        gradient_gate = output / "gradient_gate.json"
        if not gradient_gate.exists():
            write_json(gradient_gate, probe_gradients(qwen, cache, examples, config))
        if not read_json(gradient_gate)["passed"]:
            raise RuntimeError("S1 gradient gate failed")
        rows, stats = [], {}
        # Train each variant from its own initialization before any evaluation is inspected.
        for arm in labels:
            if arm == "A0":
                continue
            stage = output / "variants" / arm
            adapter = make_adapter(qwen, arm, config)
            torch.cuda.reset_peak_memory_stats()
            arm_examples = [(im, q, "CLEAN", answer) for im, q, _, answer in examples] if arm in EXTRA_LABELS else examples
            history = train(qwen, arm, adapter, cache, arm_examples, config, stage / "checkpoint.pt", fingerprint)
            stats[arm] = {
                "parameters": sum(p.numel() for p in adapter.parameters()),
                "training_condition": "CLEAN-only" if arm in EXTRA_LABELS else "CLEAN/LR4",
                "steps": len(history), "examples": len(examples), "history": history,
                "training_seconds": sum(h["seconds"] for h in history),
                "peak_allocated_bytes": max(h.get("peak_allocated_bytes", 0) for h in history),
            }
            write_json(stage / "training.json", stats[arm])
            del adapter

        for arm in labels:
            stage = output / "variants" / arm
            adapter = make_adapter(qwen, arm, config)
            if adapter is not None:
                checkpoint = torch.load(stage / "checkpoint.pt", map_location="cpu", weights_only=True)
                adapter.load_state_dict(checkpoint["adapter"])
                del checkpoint
            else:
                stats[arm] = {"parameters": 0}
            torch.cuda.reset_peak_memory_stats()
            rows.extend(predict(qwen, arm, adapter, cache, manifest, stage, scorer))
            stats[arm]["evaluation_peak_allocated_bytes_this_job"] = torch.cuda.max_memory_allocated()
            del adapter

        write_report(output, manifest, rows, stats, examples, config)
        artifacts = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and not p.name.startswith(".")}
        artifacts.update({p.relative_to(output).as_posix(): sha256(p)
                          for name in labels if name != "A0"
                          for p in (output / "variants" / name).rglob("*")
                          if p.is_file() and (p.name in {"checkpoint.pt", "training.json"} or "heatmaps" in p.parts)})
        write_json(output / "COMPLETE.json", {"stage": "s1", "binding": fingerprint, "artifacts": artifacts})
        verified_complete(output, "s1", fingerprint)
        print(f"Completed S1: {output}", flush=True)
