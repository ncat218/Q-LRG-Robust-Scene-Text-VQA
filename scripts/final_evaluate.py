"""Evaluate locked A0, S1b, Pre and A2 checkpoints on one common TextVQA split."""

import argparse
import importlib.util
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from filelock import FileLock

from qlrg.common import dataset_paths, object_hash, read_config, read_json, sha256, write_json
from qlrg.evaluation import SCORER_REVISION, evaluator
from qlrg.experiment import d0_identity, paired_images, verified_complete
from qlrg.qwen_wrapper import Qwen
from qlrg.s1_manifests import read_s1_manifests
from qlrg.s1_model import VisualResidual
from qlrg.s1_run import make_adapter, runtime_shared
from scripts.compute_cost import count_flops
from scripts.final_review import LABELS, demo, index_rows, report
from scripts.followup_common import bind, finish, write_table


def verified_pre(path):
    complete, binding = read_json(path / "COMPLETE.json"), read_json(path / "run.json")
    if complete["stage"] != "pre_11305" or complete["binding"] != object_hash(binding):
        raise ValueError("Pre run has an invalid completion marker")
    for name, digest in complete["artifacts"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Modified Pre artifact: {name}")
    if sha256(path / "pre_11305.py") != binding["worker_sha256"]:
        raise ValueError("Pre implementation differs from the trained run")
    spec = importlib.util.spec_from_file_location("qlrg_locked_pre", path / "pre_11305.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return binding, module


def prepare(output, require_runtime=True):
    launch = read_json(output / "launch.json")
    source = Path(launch["s1_run"]) / "s1"
    trained = verified_complete(source, "s1")
    pre_binding, module = verified_pre(Path(launch["pre_run"]))
    if pre_binding["source"] != object_hash(trained):
        raise ValueError("Pre was trained from another S1 source")
    paths = dataset_paths(read_config(launch["paths"]))
    config = read_config("configs/model_qwen.yaml")
    manifests, hashes = read_s1_manifests(Path(trained["dataset_dir"]), paths["textvqa"])
    if hashes != trained["dataset_hashes"] or config != trained["shared"]["model"]:
        raise ValueError("Dataset or Qwen model differs from the locked training run")
    manifest = manifests[launch["split"]]
    expected_sizes = {"val_inner": (2195, 3473), "official_val": (3166, 5000)}
    size = len(manifest["images"]), sum(len(image["questions"]) for image in manifest["images"])
    if size != expected_sizes[launch["split"]]:
        raise ValueError("Unexpected evaluation cohort")
    states, provenance = {}, {}
    checkpoints = {"S1b": source / "variants/S1b/checkpoint.pt", "A2": source / "variants/A2/checkpoint.pt",
                   "Pre": Path(launch["pre_run"]) / "pre.pt"}
    for arm, path in checkpoints.items():
        state = torch.load(path, map_location="cpu", weights_only=True)
        expected_binding = object_hash(pre_binding if arm == "Pre" else trained)
        expected_arm = "S1b" if arm == "Pre" else arm
        if (state["binding"] != expected_binding or state["arm"] != expected_arm or
                state["examples"] != 62258 or len(state["history"]) != 15565):
            raise ValueError(f"Checkpoint provenance or training budget mismatch: {arm}")
        states[arm] = state["adapter"]
        provenance[arm] = {"path": str(path), "sha256": sha256(path), "training_binding": expected_binding,
                           "example_visits": state["examples"]}
    shared = runtime_shared(paths, config) if require_runtime else pre_binding["runtime"]
    if shared != pre_binding["runtime"]:
        raise ValueError("Use the pinned runtime from job 48633")
    binding = {"stage": "final_study", "shared": shared, "split": launch["split"],
               "manifest_hash": object_hash(manifest), "checkpoints": provenance,
               "seed": launch["seed"], "scorer_revision": SCORER_REVISION,
               "worker_sha256": sha256(Path(__file__)),
               "review_sha256": sha256(Path(__file__).with_name("final_review.py")),
               "evaluation_role": "matched_checkpoint_comparison",
               "validation_history": "Official validation has been used in prior project experiments.",
               "cost_images": launch["cost_images"], "flops_images": launch["flops_images"]}
    return launch, paths, config, trained, manifest, module, states, binding


def load_models(model, module, paths, output, trained, states):
    cache = module.PreFeatures(model, paths["textvqa"], output, trained["shared"])
    adapters = {"A0": None}
    for arm in ("S1b", "A2"):
        adapters[arm] = make_adapter(model, arm, trained["config"])
        adapters[arm].load_state_dict(states[arm], strict=True)
    pre = module.PreAdapter(cache, model.model.config.vision_config.hidden_size, 318,
                            trained["config"]["spatial_blocks"]).to(model.device)
    pre.load_state_dict(states["Pre"], strict=True)
    adapters["Pre"] = pre
    for adapter in adapters.values():
        if adapter is not None:
            adapter.eval().requires_grad_(False)
    return cache, adapters


@torch.inference_mode()
def predict_image(cache, image, adapters, scorer):
    cache.load(image)
    rows = []
    for condition, base in cache.features.items():
        features = {"A0": base, "A2": base}
        for arm in ("S1b", "Pre"):
            features[arm] = adapters[arm](base, cache.grid)[0].to(cache.qwen.dtype)
        for question in image["questions"]:
            batch = cache.prompt(condition, question["question"])
            refs = scorer._compute_answer_scores(question["answers"])
            for arm in LABELS:
                context = adapters["A2"].attached() if arm == "A2" else nullcontext()
                with context:
                    answer = cache.qwen.generate(batch, features[arm])
                normalized = scorer.answer_processor(answer["answer"])
                rows.append({**answer, "condition": condition, "arm": arm, "image_id": image["image_id"],
                             "question_id": question["question_id"], "normalized_answer": normalized,
                             "visual_tokens": len(base), "score": refs.get(normalized, 0.0)})
    return rows


@torch.inference_mode()
def visual_features(model, batch, adapter, arm):
    if arm != "Pre":
        base = model.encode(batch)
        grid = tuple((batch["image_grid_thw"][0, 1:] // model.model.model.visual.spatial_merge_size).tolist())
        return base if arm != "S1b" else adapter(base, grid)[0].to(model.dtype)
    visual = model.model.model.visual
    _, h, w = batch["image_grid_thw"][0].tolist()
    merge = visual.spatial_merge_size
    window, _ = visual.get_window_index(batch["image_grid_thw"])
    window = window.to(model.device)
    grouped = torch.arange(h * w, device=model.device).reshape(h // merge, merge, w // merge, merge)
    grouped = grouped.permute(0, 2, 1, 3).reshape(-1, merge ** 2)
    order = grouped[window].flatten()
    inverse = order.argsort()

    def restore(_, args):
        raw = args[0]
        restored, _ = VisualResidual.forward(adapter, raw[inverse], (h, w))
        return (restored[order].to(raw.dtype),)

    handle = visual.merger.register_forward_pre_hook(restore)
    try:
        return model.encode(batch)
    finally:
        handle.remove()


def timed(operation):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    return result, 1000 * (time.perf_counter() - start)


@torch.inference_mode()
def profile(model, cache, adapters, manifest, root, count, flops_count):
    images = manifest["images"]
    sample = [images[i * len(images) // count] for i in range(count)]
    rows = []
    for index, image in enumerate([*sample[:2], *sample]):
        _, lr4 = paired_images(root, image)
        question = image["questions"][0]["question"]
        arms = list(LABELS)
        offset = index % len(arms)
        for arm in arms[offset:] + arms[:offset]:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            batch = model.prepare(lr4, question)
            adapter = adapters[arm]
            feature, visual_ms = timed(lambda: visual_features(model, batch, adapter, arm))
            # Check that training/evaluation and the timed native path agree on the same image.
            if index == 0:
                cache.load(image)
                base = cache.features["LR4"]
                expected = base if arm in ("A0", "A2") else adapter(base, cache.grid)[0].to(model.dtype)
                if not torch.equal(feature, expected):
                    raise RuntimeError(f"Timed and evaluated paths differ: {arm}")
            context = adapters["A2"].attached() if arm == "A2" else nullcontext()
            with context:
                answer = model.generate(batch, feature)
            request_ms = 1000 * (time.perf_counter() - started)
            peak = torch.cuda.max_memory_allocated()
            if index < 2:
                continue
            row = {"arm": arm, "image_id": image["image_id"], "visual_tokens": len(feature),
                   "answer_tokens": len(answer["token_ids"]), "visual_ms": visual_ms, "request_ms": request_ms,
                   "peak_allocated_bytes": peak, "registered_flops_decode8": None}
            if index - 2 < flops_count:
                def fixed_decode():
                    value = visual_features(model, batch, adapter, arm)
                    with model.replacement(batch, value):
                        return model.model.generate(**batch, do_sample=False, num_beams=1, use_cache=True,
                                                    min_new_tokens=8, max_new_tokens=8,
                                                    repetition_penalty=model.config["repetition_penalty"])
                with adapters["A2"].attached() if arm == "A2" else nullcontext():
                    _, row["registered_flops_decode8"] = count_flops(fixed_decode)
            rows.append(row)
    return rows


def cost_summary(rows):
    output = []
    for arm in LABELS:
        values = [row for row in rows if row["arm"] == arm]
        flops = [row["registered_flops_decode8"] for row in values if row["registered_flops_decode8"] is not None]
        row = {"arm": arm, "timed_images": len(values), "flops_images": len(flops),
               "peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in values),
               "mean_visual_tokens": sum(item["visual_tokens"] for item in values) / len(values),
               "mean_registered_flops_decode8": sum(flops) / len(flops)}
        for metric in ("visual_ms", "request_ms"):
            p50, p95 = torch.tensor([item[metric] for item in values]).quantile(torch.tensor([0.5, 0.95])).tolist()
            row[f"{metric}_p50"], row[f"{metric}_p95"] = p50, p95
        output.append(row)
    return output


def run(output, check=False):
    if not check and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run final evaluation in a Slurm GPU allocation")
    launch, paths, config, trained, manifest, module, states, binding = prepare(output, not check)
    if check:
        print(f"Checkpoint provenance verified. {len(manifest['images'])} images on {launch['split']}.", flush=True)
        return
    with FileLock(str(output / ".run.lock"), timeout=0):
        bind(output, binding)
        if (output / "COMPLETE.json").exists():
            marker = read_json(output / "COMPLETE.json")
            if marker["binding"] != object_hash(binding):
                raise ValueError("Completion binding changed")
            for name, digest in marker["artifacts"].items():
                if sha256(output / name) != digest:
                    raise ValueError(f"Changed artifact: {name}")
            print("Verified complete evaluation", flush=True)
            return
        write_json(output / "eval_manifest.json", manifest)
        model = Qwen(config, paths["hf_hub"])
        d0_identity(model, manifest, paths["textvqa"], 4, output / "identity.json", require_decode=False)
        cache, adapters = load_models(model, module, paths, output, trained, states)
        scorer = evaluator()
        rows = []
        for index, image in enumerate(manifest["images"], 1):
            path = output / "shards" / f"{image['image_id']}.json"
            one = {**manifest, "images": [image]}
            if path.exists():
                image_rows = read_json(path)
                index_rows(image_rows, one)
            else:
                image_rows = predict_image(cache, image, adapters, scorer)
                index_rows(image_rows, one)
                write_json(path, image_rows)
            rows.extend(image_rows)
            if index % 100 == 0 or index == len(manifest["images"]):
                print(f"Matched evaluation: {index}/{len(manifest['images'])} images", flush=True)
        summary = report(output, rows, manifest, launch["seed"])
        summary["parameters"] = {arm: 0 if adapter is None else sum(p.numel() for p in adapter.parameters())
                                 for arm, adapter in adapters.items()}
        summary["training_budget"] = {"images": 19758, "epochs": 2, "example_visits": 62258,
                                      "optimizer_steps": 15565, "seed": 42}
        demo(output, rows, manifest, paths["textvqa"])
        costs = profile(model, cache, adapters, manifest, paths["textvqa"], launch["cost_images"], launch["flops_images"])
        write_table(output / "cost.csv", costs)
        write_table(output / "cost_summary.csv", cost_summary(costs))
        summary["cost_policy"] = {
            "condition": "LR4", "batch_size": 1, "warmup_images": 2, "timed_images": launch["cost_images"],
            "flops_images": launch["flops_images"], "fixed_decode_tokens_for_flops": 8,
            "flops_coverage": "Registered PyTorch operators only; unsupported kernels are omitted.",
            "vram_scope": "Peak allocated inference VRAM with all compared adapters resident."}
        finish(output, binding, summary, rows)
        marker = read_json(output / "COMPLETE.json")
        for path in output.glob("*.html"):
            marker["artifacts"][path.name] = sha256(path)
        write_json(output / "COMPLETE.json", marker)
        print(f"Matched evaluation and demo complete: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.output, args.check)
