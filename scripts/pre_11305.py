"""Train Pre-Merger restoration with job 11305\'s S1b protocol."""

import argparse
import os
from pathlib import Path

import torch

from qlrg.common import dataset_paths, object_hash, read_config, read_json, sha256, write_json, write_jsonl
from qlrg.evaluation import evaluator, paired_bootstrap
from qlrg.experiment import verified_complete
from qlrg.qwen_wrapper import Qwen
from qlrg.s1_data import training_examples
from qlrg.s1_manifests import read_s1_manifests
from qlrg.s1_model import VisualResidual, text_training
from qlrg.s1_run import ImageFeatures, loss_for, make_adapter, runtime_shared, train


class PreFeatures(ImageFeatures):
    def load(self, image):
        if self.current == image["image_id"]:
            return
        visual = self.qwen.model.model.visual
        captured = []
        handle = visual.merger.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
        try:
            super().load(image)
            if len(captured) != 2:
                captured.clear()
                question = image["questions"][0]["question"]
                for condition in ("CLEAN", "LR4"):
                    fresh = self.qwen.encode(self.prompt(condition, question))
                    if not torch.equal(fresh, self.features[condition]):
                        raise RuntimeError("Cached and recomputed features differ")
        finally:
            handle.remove()
        if len(captured) != 2:
            raise RuntimeError("Expected both CLEAN and LR4 features")
        self.raw = {name: raw.clone() for name, raw in zip(("CLEAN", "LR4"), captured)}
        t, h, w = self.thw
        merge = visual.spatial_merge_size
        if t != 1 or h % merge or w % merge:
            raise ValueError("Expected one aligned still image")
        batch = self.prompt("LR4", image["questions"][0]["question"])
        window, _ = visual.get_window_index(batch["image_grid_thw"])
        window = window.to(self.qwen.device)
        raster = torch.arange(h * w, device=window.device).reshape(h // merge, merge, w // merge, merge)
        grouped = raster.permute(0, 2, 1, 3).reshape(-1, merge ** 2)
        self.order = grouped[window].flatten()
        self.inverse = self.order.argsort()
        self.reverse_windows = window.argsort()
        self.pre_grid = (h, w)


class PreAdapter(VisualResidual):
    def __init__(self, cache, width, hidden, blocks):
        super().__init__(width, hidden, blocks)
        self.cache = cache

    def forward(self, base, grid):
        condition = "CLEAN" if base is self.cache.features["CLEAN"] else "LR4"
        raw = self.cache.raw[condition]
        raster, _ = super().forward(raw[self.cache.inverse], self.cache.pre_grid)
        merger = self.cache.qwen.model.model.visual.merger
        restored = merger(raster[self.cache.order].to(raw.dtype))[self.cache.reverse_windows].float()
        return restored, restored - base.float()


@torch.inference_mode()
def assess(qwen, cache, manifest, arms):
    scorer = evaluator()
    rows = []
    for index, image in enumerate(manifest["images"], 1):
        cache.load(image)
        for condition, base in cache.features.items():
            features = {name: base if adapter is None else adapter(base, cache.grid)[0].to(qwen.dtype)
                        for name, adapter in arms.items()}
            for question in image["questions"]:
                batch = cache.prompt(condition, question["question"])
                refs = scorer._compute_answer_scores(question["answers"])
                for name, feature in features.items():
                    answer = qwen.generate(batch, feature)
                    rows.append({**answer, "arm": name, "condition": condition, "image_id": image["image_id"],
                                 "question_id": question["question_id"],
                                 "score": refs.get(scorer.answer_processor(answer["answer"]), 0.0)})
        if index % 100 == 0 or index == len(manifest["images"]):
            print(f"Evaluation {index}/{len(manifest['images'])}", flush=True)
    return rows


def run(output, source, cpu_check):
    old = verified_complete(source, "s1")
    config = old["config"]
    plan = read_json(source / "train_manifest.json")
    manifest = read_json(source / "eval_manifest.json")
    if object_hash(plan) != old["train_manifest"] or object_hash(manifest) != old["eval_manifest"]:
        raise ValueError("Manifest does not match job 11305")
    paths = dataset_paths(read_config(Path("configs/paths_hpc.yaml")))
    model_config = read_config(Path("configs/model_qwen.yaml"))
    _, hashes = read_s1_manifests(Path(old["dataset_dir"]), paths["textvqa"])
    if hashes != old["dataset_hashes"] or model_config != old["shared"]["model"]:
        raise ValueError("Dataset or model config changed")
    examples = list(training_examples(plan, config["epochs"], config["seed"], config["clean_fraction"]))
    if (len(plan), len(examples), config["epochs"]) != (19758, 62258, 2):
        raise ValueError("Unexpected job 11305 training budget")
    old_checkpoint = source / "variants/S1b/checkpoint.pt"
    state = torch.load(old_checkpoint, map_location="cpu", weights_only=True)
    if state["binding"] != object_hash(old) or state["arm"] != "S1b" or state["examples"] != len(examples):
        raise ValueError("Invalid S1b checkpoint provenance")
    if cpu_check:
        print("Verified job 11305 checkpoint and exact train/eval manifests", flush=True)
        return

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run training inside a Slurm GPU allocation")
    shared = runtime_shared(paths, model_config)
    storage = read_json(source / "storage.json")
    if shared["source_hash"] != storage["runtime_source_hash"] or any(
        shared[key] != value for key, value in old["shared"].items() if key not in ("source_hash", "a0_path_hash")
    ):
        raise ValueError("Runtime differs from the resumed job 11305")
    os.environ["QLRG_FEATURE_CACHE_MODE"] = "read-only"
    os.environ["QLRG_FEATURE_CACHE"] = storage["cache"]
    os.environ["QLRG_CHECKPOINT_SAVES"] = "5"
    binding = {"stage": "pre_11305", "runtime": shared, "source": object_hash(old),
               "checkpoint_sha256": sha256(old_checkpoint), "worker_sha256": sha256(Path(__file__))}
    write_json(output / "run.json", binding)
    fingerprint = object_hash(binding)
    qwen = Qwen(model_config, paths["hf_hub"])
    pre_cache = PreFeatures(qwen, paths["textvqa"], output, old["shared"])
    historical = make_adapter(qwen, "S1b", config).eval()
    historical.load_state_dict(state["adapter"])
    del state
    torch.manual_seed(config["seed"])
    pre = PreAdapter(pre_cache, qwen.model.config.vision_config.hidden_size, 318, config["spatial_blocks"]).to(qwen.device)
    with torch.inference_mode():
        for image in manifest["images"][:4]:
            pre_cache.load(image)
            for condition, base in pre_cache.features.items():
                patched = pre(base, pre_cache.grid)[0].to(qwen.dtype)
                if not torch.equal(base, patched):
                    raise RuntimeError(f"Pre identity changed visual features for {condition}")
                batch = pre_cache.prompt(condition, image["questions"][0]["question"])
                if not torch.equal(qwen.prefill_logits(batch, patched), qwen.prefill_logits(batch)):
                    raise RuntimeError(f"Pre identity changed prefill logits for {condition}")

    image, question, condition, answer = next(row for row in examples if row[2] == "LR4")
    pre_cache.load(image)
    with text_training(qwen.model, True):
        loss_for(qwen, "S1b", pre, pre_cache, question, condition, answer, config).backward()
    gradient = pre.output.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().max().item() == 0:
        raise RuntimeError("No finite gradient through the frozen Merger")
    pre.zero_grad(set_to_none=True)

    counts = {name: sum(p.numel() for p in adapter.parameters())
              for name, adapter in (("Pre", pre), ("S1b_11305", historical))}
    if abs(counts["Pre"] / counts["S1b_11305"] - 1) > 0.002:
        raise ValueError("Unexpected parameter mismatch")
    print(f"Parity and gradient checks passed. Train 19,758 images for two epochs. Parameters: {counts}", flush=True)
    history = train(qwen, "S1b", pre, pre_cache, examples, config, output / "pre.pt", fingerprint)
    if len(history) != 15565 or history[-1]["examples"] != len(examples):
        raise RuntimeError("Pre training did not complete the same update budget")
    write_json(output / "pre_training.json", {"steps": len(history), "history": history})
    pre.eval()
    arms = {"A0": None, "S1b_11305": historical, "Pre": pre}
    rows = assess(qwen, pre_cache, manifest, arms)
    expected = [(image["image_id"], q["question_id"]) for image in manifest["images"] for q in image["questions"]]
    groups = [image for image, _ in expected]
    comparisons = []
    for condition in ("CLEAN", "LR4"):
        scores = {}
        for name in arms:
            subset = [r for r in rows if r["condition"] == condition and r["arm"] == name]
            if [(r["image_id"], r["question_id"]) for r in subset] != expected:
                raise RuntimeError(f"Evaluation cohort changed for {condition}/{name}")
            scores[name] = [r["score"] for r in subset]
        for reference, target in (("A0", "S1b_11305"), ("A0", "Pre"), ("S1b_11305", "Pre")):
            left, right = scores[reference], scores[target]
            comparisons.append({"condition": condition, "reference": reference, "target": target,
                                "reference_pct": 100 * sum(left) / len(left), "target_pct": 100 * sum(right) / len(right),
                                **paired_bootstrap(left, right, groups, 10000, config["seed"])})
    write_jsonl(output / "predictions.jsonl", rows)
    write_json(output / "summary.json", {"questions": len(expected), "images": len(manifest["images"]), "parameters": counts,
                                               "training_examples": len(examples), "epochs": config["epochs"],
                                               "steps": len(history), "comparisons": comparisons})
    artifacts = {name: sha256(output / name) for name in ("predictions.jsonl", "summary.json", "pre_training.json")}
    write_json(output / "COMPLETE.json", {"stage": binding["stage"], "binding": fingerprint, "artifacts": artifacts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    run(args.output, args.source, args.cpu_check)
