"""Run one post-Merger ablation using the frozen Qwen and audited S1 split."""

import argparse
import os
from pathlib import Path

import torch
from filelock import FileLock

from qlrg.common import dataset_paths, object_hash, read_config, read_json, sha256, write_json
from qlrg.data import verify_pillow
from qlrg.evaluation import evaluator
from qlrg.experiment import d0_identity, paired_images, verified_complete
from qlrg.grid_alignment import image_grid
from qlrg.qwen_wrapper import Qwen, assert_aligned
from qlrg.s1_data import training_examples
from qlrg.s1_manifests import read_s1_manifests
from qlrg.s1_run import make_adapter, runtime_shared, train
from scripts.followup_common import bind, finish, write_table
from scripts.latent_methods import METHODS, make_restorer, study_loss, text_mask


ROOT = Path(__file__).resolve().parents[1]


class StudyFeatures:
    """Encode a matched pair in memory, with no persistent visual feature cache."""

    def __init__(self, qwen, root, ocr):
        self.qwen, self.root, self.ocr = qwen, root, ocr
        self.current = None

    def load(self, image):
        if image["image_id"] == self.current:
            return
        clean, lr4 = paired_images(self.root, image)
        self.images = {"CLEAN": clean, "LR4": lr4}
        question = image["questions"][0]["question"]
        batches = {name: self.qwen.prepare(value, question) for name, value in self.images.items()}
        assert_aligned(batches["CLEAN"], batches["LR4"])
        self.grid = image_grid(batches["LR4"]["image_grid_thw"][0].tolist())
        self.features = {name: self.qwen.encode(value).clone() for name, value in batches.items()}
        self.mask = text_mask(self.ocr, image, self.grid, self.qwen.device) if self.ocr is not None else None
        self.current = image["image_id"]

    def prompt(self, condition, question):
        batch = self.qwen.prepare(self.images[condition], question)
        if image_grid(batch["image_grid_thw"][0].tolist()) != self.grid:
            raise RuntimeError("Question changed the aligned visual grid")
        return batch


def training_scale(cache, images):
    total, squares, count = 0, 0, 0
    for image in images[:128]:
        cache.load(image)
        clean = cache.features["CLEAN"].float()
        total = total + clean.sum(0)
        squares = squares + clean.square().sum(0)
        count += len(clean)
    return (squares / count - (total / count).square()).clamp_min(1e-6).sqrt()


@torch.inference_mode()
def predict(cache, manifest, scorer, adapter, lora):
    rows = []
    adapter.eval()
    for index, image in enumerate(manifest["images"], 1):
        cache.load(image)
        for condition, base in cache.features.items():
            restored, _ = adapter(base, cache.grid)
            for question in image["questions"]:
                batch = cache.prompt(condition, question["question"])
                refs = scorer._compute_answer_scores(question["answers"])
                for arm, feature in (("A0", base), ("latent", restored), ("A2", base), ("A2_latent", restored)):
                    if arm.startswith("A2"):
                        with lora.attached():
                            answer = cache.qwen.generate(batch, feature.to(cache.qwen.dtype))
                    else:
                        answer = cache.qwen.generate(batch, feature.to(cache.qwen.dtype))
                    rows.append({**answer, "image_id": image["image_id"], "question_id": question["question_id"],
                                 "condition": condition, "arm": arm,
                                 "score": refs.get(scorer.answer_processor(answer["answer"]), 0.0)})
        if index % 50 == 0 or index == len(manifest["images"]):
            print(f"Evaluation: {index}/{len(manifest['images'])} images", flush=True)
    return rows


def summarize(rows, manifest, seed):
    from qlrg.evaluation import paired_bootstrap

    expected = [(im["image_id"], q["question_id"]) for im in manifest["images"] for q in im["questions"]]
    groups = [iid for iid, _ in expected]
    result = []
    for condition in ("CLEAN", "LR4"):
        arms = {}
        for arm in ("A0", "latent", "A2", "A2_latent"):
            subset = [r for r in rows if r["condition"] == condition and r["arm"] == arm]
            if [(r["image_id"], r["question_id"]) for r in subset] != expected:
                raise RuntimeError(f"Cohort changed for {condition}/{arm}")
            arms[arm] = [r["score"] for r in subset]
        for ref, target in (("A0", "latent"), ("A0", "A2"), ("A2", "A2_latent"), ("A2", "latent")):
            left, right = arms[ref], arms[target]
            result.append({"condition": condition, "reference": ref, "target": target, "questions": len(expected),
                           "reference_pct": 100 * sum(left) / len(left),
                           "target_pct": 100 * sum(right) / len(right),
                           **paired_bootstrap(left, right, groups, 10000, seed)})
    return result


def run(output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run within a Slurm GPU allocation")
    verify_pillow()
    args = read_json(output / "launch.json")
    source = Path(args["s1_run"]) / "s1"
    trained = verified_complete(source, "s1")
    paths = dataset_paths(read_config(ROOT / "configs/paths_hpc.yaml"))
    model_config = read_config(ROOT / "configs/model_qwen.yaml")
    manifests, hashes = read_s1_manifests(Path(trained["dataset_dir"]), paths["textvqa"])
    shared = runtime_shared(paths, model_config)
    if hashes != trained["dataset_hashes"] or any(
        trained["shared"].get(key) != value for key, value in shared.items() if key != "source_hash"
    ):
        raise ValueError("Study requires the audited S1 runtime and manifests")
    config = {**trained["config"], "seed": args["seed"], "epochs": args["epochs"]}
    train_images = manifests["train_inner"]["images"]
    if not 1 <= args["train_images"] <= len(train_images):
        raise ValueError("Training image count exceeds train_inner")
    plan = train_images[:args["train_images"]]
    dev = manifests["val_inner"]
    if args["eval_images"] is not None:
        dev = {**dev, "images": dev["images"][:args["eval_images"]]}
    ocr = read_json(paths["textocr"]) if args["method"] in ("text_weighted", "text_relational", "text_gate") else None
    examples = list(training_examples(plan, config["epochs"], config["seed"], config["clean_fraction"]))
    config["checkpoint_every"] = max(1, (len(examples) + 5 * config["accumulate"] - 1)
                                     // (5 * config["accumulate"]))
    checkpoint = source / "variants/A2/checkpoint.pt"
    binding = {"stage": "latent_study", "shared": shared, "method": args["method"], "config": config,
               "train_hash": object_hash(plan), "eval_hash": object_hash(dev), "a2_sha256": sha256(checkpoint),
               "ocr_sha256": sha256(paths["textocr"]) if ocr is not None else None,
               "method_sha256": sha256(ROOT / "scripts/latent_methods.py"),
               "study_sha256": sha256(Path(__file__)),
               "citation_registry_sha256": sha256(ROOT / "docs/latent_restoration_study.md")}
    with FileLock(str(output / ".run.lock"), timeout=0):
        bind(output, binding)
        qwen = Qwen(model_config, paths["hf_hub"])
        d0_identity(qwen, dev, paths["textvqa"], min(4, len(dev["images"])),
                    output / "identity.json", require_decode=False)
        cache = StudyFeatures(qwen, paths["textvqa"], ocr)
        scale = training_scale(cache, plan) if args["method"] == "bounded" else None
        width = qwen.model.config.text_config.hidden_size
        adapter = make_restorer(width, config, args["method"], qwen.device, scale)
        history = train(qwen, args["method"], adapter, cache, examples, config,
                        output / "checkpoint.pt", object_hash(binding), objective=study_loss)
        lora = make_adapter(qwen, "A2", trained["config"]).eval().requires_grad_(False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["binding"] != object_hash(trained) or state["arm"] != "A2":
            raise ValueError("A2 checkpoint does not belong to the reference S1 run")
        lora.load_state_dict(state["adapter"])
        cache.ocr = None
        rows = predict(cache, dev, evaluator(), adapter, lora)
        comparisons = summarize(rows, dev, config["seed"])
        write_table(output / "paired.csv", comparisons)
        write_json(output / "training.json", {"examples": len(examples), "steps": len(history),
                                              "history": history, "checkpoint_sha256": sha256(output / "checkpoint.pt")})
        finish(output, binding, {"method": args["method"], "split": "val_inner",
                                 "train_images": len(plan), "comparisons": comparisons}, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
