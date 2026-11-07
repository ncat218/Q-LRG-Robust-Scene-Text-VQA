import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from qlrg.a0_audit import audit_summary
from qlrg.common import read_json, write_json
from qlrg.s1_manifests import build_s1_manifests, check_disjoint, read_s1_manifests
from qlrg.s1_report import degradation_interactions


class ExpandedS1Test(unittest.TestCase):
    def test_independent_manifests_include_all_images_without_textocr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "train_images"
            images.mkdir()
            for split, start, count in (("train", 0, 20), ("val", 20, 3)):
                records = []
                for i in range(start, start + count):
                    iid = str(i)
                    Image.new("RGB", (16, 16), color=(i * 10, i, 0)).save(images / f"{iid}.jpg")
                    records.append({"image_id": iid, "question_id": i, "question": "What?",
                                    "answers": ["yes"] * 10})
                write_json(root / f"TextVQA_0.5.1_{split}.json", {"data": records})
            output = root / "manifests"
            with patch("qlrg.s1_manifests.verify_pillow"):
                build_s1_manifests(root, output, 42)
                build_s1_manifests(root, output, 42)
            manifests, hashes = read_s1_manifests(output, root)
            self.assertEqual({k: len(m["images"]) for k, m in manifests.items()},
                             {"train_inner": 18, "val_inner": 2, "official_val": 3})
            self.assertTrue(all("words" not in row for m in manifests.values() for row in m["images"]))
            duplicate = copy.deepcopy(manifests["official_val"])
            duplicate["images"][0]["image_sha256"] = manifests["train_inner"]["images"][0]["image_sha256"]
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                check_disjoint([manifests["train_inner"], duplicate])
            data = read_json(output / "train_inner.json")
            data["seed"] = 43
            write_json(output / "train_inner.json", data)
            with self.assertRaisesRegex(ValueError, "changed"):
                read_s1_manifests(output, root)

    def test_audit_detects_token_differences_even_when_scores_agree(self):
        manifest = {"images": [{"image_id": "img", "questions": [{"question_id": "q"}]}]}
        rows = [
            {"image_id": "img", "question_id": "q", "condition": condition, "path": path,
             "repeat": repeat, "token_ids": [1, 2], "answer": "yes", "score": 1.0}
            for condition in ("CLEAN", "LR4") for path in ("native", "injected") for repeat in (1, 2)
        ]
        self.assertTrue(audit_summary(rows, manifest, repeats=10)["passed"])
        rows[-1]["token_ids"] = [1, 3]
        summary = audit_summary(rows, manifest, repeats=10)
        self.assertTrue(summary["passed"])
        self.assertFalse(summary["exact_match"])
        self.assertTrue(all(c["different_scores"] == 0 for c in summary["contrasts"]))
        diagnostics = [{"cached_equals_fresh": True, "prefill_equal": False}]
        self.assertFalse(audit_summary(rows, manifest, repeats=10, diagnostics=diagnostics)["passed"])
        rows[-2]["token_ids"] = [1, 3]
        self.assertFalse(audit_summary(rows, manifest, repeats=10)["passed"])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            audit_summary(rows[:-1], manifest, repeats=10)

    def test_interaction_bootstraps_four_paired_scores(self):
        # Gains are identical on CLEAN/LR4, so the interaction and its interval are exactly zero.
        rows = [
            {"image_id": str(i), "question_id": str(i), "condition": c, "arm": a,
             "score": (0.1 if c == "CLEAN" else 0.0) + (0.3 if a == "S1b" else 0.0)}
            for i in range(3) for c in ("CLEAN", "LR4") for a in ("A0", "S1b")
        ]
        value, = degradation_interactions(rows, [("A0", "S1b")], 100, 42)
        self.assertAlmostEqual(value["delta_pp"], 0.0)
        self.assertAlmostEqual(value["ci_low_pp"], 0.0)
        self.assertAlmostEqual(value["ci_high_pp"], 0.0)

    def test_audit_runs_two_complete_passes_and_resumes_on_cpu(self):
        import os
        from types import SimpleNamespace
        from qlrg.a0_audit import run_audit
        from qlrg.common import sha256
        from qlrg.experiment import verified_complete

        class Scorer:
            def _compute_answer_scores(self, answers):
                return {"yes": 1.0}

            def answer_processor(self, answer):
                return answer

        class FakeQwen:
            def __init__(self, *args):
                pass

            def prepare(self, *args):
                return {}

            def generate(self, *args):
                return {"answer": "yes", "token_ids": [1, 2], "generation_seconds": 0.1}

        class FakeCache:
            def __init__(self, *args):
                self.features = {"CLEAN": torch.ones(1), "LR4": torch.ones(1)}
                self.thw = [1, 2, 2]
                self.current = None

            def load(self, image):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "TextVQA_0.5.1_train.json"
            write_json(annotation, {"data": []})
            source = root / "anchor"
            source.mkdir()
            write_json(source / "manifest.json", {
                "schema": 2, "split": "val_inner",
                "protocol": "rgb-raw-orientation-pillow11.3.0-bicubic-floor-div4-up-original-v1",
                "annotation_sha256": {"train": sha256(annotation)},
                "images": [{"image_id": "img", "image_rel": "train_images/img.jpg",
                            "questions": [{"question_id": "q", "question": "What?", "answers": ["yes"] * 10}]}],
            })
            write_json(source / "d0" / "predictions.jsonl", {})
            (source / "d0" / "predictions.jsonl").write_text(
                "\n".join(json.dumps({"image_id": "img", "question_id": "q", "arm": a,
                                       "answer": "yes", "score": 1.0, "token_ids": [1, 2]})
                          for a in ("CLEAN", "LR4-BIC-CONTROL")) + "\n"
            )
            output = root / "audit"
            with patch.dict(os.environ, {"SLURM_JOB_ID": "test"}), patch("qlrg.a0_audit.verify_pillow"), \
                    patch("qlrg.a0_audit.Qwen", FakeQwen), \
                    patch("qlrg.a0_audit.paired_images", return_value=(None, None)), \
                    patch("qlrg.a0_audit.evaluator", return_value=Scorer()), \
                    patch("qlrg.s1_run.ImageFeatures", FakeCache), \
                    patch("qlrg.s1_run.runtime_shared", return_value={"model": {}}):
                run_audit(source, output, {"textvqa": root, "hf_hub": root}, {})
                run_audit(source, output, {"textvqa": root, "hf_hub": root}, {})
            verified_complete(output, "audit")
            self.assertTrue(read_json(output / "summary.json")["passed"])
            self.assertEqual(read_json(output / "historical_comparisons.json"), {"d0": []})

    def test_launcher_snapshots_expanded_and_audit_commands(self):
        import sys
        import run_all
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = root / "old"
            write_json(anchor / "manifest.json", {})
            write_json(anchor / "s1" / "COMPLETE.json", {})
            data = root / "data"
            write_json(data / "COMPLETE.json", {})
            audit = root / "audit"
            write_json(audit / "COMPLETE.json", {})
            config = root / "config.json"
            write_json(config, {"seed": 42})
            cases = [
                ["--mode", "audit", "--from-run", str(anchor), "--pilot-run", str(anchor)],
                ["--mode", "s1", "--data-dir", str(data), "--baseline-audit", str(audit),
                 "--s1-config", str(config), "--time", "2-00:00:00"],
                ["--mode", "s1-eval", "--from-run", str(anchor)],
            ]
            for args in cases:
                with patch.object(run_all, "REPO", root), patch.object(sys, "argv", ["run_all.py", *args]), \
                        patch("run_all.subprocess.run") as submit:
                    run_all.main()
                    command = submit.call_args.args[0]
                    self.assertNotIn("--manifest", command)
                    run = Path(command[command.index("--run-dir") + 1])
                    mode = args[1]
                    script = (run / f"{mode}.sbatch").read_text()
                    if mode == "audit":
                        self.assertIn("audit-a0", script)
                        self.assertIn("build-s1-data", script)
                        self.assertIn("--pilot-run", script)
                    if mode == "s1":
                        self.assertEqual((run / "s1.yaml").read_bytes(), config.read_bytes())
                        self.assertIn("--data-dir", script)
                        self.assertEqual(command[-2:], ["--time", "2-00:00:00"])
                    with patch.object(sys, "argv", ["run_all.py", "--mode", mode, "--resume", str(run)]):
                        run_all.main()

    def test_expanded_training_and_locked_validation_entry_points(self):
        import os
        from contextlib import ExitStack
        from types import SimpleNamespace
        from qlrg.experiment import verified_complete
        from qlrg.s1_run import run_s1
        from qlrg.s1_evaluate import evaluate_s1

        def record(i):
            return {"image_id": str(i), "image_rel": f"train_images/{i}.jpg", "image_sha256": str(i),
                    "questions": [{"question_id": str(i), "question": "What?", "answers": ["yes"] * 10}]}

        config = json.loads(Path("configs/s1_expanded.json").read_text())
        datasets = {
            "train_inner": {"images": [record(1), record(2)]},
            "val_inner": {"images": [record(3)]}, "official_val": {"images": [record(4)]},
        }
        trained_conditions = {}

        def fake_train(qwen, arm, adapter, cache, examples, config, path, fingerprint):
            trained_conditions[arm] = [row[2] for row in examples]
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"adapter": adapter.state_dict()}, path)
            return [{"examples": len(examples), "loss": 1.0, "grad_norm": 1.0, "seconds": 0.1}]

        def fake_predict(qwen, arm, adapter, cache, manifest, stage, scorer):
            return [{"arm": arm, "condition": c, "image_id": im["image_id"], "question_id": q["question_id"],
                     "answer": "yes", "score": 1.0, "adapter_seconds_per_image": 0.0,
                     "generation_seconds": 0.1, "feature_mse_to_clean": 0.0, "feature_cosine_to_clean": 1.0}
                    for im in manifest["images"] for q in im["questions"] for c in ("CLEAN", "LR4")]

        qwen = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=8))))
        shared = {"model": {}, "paths": {}}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            output = root / "training" / "s1"
            for target, kwargs in (
                ("qlrg.s1_run.verify_pillow", {}),
                ("qlrg.s1_evaluate.verify_pillow", {}),
                ("qlrg.s1_run.evaluator", {"return_value": None}),
                ("qlrg.s1_evaluate.evaluator", {"return_value": None}),
                ("qlrg.s1_manifests.read_s1_manifests", {"return_value": (datasets, {"train_inner": "h1", "val_inner": "h2", "official_val": "h3"})}),
                ("qlrg.s1_evaluate.read_s1_manifests", {"return_value": (datasets, {"train_inner": "h1", "val_inner": "h2", "official_val": "h3"})}),
                ("qlrg.s1_run.runtime_shared", {"return_value": shared}),
                ("qlrg.s1_evaluate.runtime_shared", {"return_value": shared}),
                ("qlrg.a0_audit.validate_audit", {"return_value": ("audit-hash", {"policy": "within-path-repeatability-v2"})}),
                ("qlrg.s1_run.Qwen", {"return_value": qwen}),
                ("qlrg.s1_evaluate.Qwen", {"return_value": qwen}),
                ("qlrg.s1_run.ImageFeatures", {}),
                ("qlrg.s1_evaluate.ImageFeatures", {}),
                ("qlrg.s1_run.probe_gradients", {"return_value": {"passed": True}}),
                ("qlrg.s1_run.train", {"side_effect": fake_train}),
                ("qlrg.s1_run.predict", {"side_effect": fake_predict}),
                ("qlrg.s1_evaluate.predict", {"side_effect": fake_predict}),
                ("qlrg.s1_run.make_adapter", {"side_effect": lambda q, a, c: None if a == "A0" else torch.nn.Linear(1, 1)}),
                ("qlrg.s1_evaluate.make_adapter", {"side_effect": lambda q, a, c: None if a == "A0" else torch.nn.Linear(1, 1)}),
                ("torch.cuda.reset_peak_memory_stats", {}),
                ("torch.cuda.max_memory_allocated", {"return_value": 0}),
                ("qlrg.s1_run.d0_identity", {"side_effect": lambda q, m, r, n, path, **kwargs: write_json(path, {"passed": True})}),
            ):
                stack.enter_context(patch(target, **kwargs))
            stack.enter_context(patch.dict(os.environ, {"SLURM_JOB_ID": "test"}))
            run_s1(None, output, {"textvqa": root, "hf_hub": root}, {"max_pixels": 3136}, config,
                   dataset_dir=root / "dataset", audit_path=root / "audit")
            binding = verified_complete(output, "s1")
            self.assertEqual(binding["evaluation_role"], "hyperparameter_selection")
            self.assertEqual(set(trained_conditions), {"S1a", "S1b", "A1", "A2", "A1clean", "A2clean"})
            self.assertEqual(set(trained_conditions["A1clean"]), {"CLEAN"})
            self.assertEqual(set(trained_conditions["A2clean"]), {"CLEAN"})
            self.assertIn("LR4", trained_conditions["S1b"])
            evaluate_s1(root / "training", root / "official" / "s1_eval", {"textvqa": root, "hf_hub": root}, {})
            official = verified_complete(root / "official" / "s1_eval", "s1_eval")
            self.assertEqual(official["evaluation_role"], "official_val_locked")
            predictions = (root / "official" / "s1_eval" / "predictions.jsonl").read_text()
            self.assertNotIn('"image_id": "3"', predictions)


            trained_conditions.clear()
            sweep = root / "sweep" / "s1"
            short_config = {**config, "train_images": 2, "epochs": 1, "clean_controls": False,
                            "train_arms": ["S1b", "A1", "A2"]}
            run_s1(None, sweep, {"textvqa": root, "hf_hub": root}, {"max_pixels": 3136}, short_config,
                   dataset_dir=root / "dataset", audit_path=root / "audit")
            verified_complete(sweep, "s1")
            self.assertEqual(set(trained_conditions), {"S1b", "A1", "A2"})
            summary = read_json(sweep / "summary.json")
            self.assertEqual(set(summary["arm_labels"]), {"A0", "S1b", "A1", "A2"})

    def test_study_configs_vary_only_seed_and_learning_rate(self):
        from scripts.prepare_s1_study import prepare

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "base.json"
            write_json(template, {"epochs": 2, "seed": 1, "learning_rate": 0.1})
            output = root / "study"
            prepare(output, template, [42, 43, 44], [0.0001, 0.0002])
            metadata = read_json(output / "study.json")
            self.assertEqual(len(metadata["candidates"]), 6)
            self.assertTrue(all(read_json(output / c["config"])["epochs"] == 2 for c in metadata["candidates"]))
            self.assertEqual({c["seed"] for c in metadata["candidates"]}, {42, 43, 44})
            short = root / "short"
            prepare(short, template, [42], [0.0001, 0.0002, 0.0005], short_sweep=True)
            candidates = read_json(short / "study.json")["candidates"]
            self.assertEqual(len(candidates), 3)
            for candidate in candidates:
                config = read_json(short / candidate["config"])
                self.assertEqual((config["train_images"], config["epochs"]), (4000, 1))
                self.assertEqual(config["train_arms"], ["S1b", "A1", "A2"])
                self.assertFalse(config["clean_controls"])

    def test_clean_penalty_is_separate_from_lr4_penalty(self):
        from types import SimpleNamespace
        from qlrg.s1_run import loss_for

        base = torch.zeros(2, 3)
        adapter = lambda x, grid: (x + 1, torch.ones_like(x))
        qwen = SimpleNamespace(dtype=torch.float32, processor=SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda *a, **kw: [1], eos_token_id=2)))
        cache = SimpleNamespace(features={"CLEAN": base, "LR4": base}, grid=(1, 2),
                                prompt=lambda *a: {})
        config = {"max_answer_tokens": 4, "feature_weight": 1.0, "residual_weight": 0.01,
                  "clean_residual_weight": 0.1}
        with patch("qlrg.s1_run.teacher_forced_loss", return_value=torch.tensor(0.0)):
            clean = loss_for(qwen, "S1b", adapter, cache, {"question": "q"}, "CLEAN", "yes", config)
            lr4 = loss_for(qwen, "S1b", adapter, cache, {"question": "q"}, "LR4", "yes", config)
        self.assertAlmostEqual(float(clean - lr4), 0.1, places=5)


if __name__ == "__main__":
    unittest.main()
