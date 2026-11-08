import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from qlrg.case_review import write_case_review
from qlrg.common import object_hash, sha256, write_json, write_jsonl
from qlrg.s1_model import LanguageLoRA, LoRALinear
from scripts.crop_replay import read_crops
from scripts.query_crop import LABELS, candidate_boxes, checkpoints_for, donor_questions, predict_image, summarize
from scripts.sdrpn_runtime import selector_environment


class QueryCropTest(unittest.TestCase):
    def test_finalized_s1_arm_is_usable_before_other_arms_finish(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            s1 = root / "s1"
            arm = s1 / "variants" / "S1b"
            arm.mkdir(parents=True)
            plan = [{"questions": [{}, {}]}]
            shared = {"model": "qwen", "source_hash": "training"}
            hashes = {"train_inner": "split"}
            binding = {"stage": "s1", "shared": shared, "dataset_hashes": hashes,
                       "config": {"epochs": 2, "accumulate": 2}, "train_manifest": object_hash(plan)}
            history = [{"examples": 2}, {"examples": 4}]
            checkpoint = {"arm": "S1b", "binding": object_hash(binding), "examples": 4,
                          "history": history, "adapter": {"weight": torch.ones(1)}}
            path = arm / "checkpoint.pt"
            write_json(s1 / "run.json", binding)
            write_json(s1 / "train_manifest.json", plan)
            write_json(arm / "training.json", {"examples": 4, "steps": 2, "history": history})
            torch.save(checkpoint, path)
            settings = {"s1_run": str(root), "arm": "S1b", "split": "val_inner", "from_crop_run": "source"}
            loaded = checkpoints_for(settings, {**shared, "source_hash": "crop"}, hashes)
            self.assertEqual(loaded["S1b"]["provenance"]["sha256"], sha256(path))
            a1 = s1 / "variants" / "A1"
            write_json(a1 / "training.json", {"examples": 4, "steps": 2, "history": history})
            torch.save({**checkpoint, "arm": "A1"}, a1 / "checkpoint.pt")
            settings = {**settings, "arm": None, "arms": ["S1b", "A1"]}
            self.assertEqual(set(checkpoints_for(settings, shared, hashes)), {"S1b", "A1"})
            torch.save({**checkpoint, "examples": 2}, path)
            with self.assertRaisesRegex(ValueError, "S1b does not have a verified final checkpoint"):
                checkpoints_for(settings, shared, hashes)

    def test_replay_restores_all_tokens_on_the_archived_crop_and_full_image(self):
        lr4 = Image.new("RGB", (80, 40), "white")
        image = {"image_id": "one", "size": [80, 40], "questions": [
            {"question_id": 1, "question": "Read it.", "answers": ["open"] * 10},
            {"question_id": 2, "question": "Read the sign.", "answers": ["open"] * 10},
        ]}
        settings = {"selector": "sdrpn", "split": "val_inner", "zoom": 1.5, "seed": 42, "images": None,
                    "from_crop_run": "source"}
        hashes, shared = {"val_inner": "fixed"}, {"model": "qwen"}
        source_binding = {"settings": settings, "dataset_hashes": hashes, "shared": shared, "checkpoint": None,
                          "selected_images": ["one"], "selector": {"id": "SD-RPN"}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            write_json(source / "run.json", source_binding)
            records = [
                {"image_id": "one", "question_id": 1, "arm": "full", "preview": "review-images/full.jpg"},
                {"image_id": "one", "question_id": 1, "arm": "query", "preview": "review-images/query.jpg",
                 "crop_xyxy": [0, 0, 40, 20], "roi_selected": True, "foreground_tokens": 4,
                 "input_size": [60, 30], "grid_thw": [1, 2, 4]},
                {"image_id": "one", "question_id": 2, "arm": "full", "preview": "review-images/full.jpg"},
                {"image_id": "one", "question_id": 2, "arm": "query", "preview": "review-images/full.jpg",
                 "crop_xyxy": [0, 0, 80, 40], "roi_selected": False, "foreground_tokens": 0,
                 "input_size": [80, 40], "grid_thw": [1, 2, 4]},
            ]
            write_jsonl(source / "predictions.jsonl", records)
            write_json(source / "COMPLETE.json", {"stage": "query_crop", "binding": object_hash(source_binding),
                       "artifacts": {"predictions.jsonl": sha256(source / "predictions.jsonl")}})
            crops, _, _ = read_crops(source, output, settings, shared, hashes, [image])
            model = Mock(dtype=torch.float32)
            model.prepare.return_value = {"image_grid_thw": torch.tensor([[1, 2, 4]])}
            model.encode.return_value = torch.zeros((2, 3))
            query_proj = torch.nn.Linear(3, 3).requires_grad_(False)
            value_proj = torch.nn.Linear(3, 3).requires_grad_(False)
            attention = SimpleNamespace(q_proj=query_proj, v_proj=value_proj)
            language = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])

            def generate(batch, features):
                token = int(features[0, 0])
                if isinstance(attention.q_proj, LoRALinear):
                    self.assertIsInstance(attention.v_proj, LoRALinear)
                    self.assertIs(features, model.encode.return_value)
                    token = 4
                else:
                    self.assertIs(attention.q_proj, query_proj)
                    self.assertIs(attention.v_proj, value_proj)
                return {"answer": "open" if token == 3 else "closed", "generation_seconds": 0.1, "token_ids": [token]}

            model.generate.side_effect = generate
            adapters = {
                "A2": LanguageLoRA(language, rank=2, alpha=2),
                "S1a": lambda features, grid: (features + 1, torch.ones_like(features)),
                "S1b": lambda features, grid: (features + 2, torch.ones_like(features)),
                "A1": lambda features, grid: (features + 3, torch.ones_like(features)),
            }
            scorer = Mock()
            scorer._compute_answer_scores.return_value = {"open": 1.0}
            scorer.answer_processor.side_effect = lambda text: text
            with patch("scripts.query_crop.paired_images", return_value=(None, lr4)), patch("torch.cuda.synchronize"):
                result = predict_image(model, None, adapters, scorer, image, {}, root, output,
                                       settings, proposals=crops["one"])
            self.assertFalse((output / "review-images").exists())
            rows = {(row["question_id"], row["arm"]): row for row in result["rows"]}
            labels = {arm: arm for arm in ("full", "query")}
            for variant, token in (("S1a", 1), ("S1b", 2), ("A1", 3), ("A2", 4)):
                full, query = f"full_{variant}", f"query_{variant}"
                labels.update({full: full, query: query})
                self.assertEqual(rows[1, query]["crop_xyxy"], (0, 0, 40, 20))
                self.assertEqual(rows[1, query]["visual_tokens"], rows[1, "query"]["visual_tokens"])
                self.assertEqual(rows[1, query]["token_ids"], [token])
                self.assertEqual(rows[2, query]["token_ids"], rows[2, full]["token_ids"])
            report = summarize(result["rows"], labels,
                               {"split": "val_inner", "images": [image]}, result["selector_seconds"], 42)
            self.assertEqual(report["roi_only"]["questions"], 1)
            self.assertEqual(report["roi_only"]["accuracy_pct"]["query_A1"], 100)
            self.assertIn(("query_S1a", "query_A1"),
                          {(x["reference"], x["target"]) for x in report["roi_only"]["comparisons"]})
            self.assertIn(("query_A1", "query_A2"),
                          {(x["reference"], x["target"]) for x in report["roi_only"]["comparisons"]})
            self.assertIs(attention.q_proj, query_proj)
            self.assertIs(attention.v_proj, value_proj)
            self.assertEqual(model.encode.call_count, 3)
            self.assertEqual(model.generate.call_count, 15)

    def test_expat_preload_is_scoped_to_selector_child(self):
        with patch.dict(os.environ, {"LD_PRELOAD": "/old.so", "PYTHONPATH": "/wrong"}):
            env = selector_environment({"sdrpn_libexpat": "/conda/lib/libexpat.so.1"})
            self.assertEqual(env["LD_PRELOAD"], "/conda/lib/libexpat.so.1:/old.so")
            self.assertNotIn("PYTHONPATH", env)
            self.assertEqual(os.environ["LD_PRELOAD"], "/old.so")

    def test_sdrpn_box_and_missing_region_use_the_same_lr4(self):
        lr4 = Image.new("RGB", (80, 40), "white")
        image = {"image_id": "one", "questions": [
            {"question_id": 1, "question": "Read the sign.", "answers": ["open"] * 10},
        ]}
        proposals = {
            "image_id": "one", "lr4_sha256": hashlib.sha256(lr4.tobytes()).hexdigest(), "selector_seconds": 0.5,
            "questions": {"1": {"query": {"box": [5, 5, 15, 10], "foreground_tokens": 1},
                                "shuffled": {"box": None, "foreground_tokens": 0}}},
        }
        model = Mock()
        model.prepare.return_value = {"image_grid_thw": torch.tensor([[1, 2, 4]])}
        model.encode.return_value = torch.zeros((2, 3))
        model.generate.return_value = {"answer": "open", "generation_seconds": 0.1, "token_ids": [1]}
        scorer = Mock()
        scorer._compute_answer_scores.return_value = {"open": 1.0}
        scorer.answer_processor.side_effect = lambda text: text
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with patch("scripts.query_crop.paired_images", return_value=(None, lr4)), patch("torch.cuda.synchronize"):
                result = predict_image(model, None, {}, scorer, image, {1: (2, "Other?")}, out, out,
                                       {"zoom": 1.5}, proposals=proposals)
                with self.assertRaisesRegex(ValueError, "same LR4 pixels"):
                    predict_image(model, None, {}, scorer, image, {}, out, out, {"zoom": 1.5},
                                  proposals={**proposals, "lr4_sha256": "wrong"})
        rows = {row["arm"]: row for row in result["rows"]}
        self.assertEqual(result["selector_seconds"], 0.5)
        self.assertEqual(rows["query"]["crop_xyxy"], (5, 5, 15, 10))
        self.assertEqual(rows["query"]["input_size"], (56, 28))
        self.assertEqual(rows["shuffled"]["input_size"], (80, 40))
        self.assertFalse(rows["shuffled"]["roi_selected"])
        self.assertEqual(rows["shuffled"]["answer"], rows["full"]["answer"])
        self.assertEqual(model.generate.call_count, 2)

    def test_rank_uses_only_lr4_and_questions_and_reuses_crop_for_restoration(self):
        lr4 = Image.new("RGB", (80, 40), "white")
        clean = Image.new("RGB", lr4.size, "green")
        image = {"image_id": "one", "questions": [
            {"question_id": 1, "question": "What is the highest number?", "answers": ["71"] * 10},
        ]}
        other = {"image_id": "two", "questions": [
            {"question_id": 2, "question": "What is the highest number? ", "answers": ["100"] * 10},
            {"question_id": 3, "question": "What brand is this?", "answers": ["brand"] * 10},
        ]}
        donors = donor_questions([image, other], [image], seed=42)
        self.assertEqual(donors, {1: (3, "What brand is this?")})
        boxes = candidate_boxes(lr4.size, 0.6)
        self.assertEqual(len(boxes), 16)
        selector = Mock()
        selector.rank.return_value = [list(range(16)), list(reversed(range(16)))]
        model = Mock(dtype=torch.float32)
        model.prepare.return_value = {"image_grid_thw": torch.tensor([[1, 2, 4]])}
        model.encode.return_value = torch.zeros((2, 3))
        model.generate.side_effect = lambda batch, feature: {
            "answer": "71" if feature.any() else "70", "generation_seconds": 0.1, "token_ids": [1],
        }
        adapter = Mock(side_effect=lambda base, grid: (base + 1, torch.ones_like(base)))
        scorer = Mock()
        scorer._compute_answer_scores.return_value = {"71": 1.0}
        scorer.answer_processor.side_effect = lambda text: text

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch("scripts.query_crop.paired_images", return_value=(clean, lr4)), patch("torch.cuda.synchronize"):
                result = predict_image(model, selector, {"restore": adapter}, scorer, image, donors, output, output,
                                       {"crop_fraction": 0.6, "zoom": 1.5})
            selector.rank.assert_called_once_with(lr4, boxes, ["What is the highest number?", "What brand is this?"])
            self.assertEqual(model.prepare.call_count, 3)
            self.assertTrue(all(c.args[1] == "What is the highest number?" for c in model.prepare.call_args_list))
            rows = result["rows"]
            by_arm = {r["arm"]: r for r in rows}
            self.assertEqual(by_arm["query"]["crop_xyxy"], boxes[-1])
            self.assertEqual(by_arm["shuffled"]["crop_xyxy"], boxes[0])
            self.assertEqual(by_arm["query_restore"]["crop_xyxy"], by_arm["query"]["crop_xyxy"])
            self.assertEqual(by_arm["query_restore"]["preview"], by_arm["query"]["preview"])
            self.assertEqual(by_arm["query_restore"]["score"], 1.0)
            manifest = {"split": "val_inner", "images": [image]}
            summary = summarize(rows, LABELS, manifest, result["selector_seconds"], seed=42)
            self.assertEqual(summary["restoration_gain_crop_minus_full"]["delta_pp"], 0)
            previews = {f"{r['question_id']}-{r['arm']}": r["preview"] for r in rows}
            self.assertTrue(all((output / path).is_file() for path in previews.values()))
            report = output / "review.html"
            write_case_review(report, rows, LABELS, title="Query crop", manifest=manifest, crop_previews=previews)
            self.assertIn(json.dumps(previews, separators=(",", ":")), report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
