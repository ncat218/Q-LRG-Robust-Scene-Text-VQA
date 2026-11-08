import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

from qlrg.case_review import write_case_review
from qlrg.common import object_hash, sha256, write_json, write_jsonl
from scripts.crop_replay import read_crops
from scripts.query_crop import LABELS, candidate_boxes, checkpoint_for, donor_questions, predict_image, summarize
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
            loaded = checkpoint_for(settings, {**shared, "source_hash": "crop"}, hashes)
            self.assertEqual(loaded["provenance"]["sha256"], sha256(path))
            torch.save({**checkpoint, "examples": 2}, path)
            with self.assertRaisesRegex(ValueError, "verified final checkpoint"):
                checkpoint_for(settings, shared, hashes)

    def test_replay_restores_all_tokens_on_the_archived_crop_and_full_image(self):
        lr4 = Image.new("RGB", (80, 40), "white")
        image = {"image_id": "one", "size": [80, 40], "questions": [
            {"question_id": 1, "question": "Read it.", "answers": ["open"] * 10},
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
            ]
            write_jsonl(source / "predictions.jsonl", records)
            write_json(source / "COMPLETE.json", {"stage": "query_crop", "binding": object_hash(source_binding),
                       "artifacts": {"predictions.jsonl": sha256(source / "predictions.jsonl")}})
            crops, _, _ = read_crops(source, output, settings, shared, hashes, [image])
            model = Mock(dtype=torch.float32)
            model.prepare.return_value = {"image_grid_thw": torch.tensor([[1, 2, 4]])}
            model.encode.return_value = torch.zeros((2, 3))
            model.generate.return_value = {"answer": "open", "generation_seconds": 0.1, "token_ids": [1]}
            restorer = Mock(side_effect=lambda features, grid: (features + 1, torch.ones_like(features)))
            scorer = Mock()
            scorer._compute_answer_scores.return_value = {"open": 1.0}
            scorer.answer_processor.side_effect = lambda text: text
            with patch("scripts.query_crop.paired_images", return_value=(None, lr4)), patch("torch.cuda.synchronize"):
                result = predict_image(model, None, restorer, scorer, image, {}, root, output,
                                       settings, proposals=crops["one"])
            self.assertFalse((output / "review-images").exists())
            rows = {row["arm"]: row for row in result["rows"]}
            self.assertEqual(set(rows), {"full", "query", "full_restore", "query_restore"})
            self.assertEqual(rows["query_restore"]["crop_xyxy"], (0, 0, 40, 20))
            self.assertEqual(rows["query_restore"]["visual_tokens"], rows["query"]["visual_tokens"])
            report = summarize(result["rows"], {arm: LABELS[arm] for arm in rows},
                               {"split": "val_inner", "images": [image]}, result["selector_seconds"], 42)
            self.assertEqual(report["roi_only"]["questions"], 1)
            self.assertEqual(model.generate.call_count, 4)

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
                result = predict_image(model, None, None, scorer, image, {1: (2, "Other?")}, out, out,
                                       {"zoom": 1.5}, proposals=proposals)
                with self.assertRaisesRegex(ValueError, "same LR4 pixels"):
                    predict_image(model, None, None, scorer, image, {}, out, out, {"zoom": 1.5},
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
                result = predict_image(model, selector, adapter, scorer, image, donors, output, output,
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
