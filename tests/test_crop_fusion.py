import tempfile
import unittest
from hashlib import sha256
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

from scripts.crop_fusion import (
    FALLBACK_ARM, FUSION_ALL_LABELS, FUSION_LABELS, TOTAL_TOKENS, VIEW_TOKENS, fit_token_canvas, predict_fusion_image,
    select_fusion_cohort, summarize_fusion,
)
from scripts.fusion_confirm import (
    CONFIRM_LABELS, predict_confirmation_image, random_crop_box, summarize_confirmation,
)


class CropFusionTest(unittest.TestCase):
    def test_official_a2_confirmation_keeps_budget_and_crop_size(self):
        lr4 = Image.new("RGB", (640, 480), "white")
        q = {"question_id": 1, "question": "What is written?", "answers": ["open"] * 10}
        image = {"image_id": "im", "questions": [q, {**q, "question_id": 2}]}
        proposals = {
            "image_id": "im", "lr4_sha256": sha256(lr4.tobytes()).hexdigest(), "selector_seconds": 0.05,
            "questions": {"1": {"query": {"box": (20, 100, 240, 160), "selector_seconds": 0.03}},
                          "2": {"query": {"box": None, "selector_seconds": 0.02}}},
        }
        self.assertEqual(random_crop_box((0, 0, 640, 480), lr4.size, 42, "im", 1), (0, 0, 640, 480))
        model = Mock()
        model.prepare_views.side_effect = lambda views, _: (
            {"input_ids": torch.zeros((1, TOTAL_TOKENS + len(views)))},
            [(view.height // 28, view.width // 28) for view in views],
        )
        model.prefill_logits.return_value = torch.ones(1)
        model.generate.return_value = {"answer": "open", "token_ids": [1], "generation_seconds": 0.1}
        lora = Mock()
        lora.attached.side_effect = nullcontext
        scorer = Mock()
        scorer._compute_answer_scores.return_value = {"open": 1.0}
        scorer.answer_processor.side_effect = lambda text: text
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch("scripts.fusion_confirm.paired_images", return_value=(None, lr4)), patch("torch.cuda.synchronize"):
                result = predict_confirmation_image(model, lora, scorer, image, proposals, output, output, 42, True)
            rows = {(r["question_id"], r["arm"]): r for r in result["rows"]}
            query, random = (rows[1, arm]["crop_xyxy"] for arm in ("fusion_A2", "random_fusion_A2"))
            self.assertNotEqual(query, random)
            self.assertEqual((query[2] - query[0], query[3] - query[1]),
                             (random[2] - random[0], random[3] - random[1]))
            self.assertEqual({r["visual_tokens"] for r in rows.values()}, {576})
            self.assertEqual(rows[1, "fusion_A2"]["view_tokens"], [288, 288])
            self.assertTrue(all(rows[2, arm]["view_tokens"] == [576] for arm in CONFIRM_LABELS))
            self.assertEqual(lora.attached.call_count, 5)
            self.assertEqual(model.generate.call_count, 4)
            report = summarize_confirmation(result["rows"], {"images": [image], "split": "official_val"},
                                              42, result["identity"], result["selector_seconds"])
            self.assertEqual(report["questions"], 2)
            self.assertEqual(report["fallback_questions"], 1)
            self.assertEqual(report["random_moved_only"]["questions"], 1)
            self.assertEqual(report["arms"]["full_A2"]["mean_selector_seconds"], 0)

    def test_canvas_matches_budget_and_preserves_aspect_ratio(self):
        for size in ((1024, 768), (768, 1024), (420, 50), (41, 300)):
            for budget in (VIEW_TOKENS, TOTAL_TOKENS):
                canvas, occupancy = fit_token_canvas(Image.new("RGB", size, "white"), budget)
                self.assertEqual(canvas.width % 28, 0)
                self.assertEqual(canvas.height % 28, 0)
                self.assertEqual(canvas.width * canvas.height // 28**2, budget)
                self.assertGreater(occupancy, 0.7)
                self.assertLessEqual(occupancy, 1.0)

    def test_restoration_masks_keep_same_grids_and_exact_budget(self):
        lr4 = Image.new("RGB", (1024, 768), "white")
        question = {"question_id": 1, "question": "What is written?", "answers": ["open"] * 10}
        image = {"image_id": "im", "questions": [question, {**question, "question_id": 2}]}
        proposals = {"questions": {"1": {"query": {"box": (20, 100, 240, 160)}},
                                  "2": {"query": {"box": None}}}}
        images = select_fusion_cohort([image], {"im": proposals})
        self.assertEqual(images[0]["questions"], [question])
        model = Mock(dtype=torch.float32)

        def prepare(views, text):
            grids = [(view.height // 28, view.width // 28) for view in views]
            return {"input_ids": torch.zeros((1, TOTAL_TOKENS + len(views)))}, grids

        model.prepare_views.side_effect = prepare
        model.encode_views.side_effect = [
            (torch.full((576, 3), 1.0),), (torch.full((576, 3), 2.0),),
            (torch.full((288, 3), 1.0), torch.full((288, 3), 2.0)),
        ]
        model.prefill_logits.return_value = torch.ones(1)
        model.generate.side_effect = lambda _, features: {
            "answer": "open", "token_ids": [int(f[0, 0]) for f in features], "generation_seconds": 0.1,
        }
        adapter = Mock(side_effect=lambda feature, grid: (feature + 10, feature))
        scorer = Mock()
        scorer._compute_answer_scores.return_value = {"open": 1.0}
        scorer.answer_processor.side_effect = lambda text: text
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch("scripts.crop_fusion.paired_images", return_value=(None, lr4)), patch("torch.cuda.synchronize"):
                result = predict_fusion_image(model, adapter, scorer, images[0], proposals, output, output, True)
            rows = {row["arm"]: row for row in result["rows"]}
            self.assertEqual(rows.keys(), FUSION_LABELS.keys())
            self.assertEqual(rows["full"]["view_tokens"], {"full": 576})
            self.assertEqual(rows["crop"]["view_tokens"], {"crop": 576})
            for arm, expected in (("fusion", [1, 2]), ("fusion_full_S1b", [11, 2]),
                                  ("fusion_crop_S1b", [1, 12]), ("fusion_both_S1b", [11, 12])):
                self.assertEqual(rows[arm]["token_ids"], expected)
                self.assertEqual(rows[arm]["view_tokens"], {"full": 288, "crop": 288})
            self.assertEqual({row["visual_tokens"] for row in rows.values()}, {576})
            self.assertTrue(all((output / row["preview"]).is_file() for row in rows.values()))
            self.assertEqual(model.encode_views.call_count, 3)
            self.assertEqual(model.generate.call_count, 8)
            report = summarize_fusion(result["rows"], {"images": images, "split": "val_inner"}, 42,
                                      result["identity"], 1)
            self.assertEqual(report["excluded_no_roi_questions"], 1)
            self.assertTrue(report["identity"]["logits_exact"])
            self.assertEqual(report["visual_budget"]["total"], 576)

            model.reset_mock()
            model.encode_views.side_effect = [
                (torch.full((576, 3), 1.0),), (torch.full((576, 3), 2.0),),
                (torch.full((288, 3), 1.0), torch.full((288, 3), 2.0)), (torch.full((576, 3), 1.0),),
            ]
            lora = Mock()
            lora.attached.side_effect = nullcontext
            with patch("scripts.crop_fusion.paired_images", return_value=(None, lr4)), patch("torch.cuda.synchronize"):
                result = predict_fusion_image(model, adapter, scorer, image, proposals, output, output, True, lora)
            fallback = {row["arm"]: row for row in result["rows"] if row["question_id"] == 2}
            self.assertEqual(fallback.keys(), FUSION_ALL_LABELS.keys())
            for arm, source in FALLBACK_ARM.items():
                self.assertEqual(fallback[arm]["token_ids"], fallback[source]["token_ids"])
                self.assertEqual(fallback[arm]["visual_tokens"], 576)
                self.assertEqual(fallback[arm]["fallback_reason"], "no_roi")
            self.assertEqual(lora.attached.call_count, 2)
            self.assertEqual(model.encode_views.call_count, 4)
            self.assertEqual(model.generate.call_count, 12)
            report = summarize_fusion(result["rows"], {"images": [image], "split": "val_inner"}, 42,
                                      result["identity"], 0, include_a2=True)
            self.assertEqual(report["questions"], 2)
            self.assertEqual(report["fallback_questions"], 1)
            self.assertEqual(report["roi_only"]["questions"], 1)
            self.assertIn(("full_A2", "fusion"),
                          {(r["reference"], r["target"]) for r in report["comparisons"]})
