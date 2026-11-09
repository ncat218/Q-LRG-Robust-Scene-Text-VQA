import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

from scripts.crop_fusion import (
    FUSION_LABELS, TOTAL_TOKENS, VIEW_TOKENS, fit_token_canvas, predict_fusion_image,
    select_fusion_cohort, summarize_fusion,
)


class CropFusionTest(unittest.TestCase):
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
