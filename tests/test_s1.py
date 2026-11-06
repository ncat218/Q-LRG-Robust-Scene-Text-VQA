import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from qlrg.s1_data import training_examples
from qlrg.s1_model import LanguageLoRA, VisualResidual, teacher_forced_loss, text_training


class S1Test(unittest.TestCase):
    def test_common_training_schedule_is_deterministic(self):
        images = [{"image_id": str(i), "questions": [
            {"question_id": str(i), "answers": ["yes"] * 8 + ["no", ""]},
        ]} for i in range(10)]
        first = list(training_examples(images, 1, 42, 0.2))
        second = list(training_examples(images, 1, 42, 0.2))
        self.assertEqual(first, second)
        self.assertEqual(sum(condition == "CLEAN" for _, _, condition, _ in first), 2)
        self.assertEqual({row[0]["image_id"] for row in first}, {str(i) for i in range(10)})

    def test_teacher_forcing_gradient_through_real_qwen(self):
        try:
            from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        except ModuleNotFoundError as exc:
            if exc.name != "transformers":
                raise
            self.skipTest("Requires the pinned Transformers environment")
        from qlrg.qwen_wrapper import Qwen

        torch.manual_seed(42)
        torch.set_num_threads(1)
        cfg = Qwen2_5_VLConfig(
            image_token_id=62, vision_start_token_id=61, vision_end_token_id=60, video_token_id=59,
            vision_config={
                "depth": 1, "hidden_size": 32, "intermediate_size": 64, "out_hidden_size": 32,
                "num_heads": 4, "patch_size": 14, "temporal_patch_size": 2, "in_chans": 3,
                "spatial_merge_size": 2, "fullatt_block_indexes": [0], "window_size": 56,
            },
            text_config={
                "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
                "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
                "pad_token_id": 0, "eos_token_id": 2,
            },
        )
        qwen = Qwen.__new__(Qwen)
        qwen.model = Qwen2_5_VLForConditionalGeneration(cfg).eval().requires_grad_(False).bfloat16()
        grid = torch.tensor([[1, 4, 6]])
        ids = torch.tensor([[61] + [62] * 6 + [60, 7]])
        prompt = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "image_grid_thw": grid,
                  "pixel_values": torch.randn(24, 3 * 2 * 14 * 14).bfloat16()}
        base = qwen.encode(prompt).clone()
        self.assertFalse(base.is_inference())
        target = [9, 10, 2]

        # Check the answer-only index path against full logits and confirm base gradients stay absent.
        full = dict(prompt, input_ids=torch.cat((ids, torch.tensor([target])), dim=1),
                    attention_mask=torch.ones((1, ids.shape[1] + len(target)), dtype=torch.long))
        with torch.no_grad(), qwen.replacement(full, base):
            full_logits = qwen.model(**full, use_cache=False).logits[0].float()
        expected = torch.nn.functional.cross_entropy(
            full_logits[ids.shape[1] - 1:-1], torch.tensor(target),
        )
        torch.testing.assert_close(teacher_forced_loss(qwen, prompt, base, target), expected)

        restorer = VisualResidual(32, 8, spatial_blocks=2)
        restored, _ = restorer(base, (2, 3))
        torch.testing.assert_close(restored.bfloat16(), base)
        with text_training(qwen.model, True):
            teacher_forced_loss(qwen, prompt, restored.bfloat16(), target).backward()
        self.assertGreater(restorer.output.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in qwen.model.parameters()))
        self.assertFalse(qwen.model.is_gradient_checkpointing)

        lora = LanguageLoRA(qwen.model.model.language_model, rank=2, alpha=4)
        projection = qwen.model.model.language_model.layers[0].self_attn.q_proj
        with lora.attached(), text_training(qwen.model, True):
            loss = teacher_forced_loss(qwen, prompt, base, target)
            torch.testing.assert_close(loss, expected)
            loss.backward()
        self.assertTrue(any(delta.up.weight.grad.abs().sum().item() > 0 for delta in lora.deltas))
        self.assertIs(qwen.model.model.language_model.layers[0].self_attn.q_proj, projection)
        self.assertTrue(all(p.grad is None for p in qwen.model.parameters()))

    def test_report_keeps_conditions_and_numeric_slice_separate(self):
        from qlrg.common import read_json
        from qlrg.s1_report import CONDITIONS, LABELS, write_report, write_visuals
        from PIL import Image

        questions = [
            {"question_id": "1", "question": "Number?", "answers": ["12"] * 10},
            {"question_id": "2", "question": "Color?", "answers": ["red"] * 10},
        ]
        manifest = {"images": [{"image_id": "img", "questions": questions}]}
        rows = [
            {"condition": condition, "arm": arm, "question_id": q["question_id"], "image_id": "img",
             "answer": "blue" if arm == "S1b" else q["answers"][0],
             "score": 0.0 if arm == "S1b" else 1.0, "generation_seconds": 0.1,
             "adapter_seconds_per_image": 0.01, "feature_cosine_to_clean": 0.9,
             "feature_mse_to_clean": 0.5 if arm == "S1b" else 1.0}
            for condition in CONDITIONS for arm in LABELS for q in questions
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            image = Image.new("RGB", (32, 16))
            write_visuals(path / "variants" / "S1b", "LR4", "img", image, torch.ones(2, 4), (1, 2))
            write_report(path, manifest, rows, {}, [], {"seed": 42, "bootstrap_resamples": 100})
            summary = read_json(path / "summary.json")
            self.assertEqual(len(summary["accuracies"]), 10)
            numeric = [r for r in summary["comparisons"] if r["slice"] == "numeric_or_date"]
            self.assertTrue(numeric)
            self.assertTrue(all(r["questions"] == 1 for r in numeric))
            self.assertEqual(len(read_json(path / "feature_regressions.json")), 4)
            lr4 = (path / "review.html").read_text(encoding="utf-8")
            clean = (path / "review-clean.html").read_text(encoding="utf-8")
            self.assertIn('"img-S1b":"iVBOR', lr4)
            self.assertNotIn('"img-S1b":"iVBOR', clean)

    def test_resume_from_atomic_checkpoint_reproduces_adapter(self):
        from qlrg.s1_run import train

        config = {"learning_rate": 0.01, "weight_decay": 0.0, "accumulate": 2,
                  "max_grad_norm": 1.0, "checkpoint_every": 1}
        qwen = SimpleNamespace(device="cpu", model=None)
        cache = SimpleNamespace(load=lambda image: None)
        examples = [(i, i, "LR4", "") for i in range(5)]

        def loss(_qwen, _arm, adapter, _cache, question, _condition, _answer, _config):
            return (adapter(torch.ones(1, 1)) - question).square().mean()

        def interrupt(*args):
            if args[4] == 2:
                raise RuntimeError("simulated job interruption")
            return loss(*args)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            uninterrupted = torch.nn.Linear(1, 1)
            resumed = torch.nn.Linear(1, 1)
            resumed.load_state_dict(uninterrupted.state_dict())
            with patch("qlrg.s1_run.loss_for", side_effect=loss):
                train(qwen, "S1a", uninterrupted, cache, examples, config, Path(temp) / "full.pt", "fp")
            with patch("qlrg.s1_run.loss_for", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "interruption"):
                train(qwen, "S1a", resumed, cache, examples, config, path, "fp")
            with patch("qlrg.s1_run.loss_for", side_effect=loss):
                history = train(qwen, "S1a", resumed, cache, examples, config, path, "fp")
            for actual, expected in zip(resumed.parameters(), uninterrupted.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual([h["examples"] for h in history], [2, 4, 5])


if __name__ == "__main__":
    unittest.main()
