import unittest

import numpy as np
import torch

from qlrg.feature_injection import inject_features, replace_region


class FeatureInjectionTest(unittest.TestCase):
    def test_substitution_preserves_values(self):
        degraded = torch.arange(12).reshape(4, 3).to(torch.bfloat16)
        clean = degraded + 10
        result = replace_region(degraded, clean, np.array([[False, True], [True, False]]))
        torch.testing.assert_close(result, torch.stack((degraded[0], clean[1], clean[2], degraded[3])))

    def test_real_transformers_prefill_and_generation_with_two_grids(self):
        try:
            from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        except ModuleNotFoundError as exc:
            if exc.name != "transformers":
                raise
            self.skipTest("Requires the pinned Transformers environment")

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
                "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 4,
                "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
                "pad_token_id": 0, "eos_token_id": None,
            },
        )
        model = Qwen2_5_VLForConditionalGeneration(cfg).eval()
        grid = torch.tensor([[1, 4, 6], [1, 4, 8]])
        ids = torch.tensor([[0, 0, 61] + [62] * 6 + [60, 7], [61] + [62] * 8 + [60, 8]])
        batch = {
            "input_ids": ids, "attention_mask": ids.ne(0).long(), "image_grid_thw": grid,
            "pixel_values": torch.randn(56, 3 * 2 * 14 * 14),
        }
        with torch.inference_mode():
            base = model.model.get_image_features(batch["pixel_values"], grid)
            self.assertEqual([tensor.shape[0] for tensor in base], [6, 8])
            before = model(**batch, use_cache=False).logits
            with inject_features(model.model, grid, base):
                after = model(**batch, use_cache=False).logits
            self.assertTrue(torch.equal(before, after))
            native = model.generate(**batch, max_new_tokens=2, do_sample=False)
            with inject_features(model.model, grid, base):
                injected = model.generate(**batch, max_new_tokens=2, do_sample=False)
            self.assertTrue(torch.equal(native, injected))
            with self.assertRaisesRegex(ValueError, "shape"):
                with inject_features(model.model, grid, base[::-1]):
                    pass
            with self.assertRaisesRegex(RuntimeError, "bypassed"):
                with inject_features(model.model, grid, base):
                    pass
            self.assertNotIn("get_image_features", vars(model.model))


if __name__ == "__main__":
    unittest.main()
