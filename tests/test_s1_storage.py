import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from qlrg.common import object_hash, read_json
from qlrg.oracle import feature_pair
from qlrg.s1_storage import checkpoint_steps, resume_runtime


class S1StorageTest(unittest.TestCase):
    def test_five_writes_including_final_and_legacy_override(self):
        config = {"accumulate": 4, "checkpoint_every": 250}
        with patch.dict(os.environ, {"QLRG_CHECKPOINT_SAVES": "5"}):
            self.assertEqual(checkpoint_steps(config, 62258), {3113, 6226, 9339, 12452, 15565})

    def test_legacy_resume_preserves_checkpoint_binding(self):
        transition = read_json(Path(__file__).resolve().parents[1] / "configs/s1_storage_compat.json")
        shared = {"source_hash": transition["after_source"], "a0_path_hash": "new",
                  "model": "same", "device": "same"}
        legacy = {**shared, "source_hash": transition["before_source"],
                  "a0_path_hash": transition["before_a0_path"]}
        saved = {"shared": legacy, "config": {"checkpoint_every": 250}}
        restored = {**saved, "shared": resume_runtime(legacy, shared)}
        self.assertEqual(object_hash(restored), object_hash(saved))
        with self.assertRaises(ValueError):
            resume_runtime(legacy, {**shared, "device": "different"})

    def test_read_only_cache_hit_and_miss_do_not_write(self):
        batch = {"image_grid_thw": torch.tensor([[1, 2, 2]])}
        model = SimpleNamespace(
            encode=Mock(side_effect=lambda _: torch.ones(1, 4)), device="cpu", dtype=torch.float32,
            model=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))),
        )
        shared = {"model": {}, "source_hash": "s1", "packages": {}, "torch_build": "test"}
        with tempfile.TemporaryDirectory() as folder, patch("torch.cuda.get_device_name", return_value="test"):
            cache = Path(folder)
            with patch.dict(os.environ, {"QLRG_FEATURE_CACHE": folder, "QLRG_FEATURE_CACHE_MODE": "write"}):
                cached = feature_pair(model, {"image_sha256": "cached"}, batch, batch, cache, shared)
            before = sorted(p.name for p in cache.iterdir())
            model.encode.reset_mock()
            with patch.dict(os.environ, {"QLRG_FEATURE_CACHE": folder, "QLRG_FEATURE_CACHE_MODE": "read-only"}):
                hit = feature_pair(model, {"image_sha256": "cached"}, batch, batch, cache, shared)
                model.encode.assert_not_called()
                miss = feature_pair(model, {"image_sha256": "missing"}, batch, batch, cache, shared)
                self.assertEqual(model.encode.call_count, 2)
            self.assertEqual(before, sorted(p.name for p in cache.iterdir()))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(cached, hit)))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(cached, miss)))


if __name__ == "__main__":
    unittest.main()
