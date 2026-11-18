import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from qlrg.common import read_json
from scripts.final_evaluate import visual_features
from scripts.final_review import LABELS, demo, index_rows, report
from scripts.pre_11305 import PreAdapter
import run_final_study


class FinalStudyTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {"split": "val_inner", "images": [
            {"image_id": "a", "questions": [{"question_id": "1", "question": "Read?", "answers": ["71"] * 10}]},
            {"image_id": "b", "questions": [{"question_id": "2", "question": "Read?", "answers": ["name"] * 10}]},
        ]}
        self.rows = []
        for condition in ("CLEAN", "LR4"):
            for iid, qid in (("a", "1"), ("b", "2")):
                for arm in LABELS:
                    score = 1.0 if arm == "A0" or iid == "b" else 0.0
                    self.rows.append({"condition": condition, "image_id": iid, "question_id": qid,
                                      "arm": arm, "answer": "71" if score else "71 extra", "score": score})

    def test_all_arms_must_cover_the_same_questions(self):
        index_rows(list(reversed(self.rows)), self.manifest)
        with self.assertRaisesRegex(ValueError, "same manifest"):
            index_rows(self.rows[:-1], self.manifest)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            index_rows([*self.rows, self.rows[0]], self.manifest)
        invalid = copy.deepcopy(self.rows)
        invalid[0]["score"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid"):
            index_rows(invalid, self.manifest)

    def test_report_tracks_regressions_without_claiming_error_causes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = report(output, self.rows, self.manifest)
            contrast = next(row for row in result["comparisons"] if
                            row["condition"] == "LR4" and row["reference"] == "A0" and row["target"] == "Pre")
            self.assertEqual(contrast["delta_pp"], -50.0)
            self.assertEqual(contrast["fully_correct_to_lower"], 1)
            self.assertEqual(contrast["score_worsened"], 1)
            self.assertIn("unreviewed", (output / "error_cases.csv").read_text(encoding="utf-8"))
            with patch("scripts.final_review.paired_images", return_value=(Image.new("RGB", (16, 16)),
                                                                            Image.new("RGB", (16, 16)))):
                demo(output, self.rows, self.manifest, output)
            self.assertTrue(read_json(output / "demo_cases.json"))
            self.assertIn("bản phát lại", (output / "demo.html").read_text(encoding="utf-8"))

    def test_profile_pre_path_matches_evaluation_patch_order(self):
        class Merger(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 6)

            def forward(self, raw):
                return self.linear(raw.reshape(-1, 16))

        window = torch.tensor([1, 0, 3, 2])
        visual = SimpleNamespace(merger=Merger(), spatial_merge_size=2,
                                 get_window_index=lambda _: (window, []))
        raw = torch.arange(64).reshape(16, 4).float() / 64

        def encode(_):
            return visual.merger(raw)[window.argsort()]

        model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32,
                                model=SimpleNamespace(model=SimpleNamespace(visual=visual)), encode=encode)
        order = torch.arange(16).reshape(2, 2, 2, 2).permute(0, 2, 1, 3).reshape(4, 4)[window].flatten()
        base = encode(None)
        cache = SimpleNamespace(qwen=model, raw={"LR4": raw}, features={"CLEAN": base.clone(), "LR4": base},
                                order=order, inverse=order.argsort(), reverse_windows=window.argsort(), pre_grid=(4, 4))
        adapter = PreAdapter(cache, 4, 3, 0)
        with torch.no_grad():
            adapter.output.bias.fill_(0.2)
            expected = adapter(base, (2, 2))[0]
            actual = visual_features(model, {"image_grid_thw": torch.tensor([[1, 4, 4]])}, adapter, "Pre")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_resuming_a_prepared_eval_keeps_its_training_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            resources = {"account": "uit", "partition": "normal", "qos": "mig2-35", "gres": "gpu:2g.35gb:1"}
            (root / "configs/slurm_hpc.json").write_text(json.dumps(resources))
            output = root / "evaluation"
            output.mkdir()
            launch = {"pre_run": str(root / "pre"), "env": str(root / "env"),
                      "paths": str(root / "paths.yaml"), "training_dependency": "48633"}
            (output / "launch.json").write_text(json.dumps(launch))
            stdout = io.StringIO()
            with patch.object(run_final_study, "ROOT", root), patch("sys.stdout", stdout), \
                    patch("sys.argv", ["run_final_study.py", "--resume", str(output)]), \
                    patch.dict("os.environ", {}, clear=True), patch("run_final_study.subprocess.run"):
                run_final_study.main()
            self.assertIn("--dependency=afterok:48633", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
