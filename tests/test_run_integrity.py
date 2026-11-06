import tempfile
import unittest
from pathlib import Path

from qlrg.common import object_hash, sha256, write_json
from qlrg.experiment import validate_rows, verified_complete
from qlrg.grounding import read_review


class RunIntegrityTest(unittest.TestCase):
    def test_missing_or_partial_oracle_cohort_cannot_resume(self):
        image = {"image_id": "img", "questions": [{"question_id": "q"}]}
        rows = [{"image_id": "img", "question_id": "q", "arm": arm} for arm in ("O0", "O1", "O2")]
        validate_rows(image, rows, "s0")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_rows(image, rows + [{"image_id": "img", "question_id": "q", "arm": "O3"}], "s0")

    def test_completion_verifies_artifact_and_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding = {"some": "configuration"}
            write_json(root / "run.json", binding)
            artifacts = {}
            for name in ("predictions.jsonl", "identity.json", "baseline_summary.json", "baseline_summary.csv"):
                path = root / name
                path.write_text("artifact\n", encoding="utf-8")
                artifacts[name] = sha256(path)
            write_json(root / "COMPLETE.json", {
                "stage": "d0", "binding": object_hash(binding), "artifacts": artifacts,
            })
            self.assertEqual(verified_complete(root, "d0"), binding)
            with self.assertRaisesRegex(ValueError, "binding changed"):
                verified_complete(root, "d0", "different")
            (root / "identity.json").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verified_complete(root, "d0")

    def test_pending_grounding_never_enters_full_s0(self):
        manifest = {"images": [{
            "image_id": "i", "image_sha256": "hash", "questions": [{"question_id": "q"}],
            "words": [{"id": "w"}],
        }]}
        row = {
            "image_id": "i", "image_sha256": "hash", "question_id": "q",
            "word_ids": ["w"], "status": "pending",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.jsonl"
            write_json(path, row)
            with self.assertRaisesRegex(ValueError, "Finalize"):
                read_review(path, manifest)
            row["status"] = "accepted"
            write_json(path, row)
            self.assertEqual(read_review(path, manifest)["q"]["word_ids"], ["w"])


if __name__ == "__main__":
    unittest.main()
