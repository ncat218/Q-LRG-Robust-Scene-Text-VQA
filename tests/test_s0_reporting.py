import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qlrg.evaluation import oracle_decision_memo, summarize


class S0ReportingTest(unittest.TestCase):
    def test_clean_gap_alone_does_not_support_localized_substitution(self):
        def comparison(cohort, ref, target, delta, low, high):
            return {"cohort": cohort, "slice": "all", "reference": ref, "target": target,
                    "questions": 2, "images": 2, "delta_pp": delta, "ci_low_pp": low, "ci_high_pp": high}

        contrasts = [
            comparison("all", "O0", "O1", 18, 15, 21),
            comparison("all", "O0", "O2", 0, -1, 1),
            comparison("grounded", "O0", "O3", 5, 2, 8),
            comparison("grounded", "O4", "O3", 0, -1, 1),
        ]
        memo = oracle_decision_memo("full", 2, 2, "provisional_ocr_150", contrasts)
        self.assertIn("O2 has no clear positive effect", memo)
        self.assertIn("O3 does not show a clear positive effect over both", memo)
        self.assertIn("Answer-region evidence is provisional", memo)
        self.assertIn("does not drive the localized-intervention decision", memo)

    def test_small_text_slice_uses_grounded_answer_and_context_requires_review(self):
        class Scorer:
            def _compute_answer_scores(self, answers):
                return {"42": 1.0}

            def answer_processor(self, answer):
                return answer

        manifest = {"images": [{"questions": [{"question_id": qid, "answers": ["42"] * 10}]} for qid in ("1", "2", "3")]}
        rows = [
            {"image_id": qid, "question_id": qid, "arm": arm, "answer": "42",
             "small_ocr_text": True, "small_answer_text": qid == "1", "context_dependent": qid == "2"}
            for qid in ("1", "2", "3")
            for arm in (("O0", "O1", "O2", "O3", "O4", "O5") if qid != "3" else ("O0", "O1", "O2"))
        ]
        with tempfile.TemporaryDirectory() as directory, patch("qlrg.evaluation.evaluator", return_value=Scorer()):
            for policy in ("provisional_ocr_150", "human_reviewed"):
                comparisons = summarize(Path(directory), rows, manifest, "s0", 10, 42, grounding_policy=policy)
                small = [r for r in comparisons if r["slice"] == "small_answer_text"]
                self.assertTrue(small)
                self.assertTrue(all(r["cohort"] == "grounded" and r["questions"] == 1 for r in small))
                context = [r for r in comparisons if r["slice"] == "context_dependent"]
                self.assertEqual(bool(context), policy == "human_reviewed")
                self.assertTrue(all(r["cohort"] == "grounded" and r["questions"] == 1 for r in context))
