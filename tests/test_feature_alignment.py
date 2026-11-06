import unittest

from PIL import Image
import numpy as np

from qlrg.data import bicubic_control, candidate
from qlrg.evaluation import paired_bootstrap
from qlrg.grid_alignment import dilate, image_grid, project, random_region


class AlignmentTest(unittest.TestCase):
    def test_bicubic_floor_size_and_original_canvas(self):
        source = Image.fromarray(np.arange(11 * 15 * 3, dtype=np.uint8).reshape(11, 15, 3))
        expected = source.resize((3, 2), Image.Resampling.BICUBIC).resize((15, 11), Image.Resampling.BICUBIC)
        self.assertEqual(bicubic_control(source).tobytes(), expected.tobytes())

    def test_raw_non_square_polygon_matches_row_major_grid(self):
        words = [{"polygon": [(100, 0), (200, 0), (200, 50), (100, 50)]}]
        grid = image_grid((1, 4, 8))
        self.assertEqual(grid, (2, 4))
        mask = project(words, (400, 100), grid, "overlap", 0.05)
        self.assertEqual(mask.ravel().nonzero()[0].tolist(), [1])
        center = project(words, (400, 100), grid, "center", 0.05)
        np.testing.assert_array_equal(mask, center)
        self.assertEqual(int(dilate(mask, 1).sum()), 6)
        random, mode = random_region(mask, 42)
        self.assertEqual(mode, "translated")
        self.assertEqual(int(random.sum()), 1)
        self.assertFalse((mask & random).any())
        np.testing.assert_array_equal(random, random_region(mask, 42)[0])

    def test_grounding_requires_unique_consensus_match(self):
        question = {"answers": ["Go"] * 7 + ["Stop"] * 3}
        one = {"id": "1", "text": "GO"}
        self.assertEqual(candidate(question, [one])["word_ids"], ["1"])
        self.assertIsNone(candidate(question, [one, {"id": "2", "text": "go"}]))
        self.assertIsNone(candidate(question, [one, {"id": "3", "text": "Stop"}]))

    def test_bootstrap_resamples_images_and_weights_questions(self):
        paired = paired_bootstrap([0, 0, 1], [1, 1, 0], ["same", "same", "other"], 4000, 42)
        self.assertAlmostEqual(paired["delta_pp"], 100 / 3)
        self.assertEqual(paired["ci_low_pp"], -100)
        self.assertEqual(paired["ci_high_pp"], 100)


if __name__ == "__main__":
    unittest.main()
