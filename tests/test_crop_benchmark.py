import unittest

from scripts.crop_benchmark import crop_boxes, fraction_retained
from scripts.dbnet_crop import hybrid_boxes


class CropBenchmarkTest(unittest.TestCase):
    def test_match_by_text_fraction_and_construct_three_distinct_crops(self):
        vlm = (10, 10, 50, 50)
        large_sign = ((30, 10, 95, 50), 0.99)
        text = ((5, 20, 30, 25), 0.55)
        match, fraction, union, chosen, shared = hybrid_boxes(vlm, [large_sign, text])
        self.assertEqual(match, text[0])
        self.assertAlmostEqual(fraction, 0.8)
        self.assertEqual(union, (5, 10, 50, 50))
        self.assertEqual(chosen, (5, 20, 30, 25))
        self.assertEqual(shared, (10, 20, 30, 25))

    def test_fallback_and_matched_random_control(self):
        box = (10, 10, 40, 40)
        boxes, match, donor_match = crop_boxes(
            (100, 100), "image", "qid", 42, box, None, [((39, 39, 80, 80), 0.99)])
        self.assertIsNone(match[0])
        self.assertIsNone(donor_match[0])
        self.assertIsNone(boxes["full_A2"])
        self.assertIsNone(boxes["hybrid_shuffled"])
        self.assertEqual(boxes["sdrpn"], boxes["hybrid_union"])
        self.assertEqual(boxes["sdrpn"], boxes["hybrid_intersection"])
        union, random = boxes["hybrid_union"], boxes["hybrid_random"]
        self.assertNotEqual(union, random)
        self.assertEqual((union[2] - union[0], union[3] - union[1]),
                         (random[2] - random[0], random[3] - random[1]))

    def test_no_vlm_box_keeps_dbnet_only_arm(self):
        boxes, _, _ = crop_boxes((100, 100), "image", "qid", 42, None, None, [((10, 10, 20, 20), 0.9)])
        self.assertEqual(boxes["dbnet"], (8, 8, 22, 22))
        self.assertIsNone(boxes["hybrid_union"])
        self.assertIsNone(boxes["hybrid_random"])

    def test_coverage_uses_polygon_area(self):
        triangle = [(0, 0), (10, 0), (0, 10)]
        self.assertAlmostEqual(fraction_retained((0, 0, 5, 10), triangle), 0.75)
        self.assertAlmostEqual(fraction_retained((0, 0, 10, 10), triangle), 1.0)
        self.assertAlmostEqual(fraction_retained((10, 10, 15, 15), triangle), 0.0)


if __name__ == "__main__":
    unittest.main()
