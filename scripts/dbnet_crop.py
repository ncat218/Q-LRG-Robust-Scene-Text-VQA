"""Pinned DBNet detection and pixel box combinations for the A2 benchmark."""

import math
import sys
from pathlib import Path

from qlrg.common import sha256


MODEL_SHA256 = "62fce3a058ac519952062fbd2cf45bae661130ce753d2a0fc918f4776ba3a02a"
INPUT_SIZE = (1280, 736)
MIN_TEXT_OVERLAP = 0.30


class DBNet:
    def __init__(self, weights, runtime, threads):
        runtime = Path(runtime)
        if not (runtime / "cv2").is_dir() or sha256(weights) != MODEL_SHA256:
            raise ValueError("Expected the isolated OpenCV runtime and pinned ICDAR2015 DBNet weights")
        sys.path.insert(0, str(runtime))
        try:
            import cv2
        finally:
            sys.path.pop(0)
        if cv2.__version__ != "4.10.0" or not Path(cv2.__file__).resolve().is_relative_to(runtime.resolve()):
            raise RuntimeError("Expected the isolated OpenCV 4.10.0 runtime")
        cv2.setNumThreads(threads)
        self.model = cv2.dnn_TextDetectionModel_DB(str(weights))
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.model.setBinaryThreshold(0.30)
        self.model.setPolygonThreshold(0.50)
        self.model.setUnclipRatio(2.0)
        self.model.setMaxCandidates(200)
        self.model.setInputParams(1 / 255, INPUT_SIZE, (122.67891434, 116.66876762, 104.00698793))

    def detect(self, image):
        import numpy as np

        polygons, scores = self.model.detect(np.ascontiguousarray(np.asarray(image)[:, :, ::-1]))
        candidates = []
        for polygon, confidence in zip(polygons, scores, strict=True):
            points = np.asarray(polygon).reshape(-1, 2)
            x, y = points.min(axis=0)
            right, bottom = points.max(axis=0)
            box = (max(0, math.floor(x)), max(0, math.floor(y)),
                   min(image.width, math.ceil(right) + 1), min(image.height, math.ceil(bottom) + 1))
            if box[0] < box[2] and box[1] < box[3]:
                candidates.append((box, float(confidence)))
        return candidates


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(a, b):
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def enclosing_box(a, b):
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def best_text_box(candidates):
    if not candidates:
        return None
    box, _ = max(candidates, key=lambda row: (row[1], area(row[0]), row[0]))
    return box


def hybrid_boxes(vlm, candidates):
    if vlm is None or not candidates:
        return None, 0.0, vlm, vlm, vlm
    vlm = tuple(vlm)
    overlaps = []
    for box, score in candidates:
        shared = intersection(vlm, box)
        pixels = area(shared)
        overlaps.append((pixels / area(box), pixels, score, box, shared))
    fraction, _, _, text, shared = max(overlaps)
    if fraction < MIN_TEXT_OVERLAP:
        return None, fraction, vlm, vlm, vlm
    return text, fraction, enclosing_box(vlm, text), text, shared
