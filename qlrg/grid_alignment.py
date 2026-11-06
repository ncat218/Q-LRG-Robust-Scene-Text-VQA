"""Map raw TextOCR polygons to the row-major merged Qwen image grid."""

import math

import numpy as np


def image_grid(thw, merge=2):
    t, h, w = (int(x) for x in thw)
    if t != 1 or h <= 0 or w <= 0 or h % merge or w % merge:
        raise ValueError(f"Invalid static image grid: {thw}")
    return h // merge, w // merge


def clip_edge(polygon, axis, boundary, sign):
    if not polygon:
        return []
    result = []
    start = polygon[-1]
    start_inside = sign * (start[axis] - boundary) >= 0
    for end in polygon:
        end_inside = sign * (end[axis] - boundary) >= 0
        if start_inside != end_inside:
            ratio = (boundary - start[axis]) / (end[axis] - start[axis])
            result.append(tuple(a + ratio * (b - a) for a, b in zip(start, end)))
        if end_inside:
            result.append(end)
        start, start_inside = end, end_inside
    return result


def overlap_area(polygon, x, y):
    for axis, boundary, sign in ((0, x, 1), (0, x + 1, -1), (1, y, 1), (1, y + 1, -1)):
        polygon = clip_edge(polygon, axis, boundary, sign)
    return abs(sum(
        a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1])
    )) / 2


def contains_center(polygon, x, y):
    cx, cy = x + 0.5, y + 0.5
    inside = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        if (ay > cy) != (by > cy) and cx < ax + (cy - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def project(words, size, grid, rule, threshold):
    if rule not in {"center", "overlap"} or not 0 < threshold <= 1:
        raise ValueError("Invalid mask rule or overlap threshold")
    height, width = grid
    raw_width, raw_height = size
    result = np.zeros(grid, dtype=bool)
    # The pinned processor resizes the full canvas with no crop or padding.
    for word in words:
        polygon = [(x * width / raw_width, y * height / raw_height) for x, y in word["polygon"]]
        xs, ys = zip(*polygon)
        for y in range(max(0, math.floor(min(ys))), min(height, math.ceil(max(ys)))):
            for x in range(max(0, math.floor(min(xs))), min(width, math.ceil(max(xs)))):
                if not result[y, x]:
                    result[y, x] = (
                        contains_center(polygon, x, y) if rule == "center"
                        else overlap_area(polygon, x, y) >= threshold
                    )
    return result


def dilate(mask, radius):
    if radius < 0:
        raise ValueError("Dilation radius must be nonnegative")
    h, w = mask.shape
    padded = np.pad(mask, radius)
    result = np.zeros_like(mask)
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            result |= padded[y:y + h, x:x + w]
    return result


def random_region(mask, seed):
    """Translate the O3 shape, preferring disjoint placements, at equal token count."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("O4 requires a nonempty O3 mask")
    rng = np.random.default_rng(seed)
    height, width = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
    places = [(y, x) for y in range(h - height + 1) for x in range(w - width + 1)]
    places.remove((int(ys.min()), int(xs.min())))
    best = None
    best_overlap = len(ys)
    for index in rng.permutation(len(places)):
        y, x = places[index]
        rows, cols = ys - ys.min() + y, xs - xs.min() + x
        shared = int(mask[rows, cols].sum())
        if shared < best_overlap:
            best, best_overlap = (rows, cols), shared
        if shared == 0:
            break
    result = np.zeros_like(mask)
    if best is not None:
        result[best] = True
        return result, "translated"
    outside = np.flatnonzero(~mask.ravel())
    inside = np.flatnonzero(mask.ravel())
    count = min(len(outside), len(ys))
    result.flat[rng.choice(outside, count, replace=False)] = True
    result.flat[rng.choice(inside, len(ys) - count, replace=False)] = True
    return result, "sampled" if count else "no_spatial_alternative"
