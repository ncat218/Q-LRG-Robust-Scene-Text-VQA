"""Write a self-contained browser report from scored predictions and saved masks."""

import base64
import itertools
import json
import math
from pathlib import Path

from .common import atomic_write
from .evaluation import paired_bootstrap


S0_LABELS = {
    "O0": "giữ feature ảnh mờ Bicubic",
    "O1": "thay toàn bộ bằng feature ảnh sạch",
    "O2": "thay feature mọi vùng chữ TextOCR",
    "O3": "thay vùng đáp án xác định từ nhãn",
    "O4": "thay vùng ngẫu nhiên cùng số token",
    "O5": "thay vùng đáp án và vùng lân cận",
}


def write_case_review(path, predictions, labels, *, title, manifest=None, overlays_dir=None,
                      grounding_policy=None, repeats=10000, seed=42, image_overlays=None):
    """Render scored rows grouped by question; optional masks are named QUESTION-ARM.png."""
    rows = {}
    for p in predictions:
        qid = str(p["question_id"])
        image_id = str(p["image_id"])
        arm = p["arm"]
        r = rows.setdefault(qid, {"id": qid, "img": image_id, "arms": {}})
        if r["img"] != image_id or arm in r["arms"] or arm not in labels:
            raise ValueError(f"Invalid or duplicate case-review row: {qid}/{arm}")
        score = float(p["score"])
        if not 0 <= score <= 1:
            raise ValueError(f"Invalid TextVQA score for {qid}/{arm}: {score}")
        total = math.prod(p["grid_thw"]) // 4 if p.get("grid_thw") else None
        fraction = p.get("mask_fraction")
        replaced = round(fraction * total) if total and fraction is not None else None
        r["arms"][arm] = [p["answer"], score, replaced, total]

    if not rows:
        raise ValueError("Case review requires at least one scored prediction")
    for image in (manifest or {}).get("images", []):
        for question in image["questions"]:
            row = rows.get(str(question["question_id"]))
            if row is None:
                continue
            if row["img"] != str(image["image_id"]):
                raise ValueError(f"Case-review manifest image differs for {row['id']}")
            row["question"] = question["question"]
            row["answers"] = question.get("answers", [])

    used = {arm for r in rows.values() for arm in r["arms"]}
    active = {arm: labels[arm] for arm in labels if arm in used}
    pairs = {}
    for a, b in itertools.combinations(sorted(active), 2):
        paired = [r for r in rows.values() if a in r["arms"] and b in r["arms"]]
        if not paired:
            continue
        left = [r["arms"][a][1] for r in paired]
        right = [r["arms"][b][1] for r in paired]
        groups = [r["img"] for r in paired]
        interval = paired_bootstrap(left, right, groups, repeats, seed)
        pairs[f"{a}:{b}"] = {
            "n": len(paired), "images": len(set(groups)),
            "a": 100 * sum(left) / len(left), "b": 100 * sum(right) / len(right),
            "d": interval["delta_pp"], "lo": interval["ci_low_pp"], "hi": interval["ci_high_pp"],
            "up": sum(y - x > 1e-9 for x, y in zip(left, right)),
            "down": sum(x - y > 1e-9 for x, y in zip(left, right)),
        }

    overlays = {}
    if overlays_dir is not None:
        known = {f"{qid}-{arm}" for qid, r in rows.items() for arm in r["arms"]}
        for overlay in sorted(overlays_dir.glob("*.png")):
            if overlay.stem in known:
                overlays[overlay.stem] = base64.b64encode(overlay.read_bytes()).decode("ascii")

    if image_overlays is not None:
        known = {f"{r['img']}-{arm}" for r in rows.values() for arm in r["arms"]}
        for overlay in sorted(image_overlays):
            if overlay.stem in known:
                overlays[overlay.stem] = base64.b64encode(overlay.read_bytes()).decode("ascii")

    defaults = next((pair.split(":") for pair in pairs), list(active)[:2])
    if len(defaults) == 1:
        defaults *= 2
    if "O0:O2" in pairs:
        defaults = ["O0", "O2"]
    if "A0:S1b" in pairs:
        defaults = ["A0", "S1b"]
    data = {
        "title": title, "labels": active, "defaults": defaults, "rows": list(rows.values()),
        "images": len({r["img"] for r in rows.values()}), "predictionCount": len(predictions),
        "groundingPolicy": grounding_policy, "pairs": pairs, "overlays": overlays, "heatmaps": image_overlays is not None,
    }
    template = Path(__file__).with_name("case_review.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # Data is parsed from a JSON script element. Prevent prediction text from closing it.
    atomic_write(path, template.replace("@@DATA@@", payload.replace("<", "\\u003c")).encode("utf-8"))
