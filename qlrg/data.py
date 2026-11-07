"""Create a stable train-only diagnostic manifest and groundable candidates."""

import hashlib
import io
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image

from .common import object_hash, read_json, sha256

PROTOCOL = "rgb-raw-orientation-pillow11.3.0-bicubic-floor-div4-up-original-v1"


def rank(seed, image_id):
    return hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()


def annotation_file(root, split):
    return root / f"TextVQA_0.5.1_{split}.json"


def read_questions(root, split):
    return read_json(annotation_file(root, split))["data"]


def question_groups(rows):
    groups = {}
    seen = set()
    for row in rows:
        image_id = str(row["image_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
            raise ValueError(f"Invalid image ID: {image_id}")
        question_id = str(row["question_id"])
        if question_id in seen:
            raise ValueError(f"Duplicate question ID: {question_id}")
        seen.add(question_id)
        groups.setdefault(image_id, []).append({
            "question_id": question_id,
            "question": row["question"],
            "answers": row.get("answers", []),
        })
    return groups


def verify_pillow():
    import PIL

    if PIL.__version__ != "11.3.0":
        raise RuntimeError(f"Protocol requires Pillow 11.3.0, found {PIL.__version__}")


def load_clean(path, expected_hash=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash and digest != expected_hash:
        raise ValueError(f"Clean image changed: {path}")
    with Image.open(io.BytesIO(raw)) as image:
        orientation = image.getexif().get(274, 1)
        return image.convert("RGB"), orientation, digest


def bicubic_control(clean):
    width, height = clean.size
    bicubic = Image.Resampling.BICUBIC
    return clean.resize((max(1, width // 4), max(1, height // 4)), bicubic).resize(
        clean.size, bicubic
    )


def normalized_literal(value):
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.casefold().split())


def candidate(question, words, minimum_votes=6):
    """Find a unique literal match. Human review determines actual grounding."""
    votes = Counter(normalized_literal(a) for a in question["answers"])
    if not votes:
        return None
    answer, count = votes.most_common(1)[0]
    if count < minimum_votes or not any(c.isalnum() for c in answer):
        return None
    matches = [word["id"] for word in words if normalized_literal(word["text"]) == answer]
    competing = any(
        normalized_literal(word["text"]) != answer
        and votes.get(normalized_literal(word["text"]), 0) >= 2
        for word in words
    )
    if len(matches) != 1 or competing:
        return None
    return {"word_ids": matches, "answer": answer, "votes": count}


def get_words(ocr, image_id):
    words = []
    seen = set()
    for annotation_id in ocr["imgToAnns"].get(image_id, []):
        annotation_id = str(annotation_id)
        row = ocr["anns"][annotation_id]
        points = row["points"]
        if (
            annotation_id in seen or str(row["image_id"]) != image_id
            or len(points) < 6 or len(points) % 2
            or not all(math.isfinite(x) for x in points)
        ):
            raise ValueError(f"Invalid TextOCR annotation {annotation_id} for {image_id}")
        seen.add(annotation_id)
        words.append({
            "id": annotation_id, "text": row["utf8_string"],
            "polygon": [points[i:i + 2] for i in range(0, len(points), 2)],
        })
    return words


def split_inner(train, seed):
    ranked = sorted(train, key=lambda iid: (rank(seed, iid), iid))
    val_count = max(1, (len(ranked) + 5) // 10)
    return ranked[val_count:], ranked[:val_count]


def build_manifest(textvqa, textocr, limit=500, seed=42):
    verify_pillow()
    if limit < 1:
        raise ValueError("Image limit must be positive")
    train = question_groups(read_questions(textvqa, "train"))
    train_qids = {q["question_id"] for group in train.values() for q in group}
    for split in ("val", "test"):
        heldout = read_questions(textvqa, split)
        heldout_images = {str(row["image_id"]) for row in heldout}
        heldout_qids = {str(row["question_id"]) for row in heldout}
        if train.keys() & heldout_images or train_qids & heldout_qids:
            raise ValueError(f"Official train and {split} overlap")
    train_inner, val_inner = split_inner(train, seed)
    ocr = read_json(textocr)
    records = []
    orientation_audit = Counter()
    eligible = [iid for iid in val_inner if iid in ocr["imgs"] and ocr["imgToAnns"].get(iid)]
    if limit > len(eligible):
        raise ValueError(f"Requested {limit} images but only {len(eligible)} have TextOCR annotations")
    for image_id in eligible[:limit]:
        meta = ocr["imgs"][image_id]
        if meta.get("set") != "train":
            raise ValueError(f"TextOCR image is not train: {image_id}")
        image_rel = f"train_images/{image_id}.jpg"
        clean, orientation, digest = load_clean(textvqa / image_rel)
        if clean.size != (meta["width"], meta["height"]):
            raise ValueError(f"Raw image and TextOCR dimensions differ: {image_id}")
        if orientation != 1:
            orientation_audit["non_identity_exif_orientation"] += 1
        words = get_words(ocr, image_id)
        questions = sorted(train[image_id], key=lambda q: q["question_id"])
        for q in questions:
            if len(q["answers"]) != 10 or not all(isinstance(a, str) for a in q["answers"]):
                raise ValueError(f"Expected ten TextVQA answers for {q['question_id']}")
            q["candidate"] = candidate(q, words)
        records.append({
            "image_id": image_id, "image_rel": image_rel,
            "size": list(clean.size), "exif_orientation": orientation, "image_sha256": digest,
            "words": words, "questions": questions,
        })
    return {
        "schema": 1, "protocol": PROTOCOL, "split": "official_train/val_inner",
        "seed": seed, "requested_images": limit, "train_images": len(train),
        "val_inner_image_ids": val_inner, "train_inner_ids_hash": object_hash(train_inner),
        "eligible_val_inner_images": len(eligible),
        "annotation_sha256": {
            split: sha256(annotation_file(textvqa, split)) for split in ("train", "val", "test")
        },
        "textocr_sha256": sha256(textocr), "orientation_audit": dict(orientation_audit),
        "images": records,
    }
