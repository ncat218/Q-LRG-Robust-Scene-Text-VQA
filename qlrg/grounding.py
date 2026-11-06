"""Build an offline review pack for the O3 candidate matches."""

import base64
import html
import io
from pathlib import Path

from PIL import ImageDraw

from .common import atomic_write, read_jsonl, write_jsonl
from .data import load_clean, rank


def write_review_pack(manifest, root, output, limit):
    output = Path(output)
    if limit < 1:
        raise ValueError("Review limit must be positive")
    if output.exists():
        raise FileExistsError(output)
    items = []
    for image in manifest["images"]:
        candidates = [q for q in image["questions"] if q["candidate"]]
        if candidates:
            question = min(candidates, key=lambda q: rank(manifest["seed"], q["question_id"]))
            items.append((image, question))
    if not items:
        raise ValueError("No unambiguous candidate matches were found")
    output.mkdir(parents=True, exist_ok=False)
    rows, panels = [], []
    for image, question in items[:limit]:
        match = question["candidate"]
        row = {
            "image_id": image["image_id"], "image_sha256": image["image_sha256"],
            "question_id": question["question_id"], "word_ids": match["word_ids"],
            "status": "pending", "context_dependent": False, "note": "",
        }
        rows.append(row)
        picture, _, _ = load_clean(root / image["image_rel"], image["image_sha256"])
        width, height = picture.size
        picture.thumbnail((650, 480))
        draw = ImageDraw.Draw(picture)
        for word in image["words"]:
            if word["id"] in row["word_ids"]:
                polygon = [(x * picture.width / width, y * picture.height / height) for x, y in word["polygon"]]
                draw.line(polygon + polygon[:1], fill="#37ff72", width=3)
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=82)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        details = {
            "candidate": match, "answers": question["answers"],
            "ocr_words": {w["id"]: w["text"] for w in image["words"]},
        }
        panels.append(
            f"<section><h2>{html.escape(row['question_id'])}</h2>"
            f"<p>{html.escape(question['question'])}</p>"
            f"<img alt='Candidate box in green' src='data:image/jpeg;base64,{encoded}'>"
            f"<pre>{html.escape(str(details))}</pre></section>"
        )
    write_jsonl(output / "review.jsonl", rows)
    page = (
        "<!doctype html><meta charset='utf-8'><title>O3 grounding review</title>"
        "<style>body{font:16px system-ui;max-width:860px;margin:2rem auto}"
        "section{margin:2rem 0}pre{white-space:pre-wrap}img{max-width:100%}</style>"
        "<h1>O3 grounding review</h1><p>Verify that the green region answers the question."
        " In review.jsonl set status to accepted or rejected. Correct word_ids when needed and"
        " record the reason for rejection. Mark context_dependent if object or scene context matters.</p>"
    )
    atomic_write(output / "review.html", (page + "\n".join(panels)).encode("utf-8"))
    return len(rows)


def read_review(path, manifest):
    questions = {q["question_id"]: image for image in manifest["images"] for q in image["questions"]}
    result = {}
    for row in read_jsonl(path):
        qid = row["question_id"]
        image = questions.get(qid)
        if qid in result or image is None:
            raise ValueError(f"Invalid or duplicate review question {qid}")
        if row["image_id"] != image["image_id"] or row["image_sha256"] != image["image_sha256"]:
            raise ValueError(f"Review refers to a different image for {qid}")
        if row["status"] not in {"accepted", "rejected"}:
            raise ValueError(f"Finalize the review for {qid} before running full S0")
        ids = row["word_ids"]
        if row["status"] == "accepted":
            valid = {word["id"] for word in image["words"]}
            if not ids or len(ids) != len(set(ids)) or not set(ids) <= valid:
                raise ValueError(f"Invalid accepted OCR word IDs for {qid}")
        elif not row.get("note", "").strip():
            raise ValueError(f"Record a rejection reason for {qid}")
        if not isinstance(row.get("context_dependent", False), bool):
            raise ValueError(f"context_dependent must be boolean for {qid}")
        result[qid] = row
    if not any(row["status"] == "accepted" for row in result.values()):
        raise ValueError("Full S0 requires at least one accepted grounding")
    return result
