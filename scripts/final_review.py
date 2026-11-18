"""Paired tables, review worksheets and an offline TextVQA demonstration."""

import base64
import html
import io
import json
import math
import re
from collections import Counter
from pathlib import Path

from qlrg.case_review import write_case_review
from qlrg.common import atomic_write, write_json
from qlrg.evaluation import paired_bootstrap
from qlrg.experiment import paired_images
from scripts.followup_common import write_table


LABELS = {
    "A0": "Qwen gốc",
    "S1b": "phục hồi latent sau Merger, checkpoint job 11305",
    "Pre": "phục hồi latent trước Merger, cấu hình job 11305",
    "A2": "LoRA trên attention LLM, checkpoint job 11305",
}
PAIRS = (("A0", "S1b"), ("A0", "Pre"), ("A0", "A2"),
         ("S1b", "Pre"), ("A2", "S1b"), ("A2", "Pre"))


def index_rows(rows, manifest):
    expected = [(str(image["image_id"]), str(q["question_id"]))
                for image in manifest["images"] for q in image["questions"]]
    if len(expected) != len(set(expected)):
        raise ValueError("Manifest has duplicate questions")
    indexed = {}
    for row in rows:
        key = (row["condition"], row["arm"], str(row["image_id"]), str(row["question_id"]))
        score = float(row["score"])
        if key in indexed or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Duplicate or invalid scored prediction: {key}")
        indexed[key] = row
    required = {(condition, arm, iid, qid)
                for condition in ("CLEAN", "LR4") for arm in LABELS for iid, qid in expected}
    if indexed.keys() != required:
        raise ValueError("Every arm and condition must cover the same manifest questions")
    return expected, indexed


def report(output, rows, manifest, seed=42):
    expected, indexed = index_rows(rows, manifest)
    metadata = {(str(image["image_id"]), str(q["question_id"])): q
                for image in manifest["images"] for q in image["questions"]}
    groups = [iid for iid, _ in expected]
    accuracy, comparisons, cases = [], [], []
    scores = {}
    for condition in ("CLEAN", "LR4"):
        for arm, label in LABELS.items():
            values = [indexed[condition, arm, iid, qid]["score"] for iid, qid in expected]
            scores[condition, arm] = values
            accuracy.append({"condition": condition, "arm": arm, "label": label,
                             "questions": len(expected), "images": len(set(groups)),
                             "accuracy_pct": 100 * sum(values) / len(values)})
        for reference, target in PAIRS:
            left, right = scores[condition, reference], scores[condition, target]
            comparisons.append({"condition": condition, "reference": reference, "target": target,
                                "questions": len(expected), "images": len(set(groups)),
                                "score_improved": sum(b > a for a, b in zip(left, right)),
                                "score_worsened": sum(b < a for a, b in zip(left, right)),
                                "fully_correct_to_lower": sum(a >= 1 - 1e-9 and b < 1 - 1e-9
                                                              for a, b in zip(left, right)),
                                "zero_to_fully_correct": sum(a == 0 and b >= 1 - 1e-9 for a, b in zip(left, right)),
                                **paired_bootstrap(left, right, groups, 10000, seed)})
            for iid, qid in expected:
                a, b = indexed[condition, reference, iid, qid], indexed[condition, target, iid, qid]
                delta = b["score"] - a["score"]
                change = "improved" if delta > 1e-9 else "worsened" if delta < -1e-9 else "unchanged"
                q = metadata[iid, qid]
                majority = Counter(q["answers"]).most_common(1)[0][0]
                longer = (a["score"] >= 1 - 1e-9 and b["score"] < 1 - 1e-9 and
                          len(b["answer"].split()) > len(a["answer"].split()))
                cases.append({"condition": condition, "image_id": iid, "question_id": qid,
                              "question": q["question"], "answers_json": json.dumps(q["answers"], ensure_ascii=False),
                              "reference": reference, "target": target, "reference_answer": a["answer"],
                              "target_answer": b["answer"], "reference_score": a["score"],
                              "target_score": b["score"], "delta_score": delta, "change": change,
                              "numeric_answer_proxy": bool(re.search(r"\d", majority)),
                              "longer_answer_regression_proxy": longer,
                              "manual_reason": "", "manual_review_status": "unreviewed"})
    interactions = []
    for arm in ("S1b", "Pre", "A2"):
        clean = [a - b for a, b in zip(scores["CLEAN", arm], scores["CLEAN", "A0"])]
        lr4 = [a - b for a, b in zip(scores["LR4", arm], scores["LR4", "A0"])]
        interactions.append({"arm": arm, "contrast": "delta_LR4_minus_delta_CLEAN",
                             **paired_bootstrap(clean, lr4, groups, 10000, seed)})
    write_table(output / "accuracy.csv", accuracy)
    write_table(output / "paired.csv", comparisons)
    write_table(output / "error_cases.csv", cases)
    write_table(output / "degradation_interaction.csv", interactions)
    for condition in ("CLEAN", "LR4"):
        write_case_review(output / f"review_{condition.lower()}.html",
                          [row for row in rows if row["condition"] == condition], LABELS,
                          title=f"TextVQA {condition} · {manifest['split']}", manifest=manifest, seed=seed)
    return {"split": manifest["split"], "questions": len(expected), "images": len(set(groups)),
            "bootstrap_unit": "image_id", "bootstrap_resamples": 10000, "seed": seed,
            "accuracy": accuracy, "comparisons": comparisons, "interactions": interactions,
            "eda_policy": "Proxy flags identify cases for human review. Reasons remain unreviewed."}


def demo(output, rows, manifest, root):
    """Show a balanced selection of recorded improvements and regressions with input images."""
    expected, indexed = index_rows(rows, manifest)
    records = {str(image["image_id"]): image for image in manifest["images"]}
    questions = {(str(image["image_id"]), str(q["question_id"])): q
                 for image in manifest["images"] for q in image["questions"]}
    selected = []
    for reference, target in (("A0", "S1b"), ("A0", "Pre"), ("S1b", "Pre"), ("A2", "Pre")):
        for direction in (1, -1):
            ranked = sorted(expected, key=lambda key: direction * (
                indexed[("LR4", target, *key)]["score"] - indexed[("LR4", reference, *key)]["score"]), reverse=True)
            added = 0
            for key in ranked:
                delta = indexed[("LR4", target, *key)]["score"] - indexed[("LR4", reference, *key)]["score"]
                if delta * direction <= 1e-9:
                    break
                if key not in selected:
                    selected.append(key)
                    added += 1
                if added == 3:
                    break
    selected = selected or expected[:3]
    image_data = {}
    cards = []
    for iid, qid in selected:
        if iid not in image_data:
            clean, lr4 = paired_images(root, records[iid])
            image_data[iid] = {}
            for condition, image in (("CLEAN", clean), ("LR4", lr4)):
                image.thumbnail((800, 600))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=85)
                image_data[iid][condition] = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        q = questions[iid, qid]
        table = []
        for arm, label in LABELS.items():
            clean, lr4 = indexed["CLEAN", arm, iid, qid], indexed["LR4", arm, iid, qid]
            table.append(f"<tr><td>{html.escape(arm + ' (' + label + ')')}</td>"
                         f"<td>{html.escape(clean['answer'])} · {clean['score']:.2f}</td>"
                         f"<td>{html.escape(lr4['answer'])} · {lr4['score']:.2f}</td></tr>")
        cards.append(f"<article><h2>Câu {html.escape(qid)}</h2><p>{html.escape(q['question'])}</p>"
                     f"<p>Đáp án chuẩn · {html.escape(' | '.join(dict.fromkeys(q['answers'])))}</p>"
                     f"<div class='images'><figure><figcaption>CLEAN</figcaption><img src='{image_data[iid]['CLEAN']}'></figure>"
                     f"<figure><figcaption>LR4</figcaption><img src='{image_data[iid]['LR4']}'></figure></div>"
                     "<table><tr><th>Cấu hình</th><th>CLEAN · điểm</th><th>LR4 · điểm</th></tr>" +
                     "".join(table) + "</table></article>")
    document = ("<!doctype html><html lang='vi'><meta charset='utf-8'><title>Q-LRG demo</title>"
                "<style>body{font:17px system-ui;max-width:1200px;margin:30px auto;padding:0 18px;color:#17202a}"
                "article{border-top:1px solid #ccd3dc;padding:20px 0}.images{display:flex;gap:16px}"
                "figure{width:50%;margin:0}img{max-width:100%;max-height:420px}"
                "table{border-collapse:collapse;width:100%;margin-top:16px}th,td{border:1px solid #ccd3dc;padding:10px;text-align:left}"
                "</style><h1>Q-LRG · Kết quả từng câu</h1>"
                f"<p>{html.escape(manifest['split'])} · {len(selected)} câu được chọn theo các hướng tăng và giảm điểm.</p>"
                "<p>Ảnh hiển thị là đầu vào CLEAN và LR4. Module phục hồi chỉnh vector feature bên trong Qwen. "
                "Đây là bản phát lại kết quả đã lưu. Nguyên nhân lỗi cần được kiểm chứng thủ công.</p>" +
                "".join(cards) + "</html>")
    atomic_write(output / "demo.html", document.encode("utf-8"))
    write_json(output / "demo_cases.json", [{"image_id": iid, "question_id": qid} for iid, qid in selected])
