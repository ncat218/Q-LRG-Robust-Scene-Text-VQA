"""TextVQA scoring and question-weighted paired bootstrap over image clusters."""

import csv
import importlib.util
import io
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .common import atomic_write, sha256, write_json

SCORER_SHA256 = "e10a99e15c4658f8d0d83a05cc536c835922e701171fce0016718bec021b8c0a"
SCORER_REVISION = "72f898ec35af86567423c7a019dd1dc5175d1a5a"


def evaluator():
    source = Path(os.environ.get("QLRG_MMF_SCORER", str(Path(sys.prefix) / "m4c_evaluators.py")))
    if not source.is_file() or sha256(source) != SCORER_SHA256:
        raise RuntimeError(f"Missing or modified pinned MMF scorer: {source}")
    spec = importlib.util.spec_from_file_location("qlrg_mmf_evaluators", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TextVQAAccuracyEvaluator()


def paired_bootstrap(left, right, groups, repeats, seed):
    if len(left) != len(right) or len(left) != len(groups) or not len(left) or repeats < 1:
        raise ValueError("Paired bootstrap requires aligned nonempty scores and image IDs")
    delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    _, group_index = np.unique(groups, return_inverse=True)
    counts = np.bincount(group_index)
    sums = np.bincount(group_index, weights=delta)
    rng = np.random.default_rng(seed)
    estimates = []
    # Chunk resamples to keep memory bounded at larger diagnostic sizes.
    for start in range(0, repeats, 256):
        draw = rng.integers(len(counts), size=(min(256, repeats - start), len(counts)))
        estimates.extend(sums[draw].sum(axis=1) / counts[draw].sum(axis=1))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "delta_pp": float(delta.mean() * 100), "ci_low_pp": float(low * 100),
        "ci_high_pp": float(high * 100),
    }


def summarize(output, rows, manifest, stage, repeats, seed, grounding_policy=None):
    score_fn = evaluator()
    answer_key = {
        q["question_id"]: q["answers"] for image in manifest["images"] for q in image["questions"]
    }
    by_question = {}
    for row in rows:
        arms = by_question.setdefault(row["question_id"], {})
        if row["arm"] in arms:
            raise ValueError(f"Duplicate question/arm in report: {row['question_id']}")
        refs = score_fn._compute_answer_scores(answer_key[row["question_id"]])
        row["score"] = refs.get(score_fn.answer_processor(row["answer"]), 0.0)
        arms[row["arm"]] = row
    if answer_key.keys() != by_question.keys():
        raise ValueError("Predictions do not cover the manifest questions")
    if stage == "d0":
        comparisons = [("all", "LR4-BIC-CONTROL", "CLEAN")]
    else:
        comparisons = [
            ("all", "O0", "O1"), ("all", "O0", "O2"),
            *[("grounded", "O0", arm) for arm in ("O1", "O2", "O3", "O4", "O5")],
            ("grounded", "O2", "O3"), ("grounded", "O4", "O3"), ("grounded", "O3", "O5"),
        ]
    summaries = []
    for cohort, reference, target in comparisons:
        eligible = [
            (qid, arms) for qid, arms in by_question.items()
            if cohort == "all" or "O3" in arms
        ]
        for slice_name in ("all", "numeric_or_date", "small_answer_text", "context_dependent"):
            if stage == "d0" and slice_name != "all":
                continue
            pairs = []
            for qid, arms in eligible:
                refs = Counter(answer_key[qid])
                tags = {
                    "all": True,
                    "numeric_or_date": bool(re.search(r"\d", refs.most_common(1)[0][0])),
                    "small_answer_text": cohort == "grounded" and arms["O3"].get("small_answer_text") is True,
                    "context_dependent": (
                        cohort == "grounded" and grounding_policy == "human_reviewed"
                        and arms["O3"].get("context_dependent", False)
                    ),
                }
                if tags[slice_name]:
                    pairs.append((arms[reference], arms[target]))
            if not pairs:
                continue
            left, right = zip(*pairs)
            a = np.array([row["score"] for row in left])
            b = np.array([row["score"] for row in right])
            groups = [row["image_id"] for row in left]
            summaries.append({
                "cohort": cohort, "slice": slice_name, "reference": reference, "target": target,
                **({"grounding_policy": grounding_policy} if stage == "s0" else {}),
                "questions": len(pairs), "images": len(set(groups)),
                "reference_accuracy_pct": float(a.mean() * 100), "target_accuracy_pct": float(b.mean() * 100),
                **paired_bootstrap(a, b, groups, repeats, seed),
                "fully_correct_to_lower": int(((a == 1) & (b < 1)).sum()),
                "lower_to_fully_correct": int(((a < 1) & (b == 1)).sum()),
                "score_improved": int((b > a).sum()), "score_worsened": int((b < a).sum()),
                "mean_reference_mask_fraction": float(np.mean([row.get("mask_fraction", 0) for row in left])),
                "mean_target_mask_fraction": float(np.mean([row.get("mask_fraction", 0) for row in right])),
            })
    if not summaries:
        raise ValueError("No reportable pairs")
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)
    name = "oracle_summary" if stage == "s0" else "baseline_summary"
    atomic_write(output / f"{name}.csv", table.getvalue().encode("utf-8"))
    write_json(output / f"{name}.json", {
        "metric": "TextVQA MMF soft accuracy", "scorer_revision": SCORER_REVISION,
        "scorer_sha256": SCORER_SHA256, "cluster": "image_id",
        "repeats": repeats, "seed": seed, "comparisons": summaries,
        "slices": {
            "numeric_or_date": "Digit-containing most frequent raw answer, a proxy for numeric/date questions",
            "small_answer_text": "Grounded answers with a selected OCR polygon at most one merged cell high",
            "context_dependent": "Grounded questions flagged by a human reviewer",
        },
    })
    return summaries


def oracle_decision_memo(mode, grounded, questions, grounding_policy, comparisons):
    """Separate evidence for localized substitution from the clean reference gap."""
    by_key = {(r["cohort"], r["slice"], r["reference"], r["target"]): r for r in comparisons}
    text_effect = by_key["all", "all", "O0", "O2"]
    answer_vs_lr4 = by_key.get(("grounded", "all", "O0", "O3"))
    answer_vs_random = by_key.get(("grounded", "all", "O4", "O3"))
    text = [
        "# S0 diagnostic memo", "",
        f"Mode {mode}. Grounding usable for {grounded}/{questions} questions.",
        f"Grounding policy: {grounding_policy}.",
        "O0 (unaltered LR4 features), O1 (all clean features), O2 (clean features in TextOCR regions), "
        "O3 (clean features in the annotated answer region), O4 (random region with the O3 token budget), "
        "O5 (O3 with dilation).", "",
        "## Localized interventions", "",
    ]
    if text_effect["ci_low_pp"] > 0:
        text.append("- O2 improves over O0 on this diagnostic set. This measures the value of substituting known clean text features.")
    else:
        text.append("- O2 has no clear positive effect over O0. Check mask coverage, alignment and scoring.")
    if answer_vs_lr4 is None or answer_vs_random is None:
        text.append("- No usable paired answer-region contrast is available.")
    elif min(answer_vs_lr4["ci_low_pp"], answer_vs_random["ci_low_pp"]) > 0:
        text.append("- O3 improves over both O0 and its equal-token random control O4 on the grounded cohort.")
    else:
        text.append("- O3 does not show a clear positive effect over both O0 and O4 on the grounded cohort.")
    if grounding_policy != "human_reviewed":
        text.append("- Answer-region evidence is provisional. Review grounding and coverage before generalizing.")
    text.extend([
        "- O4 matches the O3 token budget. Comparisons with O2 or O5 may also reflect mask size.",
        "- Slice estimates are exploratory. Check sample sizes and the direction of paired effects.", "",
        "## Clean reference", "",
        "O1 approximates native CLEAN after the identity gate. Its gap from O0 describes clean-versus-LR4 "
        "headroom on the same cohort and does not drive the localized-intervention decision.",
    ])
    for cohort, arm in (("all", "O2"), ("grounded", "O3")):
        clean = by_key.get((cohort, "all", "O0", "O1"))
        localized = by_key.get((cohort, "all", "O0", arm))
        if clean and localized and clean["delta_pp"] > 0:
            recovered = 100 * localized["delta_pp"] / clean["delta_pp"]
            text.append(f"- {arm} recovers {recovered:.1f}% of the O1–O0 gap on {cohort} questions (point estimate).")
    text.extend(["", "## Paired 95% image-cluster intervals", ""])
    text.extend(
        f"- {r['target']} vs {r['reference']} ({r['cohort']}/{r['slice']}, "
        f"{r['questions']} questions, {r['images']} images): "
        f"{r['delta_pp']:+.2f} pp [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]"
        for r in comparisons
    )
    return "\n".join(text) + "\n"
