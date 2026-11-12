"""Compare frozen Vision-RL² and SD-RPN boxes with crop-only S1b restoration."""

import argparse
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from filelock import FileLock

from qlrg.common import object_hash, read_json, read_jsonl, sha256, write_json
from qlrg.evaluation import SCORER_REVISION, evaluator, paired_bootstrap
from qlrg.experiment import paired_images
from qlrg.s1_run import make_adapter
from scripts.crop_fusion import FusionQwen, TOTAL_TOKENS, VIEW_TOKENS, fit_token_canvas
from scripts.followup_common import bind, finish, load_study, write_table


MODE = "crop_restore"
SELECTORS = ("sdrpn_3b", "visionrl2_7b")
LAYOUTS = ("crop576", "fusion576", "fusion_native")
BASELINES = {"full576": "crop576", "full_native": "fusion_native"}
RESTORE = ("off", "s1b_crop")


def arm_id(selector, layout, restoration):
    return f"{selector}__{layout}__{restoration}"


ARMS = tuple(BASELINES) + tuple(arm_id(s, l, r) for s in SELECTORS for l in LAYOUTS for r in RESTORE)


def load_boxes(source, manifest_hash, checkpoint):
    archived = read_json(source / "run.json")
    complete = read_json(source / "COMPLETE.json")
    if complete["stage"] != "crop" or complete["binding"] != object_hash(archived):
        raise ValueError("Invalid selector benchmark completion binding")
    predictions = source / "predictions.jsonl"
    if sha256(predictions) != complete["artifacts"].get(predictions.name):
        raise ValueError("Archived selector predictions do not match the completion marker")
    if (archived["manifest_hash"] != manifest_hash or archived["reader_tokens"] != TOTAL_TOKENS
        or archived["padding_per_side"] != 0.20
        or archived["checkpoints"]["A2"]["sha256"] != checkpoint["sha256"]):
        raise ValueError("Selector source differs in cohort, padding, token budget or A2 checkpoint")
    proposals = {s: {} for s in SELECTORS}
    crop_arms = {f"{s}_crop": s for s in SELECTORS}
    for row in read_jsonl(predictions):
        if row["arm"] not in crop_arms:
            continue
        key = row["image_id"], row["question_id"]
        entries = proposals[crop_arms[row["arm"]]]
        if key in entries:
            raise ValueError(f"Duplicate archived box: {key}")
        entries[key] = {"box": tuple(row["crop_xyxy"]) if row["crop_xyxy"] is not None else None,
                        "selector_seconds": row["selector_seconds"],
                        "textocr_retained": row.get("answer_region_retained")}
    return proposals, archived


def input_views(lr4, box, layout):
    if box is None:
        return (lr4,), (None if layout == "fusion_native" else TOTAL_TOKENS), None
    crop = lr4.crop(box)
    if layout == "crop576":
        return (crop,), TOTAL_TOKENS, 0
    return (lr4, crop), (VIEW_TOKENS if layout == "fusion576" else None), 1


def read_layout(model, restorer, scorer, question, refs, image, box, layout, check_identity):
    torch.cuda.synchronize()
    start = time.perf_counter()
    views, tokens, crop_index = input_views(image, box, layout)
    if tokens is not None:
        views = [fit_token_canvas(view, tokens)[0] for view in views]
    batch, grids = model.prepare_views(list(views), question["question"], exact_budget=tokens is not None)
    features = model.encode_views(batch)
    if len(features) != len(grids):
        raise ValueError("Visual encoder changed the number of views")
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - start
    identity = None
    if check_identity and box is not None:
        expected = model.prefill_logits(batch)
        injected = model.prefill_logits(batch, features)
        if not torch.equal(expected, injected):
            raise RuntimeError(f"{layout}: identity injection changed logits")
        identity = {"question_id": question["question_id"], "logits_exact": True,
                    "visual_tokens": sum(h * w for h, w in grids)}

    def predict(selected, restore_seconds):
        result = model.generate(batch, selected)
        return {**result, "score": refs.get(scorer.answer_processor(result["answer"]), 0.0),
                "view_tokens": [h * w for h, w in grids], "visual_tokens": sum(h * w for h, w in grids),
                "view_grids": [list(g) for g in grids], "prompt_tokens": batch["input_ids"].shape[1],
                "restore_seconds": restore_seconds,
                "reader_seconds": encode_seconds + restore_seconds + result["generation_seconds"]}

    base = predict(features, 0.0)
    if crop_index is None:
        return base, base, identity

    torch.cuda.synchronize()
    start = time.perf_counter()
    restored, _ = restorer(features[crop_index], grids[crop_index])
    if restored.shape != features[crop_index].shape or not torch.isfinite(restored).all():
        raise RuntimeError("S1b produced an invalid crop feature tensor")
    replacement = list(features)
    replacement[crop_index] = restored.to(model.dtype)
    torch.cuda.synchronize()
    return base, predict(tuple(replacement), time.perf_counter() - start), identity


@torch.inference_mode()
def predict_image(model, restorer, scorer, image, proposals, root, checked_layouts):
    _, lr4 = paired_images(root, image)
    rows, gates = [], {}
    for question in image["questions"]:
        qid = question["question_id"]
        key = image["image_id"], qid
        case = {"image_id": key[0], "question_id": qid}
        refs = scorer._compute_answer_scores(question["answers"])
        results = {}
        for arm, layout in BASELINES.items():
            base, _, _ = read_layout(model, restorer, scorer, question, refs, lr4, None, layout, False)
            results[layout, None] = (base, base)
            rows.append({**base, **case, "arm": arm, "crop_xyxy": None, "textocr_retained": None,
                         "selector_seconds": 0.0, "pipeline_seconds": base["reader_seconds"]})
        results["fusion576", None] = results["crop576", None]
        for selector in SELECTORS:
            proposal = proposals[selector][key]
            box = proposal["box"]
            if box is not None and not (0 <= box[0] < box[2] <= lr4.width and 0 <= box[1] < box[3] <= lr4.height):
                raise ValueError(f"Archived box is outside the image: {selector} {key} {box}")
            for layout in LAYOUTS:
                if (layout, box) not in results:
                    off, restored, identity = read_layout(model, restorer, scorer, question, refs, lr4, box, layout,
                                                          layout not in checked_layouts and layout not in gates)
                    results[layout, box] = off, restored
                    if identity is not None:
                        gates[layout] = identity
                selector_seconds = proposal["selector_seconds"]
                for restoration, result in zip(RESTORE, results[layout, box]):
                    rows.append({**result, **case, "arm": arm_id(selector, layout, restoration),
                                 "crop_xyxy": box, "textocr_retained": proposal["textocr_retained"],
                                 "selector_seconds": selector_seconds,
                                 "pipeline_seconds": selector_seconds + result["reader_seconds"]})
    return {"rows": rows, "identity": gates}


def report(output, rows, manifest, seed, identity, smoke):
    cases = [(im["image_id"], q["question_id"]) for im in manifest["images"] for q in im["questions"]]
    by_arm = {arm: [] for arm in ARMS}
    for row in rows:
        arm = row["arm"]
        if ("fusion576" in arm and row["crop_xyxy"] is not None
            and row["view_tokens"] != [VIEW_TOKENS, VIEW_TOKENS]):
            raise ValueError(f"Incorrect fair fusion split: {arm}")
        if ("576" in arm and row["visual_tokens"] != TOTAL_TOKENS
            or sum(row["view_tokens"]) != row["visual_tokens"]):
            raise ValueError(f"Incorrect reader token count: {arm}")
        by_arm[arm].append(row)
    if set(identity) != set(LAYOUTS) or any(not gate["logits_exact"] for gate in identity.values()):
        raise ValueError("Missing identity gates for the three layouts")
    metrics, scores = [], {}
    for arm, predictions in by_arm.items():
        if [(row["image_id"], row["question_id"]) for row in predictions] != cases:
            raise ValueError(f"Incorrect evaluation cohort: {arm}")
        scores[arm] = np.asarray([row["score"] for row in predictions])
        tokens = [row["visual_tokens"] for row in predictions]
        row = {"arm": arm, "questions": len(cases), "accuracy_pct": float(scores[arm].mean() * 100),
               "visual_tokens_mean": mean(tokens), "visual_tokens_p95": float(np.percentile(tokens, 95)),
               "visual_tokens_max": max(tokens), "fallbacks": sum(r["crop_xyxy"] is None for r in predictions)
               if arm not in BASELINES else 0}
        for component in ("selector", "restore", "reader", "pipeline"):
            timings = [1000 * r.get(f"{component}_seconds", 0.0) for r in predictions]
            for p in (50, 95):
                row[f"{component}_p{p}_ms"] = float(np.percentile(timings, p))
        metrics.append(row)

    pairs = [(arm_id(SELECTORS[0], layout, flag), arm_id(SELECTORS[1], layout, flag))
             for layout in LAYOUTS for flag in RESTORE]
    pairs.extend((arm_id(s, layout, "off"), arm_id(s, layout, "s1b_crop"))
                 for s in SELECTORS for layout in LAYOUTS)
    pairs.extend((arm_id(s, "fusion576", flag), arm_id(s, "fusion_native", flag))
                 for s in SELECTORS for flag in RESTORE)
    pairs.extend(("full_native" if layout == "fusion_native" else "full576", arm_id(s, layout, flag))
                 for s in SELECTORS for layout in LAYOUTS for flag in RESTORE)
    common = [i for i in range(len(cases)) if all(by_arm[arm_id(s, "crop576", "off")][i]["crop_xyxy"]
                                                is not None for s in SELECTORS)]
    grounded = [i for i in common if all(
        (by_arm[arm_id(s, "crop576", "off")][i]["textocr_retained"] or 0) >= 0.9 for s in SELECTORS)]
    cohorts = {"all": list(range(len(cases))), "both_boxes": common, "both_retain_90pct": grounded}
    comparisons, interactions = [], []
    for cohort, indexes in cohorts.items():
        if not indexes:
            continue
        groups = [cases[i][0] for i in indexes]
        for left, right in pairs:
            a, b = scores[left][indexes], scores[right][indexes]
            comparisons.append({"cohort": cohort, "reference": left, "target": right, "questions": len(groups),
                                **paired_bootstrap(a, b, groups, 10000, seed)})
        for layout in LAYOUTS:
            a, b = [(scores[arm_id(s, layout, "s1b_crop")]-scores[arm_id(s, layout, "off")])[indexes]
                    for s in SELECTORS]
            interactions.append({"cohort": cohort, "layout": layout, "questions": len(groups),
                                 **paired_bootstrap(a, b, groups, 10000, seed)})
    write_table(output / "accuracy.csv", metrics)
    write_table(output / "paired.csv", comparisons)
    write_table(output / "restoration_interaction.csv", interactions)
    return {"split": manifest["split"], "smoke": smoke, "questions": len(cases), "images": len(manifest["images"]),
            "cohorts": {name: len(i) for name, i in cohorts.items()}, "accuracy": metrics,
            "comparisons": comparisons, "restoration_interaction": interactions,
            "identity": identity, "bootstrap_unit": "image_id", "bootstrap_resamples": 10000,
            "scorer_revision": SCORER_REVISION}


def run(output):
    with FileLock(str(output / ".run.lock"), timeout=0):
        launch, paths, config, manifest, checkpoints, binding = load_study(output, MODE, ("A2", "S1b"))
        source = Path(launch["selector_benchmark"])
        proposals, archived = load_boxes(source, binding["manifest_hash"], checkpoints["A2"]["provenance"])
        expected = {(im["image_id"], q["question_id"]) for im in manifest["images"] for q in im["questions"]}
        if any(entries.keys() != expected for entries in proposals.values()):
            raise ValueError("Archived box results do not cover the 5,000 official questions")
        smoke_images = launch.get("smoke_images")
        if smoke_images is not None:
            manifest = {**manifest, "images": manifest["images"][:smoke_images]}
        binding.update({"selector_benchmark": str(source), "selector_binding": object_hash(archived),
                        "smoke_images": smoke_images, "arms": list(ARMS), "native_max_pixels_per_view": config["max_pixels"],
                        "worker_sha256": sha256(Path(__file__))})
        bind(output, binding)
        model = FusionQwen(config, paths["hf_hub"])
        torch.backends.cuda.matmul.allow_tf32 = False
        adapters = {}
        for name, record in checkpoints.items():
            adapter = make_adapter(model, name, record["config"]).eval()
            adapter.load_state_dict(record["state"])
            adapters[name] = adapter
        scorer = evaluator()
        rows, identity = [], {}
        with adapters["A2"].attached():
            for index, im in enumerate(manifest["images"], 1):
                shard = output / "shards" / f"{im['image_id']}.json"
                batch = read_json(shard) if shard.is_file() else predict_image(
                    model, adapters["S1b"], scorer, im, proposals, paths["textvqa"], identity,
                )
                if not shard.is_file():
                    write_json(shard, batch)
                rows.extend(batch["rows"])
                identity.update(batch["identity"])
                if index <= 3 or index % 50 == 0 or index == len(manifest["images"]):
                    print(f"Crop S1b: {index}/{len(manifest['images'])} images, gates: {sorted(identity)}", flush=True)
        finish(output, binding, report(output, rows, manifest, launch["seed"], identity, smoke_images is not None),
               rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
