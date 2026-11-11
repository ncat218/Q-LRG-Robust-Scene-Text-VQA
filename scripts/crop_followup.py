"""Benchmark public query-conditioned crop selectors using the same A2 reader."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import mean
from urllib.request import urlopen

import numpy as np
import torch
from filelock import FileLock

from qlrg.common import atomic_write, object_hash, read_json, sha256, write_json
from qlrg.evaluation import SCORER_REVISION, evaluator, paired_bootstrap
from qlrg.experiment import paired_images
from qlrg.s1_manifests import read_s1_manifests
from qlrg.s1_run import make_adapter
from scripts.crop_benchmark import answer_polygons, fraction_retained
from scripts.crop_fusion import FusionQwen, TOTAL_TOKENS, VIEW_TOKENS
from scripts.followup_common import bind, finish, load_study, write_table
from scripts.fusion_confirm import read_a2_views
from scripts.query_crop import PADDING, padded_box
from scripts.sdrpn_runtime import selector_environment
from scripts.selector_assets import DATA_REVISION, MODELS, UPSTREAM_REVISION, prepare
from scripts.vicrop_select import UPSTREAM_REVISION as VICROP_REVISION, ViCrop, calibrate


SELECTORS = ("sdrpn_3b", "vicrop_3b", *MODELS)
ARMS = ("full_A2", *(f"{arm}_{view}" for arm in SELECTORS for view in ("fusion", "crop")))
LABELS = {"full_A2": "A2 · toàn ảnh", "sdrpn_3b": "SD-RPN 3B", "vicrop_3b": "ViCrop relative attention 3B",
          "sdrpn_7b": "SD-RPN 7B", "visionrl2_7b": "Vision-RL² 7B"}


TEXTOCR_VAL_URL = "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json"
TEXTOCR_VAL_SHA256 = "0d04c1c9f5ba99e90be5b0df9ab34f6237b3fbc55443f263133d5c49b6435918"


def validation_ocr(repo):
    path = repo / "local/TextOCR_0.1_val.json"
    if not path.is_file():
        with urlopen(TEXTOCR_VAL_URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != TEXTOCR_VAL_SHA256:
            raise ValueError("Unexpected TextOCR validation download")
        atomic_write(path, data)
    if sha256(path) != TEXTOCR_VAL_SHA256:
        raise ValueError(f"Unexpected TextOCR validation annotation: {path}")
    return path


def pad(box, size):
    return padded_box(box, size, PADDING["pad20"]) if box is not None else None


def overlap_audit(manifest, files):
    eval_ids = {str(im["image_id"]) for im in manifest["images"]}
    results = {}
    for path in files:
        image_ids = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                name = Path(str(row["image"])).stem
                dataset = str(row.get("dataset", "")).lower().replace("_", "").replace("-", "")
                if "textvqa" in dataset or name in eval_ids:
                    image_ids.add(name)
        results[path.name] = {"image_ids": len(image_ids), "official_val_overlap": sorted(eval_ids & image_ids),
                              "sha256": sha256(path)}
    return {"data_revision": DATA_REVISION, "files": results,
            "scope": "Released VisualCoT candidate pool and Vision-RL² 7B RL pool; full original SD-RPN corpora unverified."}


def run_7b(launch, paths, config, manifest, output, vendor, checkpoints):
    request = {"textvqa": str(paths["textvqa"]), "prompt": config["prompt"], "seed": launch["seed"],
               "images": [{**{key: record[key] for key in ("image_id", "image_rel", "image_sha256", "size",
                                                         "exif_orientation")},
                           "questions": [{"question_id": q["question_id"], "query": q["question"]}
                                         for q in record["questions"]]} for record in manifest["images"]]}
    request_path = output / "selector-request.json"
    write_json(request_path, request)
    env = selector_environment(launch)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "XDG_CACHE_HOME": str(output / ".selector-cache"), "PYTHONDONTWRITEBYTECODE": "1"})
    for arm, checkpoint in checkpoints.items():
        subprocess.run([
            str(Path(launch["sdrpn_env"]) / "bin/python"), "-B", str(
                Path(launch["repo"]) / "scripts/visionrl2_select.py"),
            "--request", str(request_path), "--output", str(output / arm), "--vendor", str(vendor),
            "--checkpoint", str(checkpoint), "--processor", str(checkpoints["sdrpn_7b"]),
        ], cwd=launch["repo"], env=env, check=True)


def proposal_map(directory, record, image, wrapped=False):
    value = read_json(directory / f"{object_hash(record['image_id'])[:24]}.json")
    if (value["image_id"] != record["image_id"]
        or value["lr4_sha256"] != hashlib.sha256(image.tobytes()).hexdigest()
        or set(value["questions"]) != {str(q["question_id"]) for q in record["questions"]}):
        raise ValueError(f"Proposal source used different pixels or question IDs for {record['image_id']}")
    return {key: data["query"] if wrapped else data for key, data in value["questions"].items()}


@torch.inference_mode()
def predict(model, lora, scorer, vicrop, layer, record, paths, source, output, check_identity):
    _, lr4 = paired_images(paths["textvqa"], record)
    proposals = {"sdrpn_3b": proposal_map(source / "sdrpn-proposals", record, lr4, wrapped=True)}
    proposals.update({arm: proposal_map(output / arm, record, lr4) for arm in MODELS})
    state = vicrop.start_image(lr4, (layer,))
    rows, identity = [], None
    for question in record["questions"]:
        qid = question["question_id"]
        refs = scorer._compute_answer_scores(question["answers"])
        boxes, shared_seconds, query_seconds = vicrop.locate(lr4, question["question"], state)
        regions = {arm: data[str(qid)] for arm, data in proposals.items()}
        regions["vicrop_3b"] = {"box": boxes[layer], "selector_seconds": shared_seconds + query_seconds}
        full, _ = read_a2_views(model, lora, scorer, question, refs, lr4, None, None, None, False)
        rows.append({**full, "image_id": record["image_id"], "question_id": qid, "arm": "full_A2",
                     "raw_box": None, "selector_seconds": 0.0, "pipeline_seconds": full["reader_seconds"]})
        predictions = {}
        for arm, region in regions.items():
            raw = region["box"]
            box = pad(raw, lr4.size)
            for view in ("fusion", "crop"):
                if box is None:
                    reader = full
                else:
                    key = (box, view)
                    if key not in predictions:
                        predictions[key], gate = read_a2_views(
                            model, lora, scorer, question, refs, lr4, box, None, None,
                            check_identity and identity is None and view == "fusion", crop_only=view == "crop",
                        )
                        identity = identity or gate
                    reader = predictions[key]
                seconds = region["selector_seconds"]
                rows.append({**reader, "image_id": record["image_id"], "question_id": qid, "arm": f"{arm}_{view}",
                             "raw_box": raw, "selector_seconds": seconds,
                             "pipeline_seconds": seconds + reader["reader_seconds"]})
    return {"rows": rows, "identity": identity}


def report(output, rows, manifest, polygons, seed, identity, audit, calibration):
    cases = [(record["image_id"], q["question_id"]) for record in manifest["images"] for q in record["questions"]]
    groups = [iid for iid, _ in cases]
    by_arm = {arm: [] for arm in ARMS}
    for row in rows:
        if row["visual_tokens"] != TOTAL_TOKENS:
            raise ValueError("Crop benchmark changed the visual token budget")
        expected_views = [VIEW_TOKENS, VIEW_TOKENS] if row["arm"].endswith("_fusion") and row["crop_xyxy"] else [
            TOTAL_TOKENS]
        if row["view_tokens"] != expected_views:
            raise ValueError(f"Incorrect reader layout: {row['arm']}")
        by_arm[row["arm"]].append(row)
    if not identity or not identity["logits_exact"]:
        raise ValueError("Missing two-view identity check")
    tables, scores = [], {}
    for arm, predictions in by_arm.items():
        if [(row["image_id"], row["question_id"]) for row in predictions] != cases:
            raise ValueError(f"Wrong evaluation cohort for {arm}")
        scores[arm] = [r["score"] for r in predictions]
        fractions = []
        for row in predictions:
            polygon = polygons.get(row["question_id"])
            if polygon is not None and row["crop_xyxy"] is not None:
                row["answer_region_retained"] = fraction_retained(row["crop_xyxy"], polygon)
                fractions.append(row["answer_region_retained"])
        label = LABELS[arm] if arm == "full_A2" else LABELS[arm.rsplit("_", 1)[0]]
        table = {"arm": arm, "selector": label, "questions": len(cases), "accuracy_pct": 100 * mean(scores[arm]),
                 "visual_tokens": TOTAL_TOKENS, "grounded_questions": len(polygons),
                 "grounded_cropped": len(fractions), "grounded_retains_90pct": sum(f >= 0.90 for f in fractions),
                 "fallback_questions": sum(r["crop_xyxy"] is None for r in predictions) if arm != "full_A2" else 0}
        for component in ("selector", "reader", "pipeline"):
            timings = [1000 * row[f"{component}_seconds"] for row in predictions]
            table[f"{component}_p50_ms"] = float(np.median(timings))
            table[f"{component}_p95_ms"] = float(np.percentile(timings, 95))
        tables.append(table)
    pairs = [("full_A2", arm) for arm in ARMS if arm != "full_A2"]
    for view in ("fusion", "crop"):
        pairs.extend((f"sdrpn_3b_{view}", f"{arm}_{view}") for arm in SELECTORS if arm != "sdrpn_3b")
        pairs.append((f"sdrpn_7b_{view}", f"visionrl2_7b_{view}"))
    pairs.extend((f"{arm}_crop", f"{arm}_fusion") for arm in SELECTORS)
    comparisons = [{"reference": a, "target": b, "questions": len(cases),
                    **paired_bootstrap(scores[a], scores[b], groups, 10000, seed)} for a, b in pairs]
    common = [i for i, (_, qid) in enumerate(cases) if qid in polygons and
              all("answer_region_retained" in by_arm[f"{arm}_fusion"][i] for arm in SELECTORS)]
    coverage = {arm: sum(by_arm[f"{arm}_fusion"][i]["answer_region_retained"] >= 0.90 for i in common)
                for arm in SELECTORS}
    write_table(output / "accuracy.csv", tables)
    write_table(output / "paired.csv", comparisons)
    return {"split": manifest["split"], "condition": "LR4", "questions": len(cases), "images": len(set(groups)),
            "arms": tables, "comparisons": comparisons, "identity": identity, "bootstrap_resamples": 10000,
            "bootstrap_unit": "image_id", "scorer_revision": SCORER_REVISION, "training_overlap": audit,
            "grounding": {"policy": "provisional unique TextOCR literal with at least 8/10 votes",
                          "common_grounded_crops": len(common), "common_retains_90pct": coverage},
            "vicrop_calibration": calibration}


def run(output):
    with FileLock(str(output / ".run.lock"), timeout=0):
        launch, paths, config, manifest, checkpoints, binding = load_study(output, "crop", ("A2",))
        source = Path(launch["source"])
        done, original = (read_json(source / name) for name in ("COMPLETE.json", "run.json"))
        if done["stage"] != "query_crop" or done["binding"] != object_hash(original):
            raise ValueError("SD-RPN 3B proposal source is not a verified run")
        repo = Path(launch["repo"])
        textocr_val = validation_ocr(repo)
        polygons = answer_polygons(manifest, textocr_val)
        if not polygons:
            raise ValueError("TextOCR validation has no grounded questions matching the manifest")
        print(f"TextOCR validation grounding: {len(polygons)}/5000 questions", flush=True)
        vendor, weights, training = prepare(repo / "local/visionrl2")
        audit = overlap_audit(manifest, training)
        write_json(output / "training_overlap.json", audit)
        binding.update({"sdrpn_3b_source_binding": done["binding"], "models": {arm: list(x) for arm, x in MODELS.items()},
                        "visionrl2_commit": UPSTREAM_REVISION, "vicrop_reference_commit": VICROP_REVISION,
                        "worker_sha256": sha256(Path(__file__)), "selector_visual_token_cap": 288,
                        "textocr_val_sha256": TEXTOCR_VAL_SHA256, "roi_threshold": 0.15, "padding_per_side": PADDING["pad20"], "reader_tokens": TOTAL_TOKENS})
        bind(output, binding)
        run_7b(launch, paths, config, manifest, output, vendor, weights)
        model = FusionQwen(config, paths["hf_hub"])
        torch.backends.cuda.matmul.allow_tf32 = False
        trained = checkpoints["A2"]
        lora = make_adapter(model, "A2", trained["config"]).eval()
        lora.load_state_dict(trained["state"])
        scorer = evaluator()
        selector = ViCrop(model)
        datasets, _ = read_s1_manifests(Path(launch["data_dir"]), paths["textvqa"])
        dev = datasets["val_inner"]
        layer, calibration = calibrate(selector, dev, answer_polygons(dev, paths["textocr"]),
                                       paths["textvqa"], launch["seed"], pad)
        write_json(output / "vicrop_calibration.json", calibration)
        rows, identity = [], None
        for index, record in enumerate(manifest["images"], 1):
            shard = output / "shards" / f"{record['image_id']}.json"
            if shard.is_file():
                batch = read_json(shard)
            else:
                batch = predict(model, lora, scorer, selector, layer, record, paths, source, output, identity is None)
                write_json(shard, batch)
            rows.extend(batch["rows"])
            identity = identity or batch["identity"]
            if index % 50 == 0 or index == len(manifest["images"]):
                print(f"Crop benchmark: {index}/{len(manifest['images'])} images", flush=True)
        finish(output, binding, report(output, rows, manifest, polygons, launch["seed"], identity, audit, calibration),
               rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
