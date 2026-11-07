"""Rebuild the locked split, plan common S1 examples and audit baseline parity."""

from .common import object_hash, read_json, read_jsonl, sha256
from .data import PROTOCOL, annotation_file, load_clean, question_groups, rank, read_questions, split_inner
from .experiment import validate_manifest, verified_complete


def read_eval_anchor(source, paths, model_config):
    s0 = verified_complete(source / "s0", "s0")
    d0 = verified_complete(source / "d0", "d0")
    shared = d0["shared"]
    if (
        not read_json(source / "d0" / "identity.json")["passed"]
        or s0["d0_binding"] != object_hash(d0) or s0["shared"] != shared
        or shared["mode"] != "full" or shared["model"] != model_config
        or shared.get("preprocessing") != PROTOCOL
    ):
        raise ValueError("S1 needs a matching full D0/S0 anchor with a passing identity gate")
    manifest_path = source / "manifest.json"
    if sha256(manifest_path) != shared["manifest_sha256"]:
        raise ValueError("The evaluation manifest changed since D0/S0")
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    for split, digest in manifest["annotation_sha256"].items():
        if sha256(annotation_file(paths["textvqa"], split)) != digest:
            raise ValueError(f"Source TextVQA annotations changed for {split}")
    return manifest, object_hash(d0), object_hash(s0)


def build_train_plan(textvqa, evaluation, count, train_seed):
    train = question_groups(read_questions(textvqa, "train"))
    inner, val = split_inner(train, evaluation["seed"])
    if val != evaluation["val_inner_image_ids"] or object_hash(inner) != evaluation["train_inner_ids_hash"]:
        raise ValueError("Rebuilt train/val split differs from the D0/S0 split")
    eval_ids = {row["image_id"] for row in evaluation["images"]}
    if not eval_ids <= set(val) or set(inner) & eval_ids:
        raise ValueError("Train/evaluation image split overlaps")
    if not 1 <= count <= len(inner):
        raise ValueError(f"Requested {count} training images, available {len(inner)}")
    selected = sorted(inner, key=lambda iid: (rank(train_seed, f"s1-train:{iid}"), iid))[:count]
    records = []
    eval_questions = {q["question_id"] for row in evaluation["images"] for q in row["questions"]}
    for iid in selected:
        rel = f"train_images/{iid}.jpg"
        image, orientation, digest = load_clean(textvqa / rel)
        questions = sorted(train[iid], key=lambda q: q["question_id"])
        for q in questions:
            answers = q["answers"]
            if q["question_id"] in eval_questions or len(answers) != 10 or not all(isinstance(a, str) for a in answers):
                raise ValueError(f"Unexpected train answers or overlap: {q['question_id']}")
            if not any(a.strip() for a in answers):
                raise ValueError(f"No nonempty answers: {q['question_id']}")
        records.append({"image_id": iid, "image_rel": rel, "image_sha256": digest,
                        "exif_orientation": orientation, "size": list(image.size), "questions": questions})
    # Detect byte-identical duplicates crossing the fixed evaluation boundary.
    if {r["image_sha256"] for r in records} & {r["image_sha256"] for r in evaluation["images"]}:
        raise ValueError("Train/evaluation images overlap by SHA-256")
    return records


def training_examples(records, epochs, seed, clean_fraction):
    for epoch in range(epochs):
        ordered = sorted(records, key=lambda r: rank(seed, f"order:{epoch}:{r['image_id']}"))
        clean_count = min(len(ordered) - 1, round(len(ordered) * clean_fraction))
        clean_ids = {r["image_id"] for r in sorted(
            ordered, key=lambda r: rank(seed, f"clean:{epoch}:{r['image_id']}")
        )[:clean_count]}
        for image in ordered:
            condition = "CLEAN" if image["image_id"] in clean_ids else "LR4"
            for question in image["questions"]:
                answers = [a.strip() for a in question["answers"] if a.strip()]
                index = int(rank(seed, f"answer:{epoch}:{question['question_id']}")[:16], 16) % len(answers)
                yield image, question, condition, answers[index]


def compare_archived_baselines(source, manifest):
    """Compare the raw predictions from job 10756, instead of inferring from averages."""
    keys = {(image["image_id"], q["question_id"]) for image in manifest["images"] for q in image["questions"]}
    results = {}
    archives = {
        "d0": read_jsonl(source / "d0" / "predictions.jsonl"),
        "s0": read_jsonl(source / "s0" / "oracle_predictions.jsonl"),
    }
    for condition, native, injected in (("LR4", "LR4-BIC-CONTROL", "O0"), ("CLEAN", "CLEAN", "O1")):
        a = {(r["image_id"], r["question_id"]): r for r in archives["d0"] if r["arm"] == native}
        b = {(r["image_id"], r["question_id"]): r for r in archives["s0"] if r["arm"] == injected}
        if a.keys() != keys or b.keys() != keys:
            raise ValueError(f"Archived D0/S0 predictions do not cover the same questions: {condition}")
        diffs = []
        for key in sorted(keys):
            left, right = a[key], b[key]
            if left["token_ids"] != right["token_ids"] or left["answer"] != right["answer"] or left["score"] != right["score"]:
                diffs.append({"image_id": key[0], "question_id": key[1], "native_answer": left["answer"],
                              "injected_answer": right["answer"], "native_score": left["score"],
                              "injected_score": right["score"], "tokens_equal": left["token_ids"] == right["token_ids"]})
        results[condition] = {"questions": len(keys), "differences": diffs}
    return results
