"""TextVQA manifests for S1 with a fixed image-level split and no OCR dependency."""

from itertools import combinations

from .common import object_hash, read_json, sha256, write_json
from .data import PROTOCOL, annotation_file, load_clean, question_groups, read_questions, split_inner, verify_pillow


S1_SPLITS = ("train_inner", "val_inner", "official_val")


def validate_s1_manifest(manifest):
    if (
        manifest["schema"] != 2 or manifest["protocol"] != PROTOCOL
        or manifest["split"] not in S1_SPLITS or not manifest["images"]
    ):
        raise ValueError("Expected a nonempty TextVQA S1 manifest")
    seen_images, seen_questions = set(), set()
    for row in manifest["images"]:
        image_id = row["image_id"]
        if (
            image_id in seen_images or row["image_rel"] != f"train_images/{image_id}.jpg"
            or "/" in image_id or "\\" in image_id or not row["questions"]
        ):
            raise ValueError(f"Invalid or duplicate manifest image: {image_id}")
        seen_images.add(image_id)
        for q in row["questions"]:
            answers = q["answers"]
            if (
                q["question_id"] in seen_questions or len(answers) != 10
                or not all(isinstance(answer, str) for answer in answers)
                or not any(answer.strip() for answer in answers)
            ):
                raise ValueError(f"Invalid or duplicate manifest question: {q['question_id']}")
            seen_questions.add(q["question_id"])


def check_disjoint(manifests):
    def keys(manifest):
        images = manifest["images"]
        return (
            {r["image_id"] for r in images}, {r["image_sha256"] for r in images},
            {q["question_id"] for r in images for q in r["questions"]},
        )

    for left, right in combinations(manifests, 2):
        for name, a, b in zip(("image IDs", "image SHA-256", "question IDs"), keys(left), keys(right)):
            overlap = a & b
            if overlap:
                raise ValueError(f"{left['split']} and {right['split']} overlap by {name}: {len(overlap)}")


def build_s1_manifests(textvqa, output, seed):
    verify_pillow()
    if (output / "COMPLETE.json").exists():
        manifests, _ = read_s1_manifests(output, textvqa)
        if any(m["seed"] != seed for m in manifests.values()):
            raise ValueError("Dataset seed changed")
        return
    train = question_groups(read_questions(textvqa, "train"))
    official_val = question_groups(read_questions(textvqa, "val"))
    inner, dev = split_inner(train, seed)
    annotations = {split: sha256(annotation_file(textvqa, split)) for split in ("train", "val")}
    manifests = {}
    for split, groups, ids in (
        ("train_inner", train, inner), ("val_inner", train, dev), ("official_val", official_val, sorted(official_val)),
    ):
        records = []
        for image_id in ids:
            rel = f"train_images/{image_id}.jpg"
            image, orientation, digest = load_clean(textvqa / rel)
            records.append({
                "image_id": image_id, "image_rel": rel, "image_sha256": digest,
                "size": list(image.size), "exif_orientation": orientation,
                "questions": sorted(groups[image_id], key=lambda q: q["question_id"]),
            })
        manifest = {"schema": 2, "split": split, "protocol": PROTOCOL, "seed": seed,
                    "annotation_sha256": annotations, "images": records}
        validate_s1_manifest(manifest)
        manifests[split] = manifest
        print(f"{split}: {len(records)} images, {sum(len(r['questions']) for r in records)} questions", flush=True)
    check_disjoint(manifests.values())
    output.mkdir(parents=True, exist_ok=True)
    for split, manifest in manifests.items():
        path = output / f"{split}.json"
        if path.exists() and read_json(path) != manifest:
            raise ValueError(f"An incomplete dataset manifest differs for {split}")
        write_json(path, manifest)
    write_json(output / "COMPLETE.json", {"manifests": {s: object_hash(m) for s, m in manifests.items()}})


def read_s1_manifests(output, textvqa):
    expected = read_json(output / "COMPLETE.json")["manifests"]
    manifests = {}
    annotations = {split: sha256(annotation_file(textvqa, split)) for split in ("train", "val")}
    for split in S1_SPLITS:
        manifest = read_json(output / f"{split}.json")
        validate_s1_manifest(manifest)
        if manifest["split"] != split or expected[split] != object_hash(manifest):
            raise ValueError(f"Dataset manifest changed for {split}")
        if manifest["annotation_sha256"] != annotations:
            raise ValueError(f"Source annotations changed for {split}")
        manifests[split] = manifest
    check_disjoint(manifests.values())
    return manifests, expected
