"""Convert the project's NEU-DET XML annotations to a reproducible YOLO dataset."""

import argparse
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "NEU-DET-v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}. Use --output with a new directory.")

    # Match by stem: some original XML filename fields omit the .jpg extension.
    images = {}
    for image in sorted((source / "IMAGES").iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        if image.stem in images:
            raise ValueError(f"Duplicate image stem: {image.stem}")
        images[image.stem] = image

    groups = defaultdict(list)
    manifest = {"seed": args.seed, "ratios": [0.8, 0.1, 0.1], "classes": list(CLASSES),
                "source": str(source), "splits": {}, "clipped_boxes": 0}
    annotations = sorted((source / "ANNOTATIONS").glob("*.xml"))
    if not annotations or {p.stem for p in annotations} != set(images):
        raise ValueError("Every image must have exactly one XML file with the same stem.")

    # Validate all inputs before creating output files. Retain every annotated object,
    # including secondary classes that differ from the image's filename prefix.
    for annotation in annotations:
        image = images[annotation.stem]
        xml = ET.parse(annotation).getroot()
        width, height = int(xml.findtext("size/width")), int(xml.findtext("size/height"))
        with Image.open(image) as img:
            if img.size != (width, height):
                raise ValueError(f"Image/XML size mismatch: {image}")
            img.verify()
        primary_class = image.stem.rsplit("_", 1)[0]
        if primary_class not in CLASSES:
            raise ValueError(f"Unknown filename class: {image}")
        labels = []
        for obj in xml.findall("object"):
            name = obj.findtext("name", "").strip()
            if name not in CLASSES:
                raise ValueError(f"Unknown object class {name!r}: {annotation}")
            box = obj.find("bndbox")
            xmin, ymin, xmax, ymax = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
                raise ValueError(f"Non-finite box: {annotation}")
            clipped = (max(0, min(width, xmin)), max(0, min(height, ymin)),
                       max(0, min(width, xmax)), max(0, min(height, ymax)))
            manifest["clipped_boxes"] += clipped != (xmin, ymin, xmax, ymax)
            xmin, ymin, xmax, ymax = clipped
            if xmax <= xmin or ymax <= ymin:
                raise ValueError(f"Empty/reversed box: {annotation}")
            labels.append((CLASSES.index(name), (xmin + xmax) / (2 * width),
                           (ymin + ymax) / (2 * height), (xmax - xmin) / width,
                           (ymax - ymin) / height))
        if not labels:
            raise ValueError(f"No objects found: {annotation}")
        groups[primary_class].append((image, labels))

    splits = {"train": [], "val": [], "test": []}
    rng = random.Random(args.seed)
    for name in CLASSES:
        records = groups[name]
        if len(records) < 10:
            raise ValueError(f"Too few images for an 80/10/10 split: {name}")
        rng.shuffle(records)
        n_train, n_val = int(len(records) * 0.8), int(len(records) * 0.1)
        splits["train"].extend(records[:n_train])
        splits["val"].extend(records[n_train:n_train + n_val])
        splits["test"].extend(records[n_train + n_val:])

    for split, records in splits.items():
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        object_counts = Counter()
        for image, labels in records:
            shutil.copy2(image, image_dir / image.name)
            text = "\n".join(f"{row[0]} " + " ".join(f"{v:.8f}" for v in row[1:]) for row in labels)
            (label_dir / f"{image.stem}.txt").write_text(text + "\n", encoding="utf-8")
            object_counts.update(CLASSES[row[0]] for row in labels)
        manifest["splits"][split] = {"images": [image.name for image, _ in records],
                                      "objects": dict(object_counts)}
        print(f"{split}: {len(records)} images, {sum(object_counts.values())} objects")

    yaml = (f"path: {json.dumps(output.as_posix())}\n"
            "train: images/train\nval: images/val\ntest: images/test\nnames:\n")
    yaml += "".join(f"  {idx}: {name}\n" for idx, name in enumerate(CLASSES))
    (output / "neu_det.yaml").write_text(yaml, encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dataset: {output / 'neu_det.yaml'}")
    print(f"Boxes clipped to image boundaries: {manifest['clipped_boxes']}")


if __name__ == "__main__":
    main()
