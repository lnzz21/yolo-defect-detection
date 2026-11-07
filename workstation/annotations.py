"""Reviewed captures and export of a versioned YOLO dataset."""

from pathlib import Path
from uuid import uuid4

import cv2
import yaml

from workstation.dataset import dataset_names, dataset_yaml, image_files, snapshot_dataset
from workstation.store import STEEL_CLASSES, now, read_json, staging_directory, write_json


class Annotations:
    def __init__(self, store):
        self.store = store
        self.root = store.root / "annotations"

    def image(self, item_id):
        return self.root / "images" / f"{item_id}.png"

    def capture(self, frame, metadata):
        item_id = uuid4().hex
        image = self.image(item_id)
        image.parent.mkdir(parents=True, exist_ok=True)
        ok, png = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("无法保存原始视频帧。")
        image.write_bytes(png.tobytes())
        record = {"id": item_id, "source": metadata["source"], "frame": metadata["frame"],
                  "position_ms": metadata["position_ms"], "captured_at": metadata["captured_at"],
                  "group": metadata.get("group", metadata["source"]), "width": int(frame.shape[1]),
                  "height": int(frame.shape[0]), "state": "pending", "boxes": []}
        write_json(self.root / "items" / f"{item_id}.json", record)
        return record

    def items(self):
        records = [read_json(path) for path in (self.root / "items").glob("*.json")]
        return sorted(records, key=lambda item: (item["captured_at"], item["frame"], item["id"]), reverse=True)

    def save(self, item_id, boxes, state):
        path = self.root / "items" / f"{item_id}.json"
        record = read_json(path)
        width, height = record["width"], record["height"]
        rows = []
        for box in boxes:
            cls = box["class_id"]
            x1, y1, x2, y2 = box["xyxy"]
            if not (0 <= cls < len(STEEL_CLASSES) and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("标注框的类别或范围无效。")
            rows.append(f"{cls} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} "
                        f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
        if state == "reviewed" and not rows:
            raise ValueError("请先画框，或使用确认无缺陷。")
        if state == "negative" and rows:
            raise ValueError("画面仍有标注框。请核对后再确认无缺陷。")
        record.update(boxes=boxes, state=state, reviewed_at=None if state == "pending" else now())
        if state != "pending":
            label = self.root / "labels" / f"{item_id}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            temporary = label.with_suffix(".tmp")
            temporary.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            temporary.replace(label)
        write_json(path, record)
        return record

    def export(self, base, log):
        reviewed = [item for item in self.items() if item["state"] in {"reviewed", "negative"}]
        if not reviewed:
            raise ValueError("请先审核至少一个抓帧样本。")
        if base:
            yaml_path = dataset_yaml(base)
            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
            if dataset_names(config) != STEEL_CLASSES:
                raise ValueError("基础数据集的类别名称和顺序需与钢材标注一致。")
            source = (yaml_path.parent / (config.get("path") or "")).resolve()
            splits = {split: image_files(source, config[split]) for split in ("train", "val", "test") if config.get(split)}
            splits.setdefault("train", []).extend(self.image(item["id"]) for item in reviewed)
        else:
            positives = {item["group"] for item in reviewed if item["state"] == "reviewed"}
            if len(positives) < 2:
                raise ValueError("独立划分需要至少两个含缺陷标注的不同来源。也可选择基础数据集来保留已有验证集。")
            val_group = sorted(positives)[-1]
            splits = {"train": [], "val": []}
            for item in reviewed:
                split = "val" if item["group"] == val_group else "train"
                splits[split].append(self.image(item["id"]))
        with staging_directory(self.store.root, ".export-") as root:
            config = {"names": STEEL_CLASSES}
            for split, images in splits.items():
                manifest = root / f"{split}.txt"
                manifest.write_text("\n".join(path.as_posix() for path in images) + "\n", encoding="utf-8")
                config[split] = manifest.name
            config_path = root / "data.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            return snapshot_dataset(config_path, self.store.versions, log)
