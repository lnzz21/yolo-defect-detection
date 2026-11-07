"""Validate local YOLO detection data and keep content-addressed training snapshots."""

import hashlib
import json
import math
from pathlib import Path

import yaml

from workstation.store import staging_directory, write_json


IMAGE_TYPES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def dataset_yaml(source):
    if not str(source).strip():
        raise ValueError("请选择 YOLO 数据集目录或 YAML 文件。")
    path = Path(source).resolve()
    if path.is_dir():
        choices = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".yaml", ".yml"})
        if len(choices) != 1:
            raise ValueError("数据集目录需要恰好一个 YAML。请直接选择要用的配置文件。")
        return choices[0]
    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
        return path
    raise ValueError("请选择本地 YOLO 数据集目录或 YAML 文件。")


def dataset_names(config):
    names = config.get("names")
    if isinstance(names, dict):
        try:
            names = [names[index] if index in names else names[str(index)] for index in range(len(names))]
        except KeyError as error:
            raise ValueError("类别编号必须从 0 开始并连续。") from error
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("YAML 需要列出类别 names。")
    if len(set(names)) != len(names) or config.get("nc", len(names)) != len(names):
        raise ValueError("类别名须唯一，nc 应与类别数一致。")
    return names


def image_files(root, entry):
    sources = [entry] if isinstance(entry, str) else entry
    if not isinstance(sources, list) or any(not isinstance(source, str) for source in sources):
        raise ValueError("数据划分应指定本地图片目录或 TXT 图片清单。")
    images = []
    for source in sources:
        path = (root / source).resolve()
        if path.is_dir():
            images.extend(sorted(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_TYPES))
        elif path.is_file() and path.suffix.lower() == ".txt":
            images.extend((path.parent / line.strip()).resolve()
                          for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
        else:
            raise ValueError(f"找不到图片目录或清单 {path}")
    if not images:
        raise ValueError("数据划分没有图片。")
    if len(images) != len(set(images)):
        raise ValueError("同一划分中包含重复图片路径。")
    return images


def label_path(image):
    parts = list(image.parts)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def checked_label(path, nc):
    if not path.is_file():
        raise ValueError(f"缺少标签 {path}。确认无缺陷的图片请使用空 TXT。")
    label = path.read_text(encoding="utf-8-sig").strip()
    for row, line in enumerate(label.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        try:
            cls = int(fields[0])
            x, y, w, h = map(float, fields[1:])
            if not (0 <= cls < nc and all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h))
                    and w > 0 and h > 0 and x - w / 2 >= -1e-6 and y - h / 2 >= -1e-6
                    and x + w / 2 <= 1.000001 and y + h / 2 <= 1.000001):
                raise ValueError()
        except ValueError as error:
            raise ValueError(f"{path} 第 {row} 行需要有效的类别编号和四个归一化检测框数值。") from error
    return label + "\n" if label else ""


def snapshot_dataset(source, versions, log):
    config_path = dataset_yaml(source)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("无法读取 YOLO 数据集配置。")
    names = dataset_names(config)
    root = (config_path.parent / (config.get("path") or "")).resolve()
    for split in ("train", "val"):
        if not config.get(split):
            raise ValueError(f"YAML 缺少 {split} 划分。")
    counts, seen, entries, origins = {}, {}, [], {}
    local = {"names": names}
    versions = versions.resolve()
    with staging_directory(versions, ".snapshot-") as staging:
        for split in ("train", "val", "test"):
            if not config.get(split):
                continue
            images = image_files(root, config[split])
            counts[split] = len(images)
            local[split] = f"images/{split}"
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
            positive_count = 0
            for index, image in enumerate(images, 1):
                if not image.is_file() or image.suffix.lower() not in IMAGE_TYPES:
                    raise ValueError(f"清单中的图片无效 {image}")
                label = checked_label(label_path(image), len(names))
                positive_count += bool(label)
                content = image.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                previous = seen.setdefault(digest, split)
                if previous != split:
                    raise ValueError(f"同一图像内容出现在 {previous} 和 {split} 中 {image}")
                name = f"{index:06d}_{digest[:16]}"
                filename = name + image.suffix.lower()
                (staging / "images" / split / filename).write_bytes(content)
                (staging / "labels" / split / f"{name}.txt").write_bytes(label.encode("utf-8"))
                entries.append([split, filename, digest, hashlib.sha256(label.encode()).hexdigest()])
                origins[f"{split}/{filename}"] = str(image)
            if split in {"train", "val"} and positive_count == 0:
                raise ValueError(f"{split} 划分至少需要一张有标注框的图片。")
            log(f"已检查 {split} 的 {len(images)} 张图像，其中 {positive_count} 张有缺陷框。")

        version = hashlib.sha256(json.dumps([names, entries], ensure_ascii=False).encode()).hexdigest()
        target = versions / version
        if not target.is_dir():
            (staging / "data.yaml").write_text(yaml.safe_dump(local, allow_unicode=True, sort_keys=False), encoding="utf-8")
            write_json(staging / "manifest.json", {"id": version, "images": entries, "classes": names,
                                                    "source_images": origins})
            if staging.resolve().parent != versions or target.resolve().parent != versions:
                raise RuntimeError("数据集版本必须位于工作台版本目录内。")
            try:
                staging.rename(target)
            except OSError:
                if not (target / "manifest.json").is_file():
                    raise
    return {"id": version, "yaml": str(target / "data.yaml"), "source": str(config_path),
            "classes": names, "counts": counts}
