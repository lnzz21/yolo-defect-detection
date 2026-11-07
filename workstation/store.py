"""Persistent training history and model selection for one workstation."""

import json
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4


PROJECT = Path(__file__).resolve().parents[1]
STORAGE = PROJECT / "runs" / "workstation"
STEEL_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        except Exception:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def staging_directory(root, prefix):
    """Create a private name while inheriting the workstation folder permissions."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / (prefix + uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.exists():
            resolved = path.resolve()
            if resolved != path or resolved.parent != root:
                raise RuntimeError("拒绝清理工作台暂存目录以外的路径。")
            shutil.rmtree(resolved)


class WorkstationStore:
    def __init__(self, root=STORAGE):
        self.root = Path(root).resolve()
        self.jobs = self.root / "jobs"
        self.versions = self.root / "datasets"
        self.selection_file = self.root / "selection.json"

    def create_job(self, settings):
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        directory = self.jobs / job_id
        directory.mkdir(parents=True)
        write_json(directory / "run.json", {
            "id": job_id, "created_at": now(), "state": "starting", "settings": settings,
        })
        return directory

    def update_job(self, directory, **fields):
        path = directory / "run.json"
        record = read_json(path)
        record.update(fields)
        write_json(path, record)
        return record

    def history(self):
        records = []
        for path in sorted(self.jobs.glob("*/run.json"), reverse=True):
            try:
                records.append((path.parent, read_json(path)))
            except (OSError, ValueError):
                continue
        return records

    def models(self):
        models = {}
        baseline = PROJECT / "ptfiles" / "yolo11n_neu_det.pt"
        if baseline.is_file():
            models["baseline"] = {
                "id": "baseline", "name": "原有 NEU-DET 模型", "weights": baseline,
                "classes": STEEL_CLASSES, "imgsz": 640, "type": "yolo11-detect",
            }
        for directory, record in self.history():
            weights = directory / "training" / "weights" / "best.pt"
            if record.get("state") == "completed" and weights.is_file():
                models[record["id"]] = {
                    "id": record["id"], "name": record["id"], "weights": weights,
                    "classes": record["dataset"]["classes"], "imgsz": record["settings"]["imgsz"],
                    "type": "yolo11-detect",
                }
        return models

    def selection(self):
        if self.selection_file.is_file():
            return read_json(self.selection_file)
        return {"active": "baseline" if "baseline" in self.models() else None, "previous": None}

    def active_model(self):
        return self.models().get(self.selection()["active"])

    def activate(self, model_id):
        if model_id not in self.models():
            raise ValueError("所选模型的权重文件不存在。")
        current = self.selection()
        if model_id != current["active"]:
            write_json(self.selection_file, {"active": model_id, "previous": current["active"]})

    def rollback(self):
        previous = self.selection()["previous"]
        if not previous:
            raise ValueError("当前没有可切回的模型。")
        self.activate(previous)
