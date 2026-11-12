"""Collect postprocessed detections for one inference task."""

from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from workstation.store import now


def report_source(source):
    if isinstance(source, list):
        return [report_source(path) for path in source]
    source = str(source)
    if "://" in source:
        try:
            url = urlsplit(source)
        except ValueError:
            return "network source"
        return f"{url.scheme}://{url.hostname or ''}"
    return source


def count_classes(result):
    detections = result.obb if result.obb is not None else result.boxes
    if detections is None:
        return {}
    return dict(Counter(result.names[int(index)] for index in detections.cls.tolist()))


class InferenceReport:
    def __init__(self, run_id, source, model):
        self.run_id = run_id
        self.source = report_source(source)
        self.model = Path(model).name
        self.started_at = now()
        self.files = {}

    def add(self, path, counts):
        row = self.files.setdefault(str(path), {"frames": 0, "counts": Counter()})
        row["frames"] += 1
        row["counts"].update(counts)

    def finish(self, state, error=None):
        totals = Counter()
        files = []
        for path, row in self.files.items():
            totals.update(row["counts"])
            files.append({"path": path, "frames": row["frames"], "counts": dict(row["counts"])})
        return {
            "id": self.run_id, "source": self.source, "model": self.model,
            "started_at": self.started_at, "ended_at": now(), "state": state,
            "files": files, "counts": dict(totals), "total_boxes": sum(totals.values()),
            "processed_frames": sum(row["frames"] for row in files), "error": error,
        }
