"""Process entry point for a single training task created by the workbench."""

import argparse
import hashlib
import json
import math
import os
import shutil
import traceback
from pathlib import Path

from PySide6.QtCore import QLockFile

from train_neu_det import train_detection
from workstation.dataset import snapshot_dataset
from workstation.store import WorkstationStore, now, read_json


EVENT_PREFIX = "@workstation "


def event(kind, **payload):
    print(EVENT_PREFIX + json.dumps({"kind": kind, **payload}, ensure_ascii=False, allow_nan=False), flush=True)


def configure_worker(root):
    # Set these before importing Ultralytics or matplotlib in this process.
    environment = root / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(environment)
    os.environ["MPLCONFIGDIR"] = str(environment / "matplotlib")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    # The framework checks for these assets even with plots disabled. Reuse a local font.
    from matplotlib import get_data_path

    bundled_font = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    for filename in ("Arial.ttf", "Arial.Unicode.ttf"):
        destination = environment / filename
        if not destination.exists():
            shutil.copyfile(bundled_font, destination)


def train_job(directory, store):
    record = read_json(directory / "run.json")
    settings = record["settings"]
    stop_request = directory / "stop.request"
    store.update_job(directory, state="running", started_at=now())
    event("stage", message="正在检查并冻结训练数据版本…")
    dataset = snapshot_dataset(settings["data"], store.versions, lambda msg: print(msg, flush=True))
    weights = Path(settings["weights"])
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise ValueError("请选择本地 YOLO11 检测权重 .pt。")
    store.update_job(directory, dataset=dataset, initial_weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
    event("dataset", **dataset)
    if stop_request.exists():
        store.update_job(directory, state="stopped", ended_at=now())
        return

    configure_worker(store.root)
    import torch
    from ultralytics import YOLO, __version__, settings as integrations

    integrations.update({name: False for name in (
        "sync", "hub", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune",
        "tensorboard", "wandb", "vscode_msg",
    ) if name in integrations})
    device = settings["device"]
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。可在训练页改选 CPU。")
    model = YOLO(str(weights))
    if model.task != "detect" or "C2PSA" not in {type(layer).__name__ for layer in model.model.model}:
        raise ValueError("本工作台需要 YOLO11 水平框检测模型。")
    store.update_job(directory, environment={"ultralytics": __version__, "torch": torch.__version__})
    last_epoch = 0

    def record_epoch(trainer):
        nonlocal last_epoch
        epoch = trainer.epoch + 1
        if epoch <= last_epoch:
            return  # Ultralytics also calls this hook when evaluating best.pt at the end.
        last_epoch = epoch
        raw = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics}
        metrics = {key: float(value) for key, value in raw.items() if math.isfinite(float(value))}
        store.update_job(directory, last_epoch=epoch, metrics=metrics)
        event("epoch", epoch=epoch, epochs=trainer.epochs, metrics=metrics)
        if stop_request.exists() and epoch < trainer.epochs:
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", record_epoch)
    event("stage", message="训练中，每轮验证完成后更新曲线。")
    train_detection(
        model,
        dict(data=dataset["yaml"], imgsz=settings["imgsz"], batch=settings["batch"],
             device=device, workers=0, project=str(directory)),
        epochs=settings["epochs"], name="training", amp=False, plots=False,
    )
    if stop_request.exists() and last_epoch < settings["epochs"]:
        store.update_job(directory, state="stopped", ended_at=now())
        return
    if last_epoch == 0 or not (directory / "training" / "weights" / "best.pt").is_file():
        raise RuntimeError("训练结束时没有找到已验证的 best.pt。")
    best_metrics = {key: float(value) for key, value in model.trainer.metrics.items() if math.isfinite(float(value))}
    store.update_job(directory, state="completed", ended_at=now(),
                     best_weights="training/weights/best.pt", last_weights="training/weights/last.pt",
                     results_csv="training/results.csv", best_metrics=best_metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    directory = parser.parse_args().job.resolve()
    store = WorkstationStore(directory.parent.parent)
    lock = QLockFile(str(store.root / "training.lock"))
    lock.setStaleLockTime(0)
    acquired = False
    try:
        acquired = lock.tryLock(0)
        if not acquired:
            raise RuntimeError("该工作台已有训练任务在运行。")
        for other, record in store.history():
            if other != directory and record.get("state") == "running":
                store.update_job(other, state="failed", error="上次训练进程已退出。", ended_at=now())
        train_job(directory, store)
    except Exception as error:
        store.update_job(directory, state="failed", error=str(error), ended_at=now())
        traceback.print_exc()
        return 1
    finally:
        if acquired:
            lock.unlock()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
