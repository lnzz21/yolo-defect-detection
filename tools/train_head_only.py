"""Compare Detect-only fine-tuning with the completed full fine-tuning baseline."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "head-only-ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def frozen_digest(model, detect_index):
    """Include parameters AND BatchNorm buffers of the frozen feature extractor."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith("model.") and int(name.split(".")[1]) < detect_index:
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "runs/detect/neu_yolo11n")
    parser.add_argument("--name", default="neu_yolo11n_head_only")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO, settings
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.nn.modules import Detect

    # Keep this experiment's integrations local to its isolated settings directory.
    settings.update({key: False for key in (
        "sync", "hub", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune",
        "tensorboard", "wandb", "vscode_msg"
    )})
    baseline = args.baseline.resolve()
    baseline_config = yaml.safe_load((baseline / "args.yaml").read_text(encoding="utf-8"))
    pretrained = Path(baseline_config["model"])
    data_file = Path(baseline_config["data"])
    original_best = baseline / "weights/best.pt"
    for path in (pretrained, data_file, original_best, baseline / "results.csv"):
        if not path.is_file():
            parser.error(f"Required input is missing: {path}")
    if not torch.cuda.is_available():
        parser.error("This comparison requires the same CUDA training device as the baseline.")

    model = YOLO(str(pretrained))
    detect_index = len(model.model.model) - 1
    if not isinstance(model.model.model[-1], Detect) or detect_index != 23:
        parser.error("Expected YOLO11n with a single Detect module at index 23.")
    run_args = {key: value for key, value in baseline_config.items() if key in DEFAULT_CFG_DICT}
    run_args.update(freeze=detect_index, name=args.name, exist_ok=False, resume=False)
    run_dir = Path(run_args["project"]) / args.name
    audit = {
        "baseline": str(baseline), "initial_weights": str(pretrained),
        "initial_weights_sha256": hashlib.sha256(pretrained.read_bytes()).hexdigest(),
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "detect_index": detect_index,
        "freeze": detect_index,
        "changed_arguments": {key: {"baseline": baseline_config.get(key), "head_only": value}
                              for key, value in run_args.items() if value != baseline_config.get(key)},
        "training_arguments": run_args,
    }
    print(json.dumps(audit, ensure_ascii=True), flush=True)
    if args.check:
        return

    if not args.evaluate_only:
        if run_dir.exists():
            parser.error(f"Output exists: {run_dir}. Choose a new --name or use --evaluate-only.")
        state = {"last_epoch": 0, "started": time.time()}

        def verify_start(trainer):
            actual_trainable = [name for name, parameter in trainer.model.named_parameters()
                                if parameter.requires_grad]
            if not actual_trainable or any(not name.startswith(f"model.{detect_index}.")
                                           for name in actual_trainable):
                raise RuntimeError("Freeze verification failed: trainable tensors outside Detect.")
            state["frozen_sha256"] = frozen_digest(trainer.model, detect_index)
            audit.update(
                trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                frozen_parameters=sum(p.numel() for p in trainer.model.parameters() if not p.requires_grad),
                trainable_tensor_names=actual_trainable,
                frozen_initial_sha256=state["frozen_sha256"],
            )
            write_json(trainer.save_dir / "freezing_audit.json", audit)
            print("FREEZE_VERIFIED " + json.dumps({key: audit[key] for key in
                  ("detect_index", "trainable_parameters", "frozen_parameters")}), flush=True)

        def record_epoch(trainer):
            epoch = trainer.epoch + 1
            if epoch == state["last_epoch"]:
                return
            if frozen_digest(trainer.model, detect_index) != state["frozen_sha256"]:
                raise RuntimeError("Frozen feature-extractor parameters or BatchNorm buffers changed.")
            state["last_epoch"] = epoch
            progress = {"epoch": epoch, "epochs": trainer.epochs,
                        "elapsed_seconds": round(time.time() - state["started"], 1),
                        "frozen_state_unchanged": True,
                        "metrics": {key: float(value) for key, value in trainer.metrics.items()},
                        "completed": False}
            write_json(trainer.save_dir / "progress.json", progress)
            print("EPOCH_RESULT " + json.dumps(progress), flush=True)

        model.add_callback("on_train_start", verify_start)
        model.add_callback("on_fit_epoch_end", record_epoch)
        model.train(**run_args)
        run_dir = Path(model.trainer.save_dir)
        audit["frozen_final_sha256"] = frozen_digest(model.trainer.model, detect_index)
        audit["frozen_state_unchanged"] = audit["frozen_final_sha256"] == audit["frozen_initial_sha256"]
        if not audit["frozen_state_unchanged"]:
            raise RuntimeError("Final frozen-state verification failed.")
        write_json(run_dir / "freezing_audit.json", audit)
    else:
        audit = json.loads((run_dir / "freezing_audit.json").read_text(encoding="utf-8"))

    # Compare the two selected checkpoints with exactly the same validation settings.
    evaluation = dict(data=str(data_file), split="val", imgsz=baseline_config["imgsz"],
                      batch=baseline_config["batch"], device=baseline_config["device"],
                      workers=baseline_config["workers"], seed=baseline_config["seed"],
                      conf=0.001, iou=0.7, max_det=300, half=False,
                      plots=False, verbose=False, project=str(run_dir / "evaluation"))
    rows = []
    for label, weights in (("full_fine_tuning", original_best),
                           ("detect_only", run_dir / "weights/best.pt")):
        checkpoint = YOLO(str(weights))
        metrics = checkpoint.val(**evaluation, name=label, exist_ok=False)
        values = {"method": label, "weights": str(weights), "precision": float(metrics.box.mp),
                  "recall": float(metrics.box.mr), "mAP50": float(metrics.box.map50),
                  "mAP50_95": float(metrics.box.map)}
        values["per_class"] = {
            metrics.names[int(class_id)]: dict(zip(("precision", "recall", "mAP50", "mAP50_95"),
                (float(v) for v in metrics.box.class_result(index))))
            for index, class_id in enumerate(metrics.box.ap_class_index)
        }
        rows.append(values)
        print("VALIDATION_RESULT " + json.dumps(values), flush=True)

    with (run_dir / "results.csv").open(newline="", encoding="utf-8-sig") as file:
        completed_epochs = len(list(csv.DictReader(file)))
    comparison = {"split": "val", "single_seed": True, "completed_epochs": completed_epochs,
                  "evaluation_settings": evaluation, "freezing": audit, "results": rows,
                  "head_minus_full_percentage_points": {
                      key: (rows[1][key] - rows[0][key]) * 100
                      for key in ("precision", "recall", "mAP50", "mAP50_95")}}
    write_json(run_dir / "comparison.json", comparison)
    markdown = ["# YOLO11n 检测头微调对比", "", "同一初始权重、数据划分、随机种子和训练设置；只改变冻结范围。",
                "", "| 方案 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        name = "全参数微调" if row["method"] == "full_fine_tuning" else "只微调 Detect 检测头"
        markdown.append("| " + name + " | " + " | ".join(
            f"{row[key] * 100:.2f}%" for key in ("precision", "recall", "mAP50", "mAP50_95")) + " |")
    markdown += ["", f"检测头方案实际完成 {completed_epochs} 轮，最多 100 轮，沿用 patience=30。",
                 "冻结第 0～22 层，仅训练第 23 层 Detect 的可训练参数；框架固定的 DFL 投影不训练。",
                 f"冻结层参数和 BatchNorm 缓冲区保持不变：{audit.get('frozen_state_unchanged')}。",
                 "", "指标来自重新评估的同一验证集，尚未使用保留测试集；单个随机种子的结果不代表普遍结论。"]
    (run_dir / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_json(run_dir / "progress.json", {"completed": True, "epoch": completed_epochs,
                                          "comparison": str(run_dir / "comparison.json")})
    print("COMPARISON_SAVED " + str(run_dir / "comparison.md"), flush=True)


if __name__ == "__main__":
    main()
