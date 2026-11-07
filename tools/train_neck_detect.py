"""Freeze YOLO11n Backbone and compare Neck + Detect with existing baselines."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "neck-detect-ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")
FREEZE_COUNT = 11  # Backbone modules 0-10; train Neck 11-22 and Detect 23.


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def frozen_digest(model):
    """Audit both Backbone parameters and BatchNorm running statistics."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith("model.") and int(name.split(".")[1]) < FREEZE_COUNT:
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "runs/detect/neu_yolo11n")
    parser.add_argument("--head-only", type=Path, default=ROOT / "runs/detect/neu_yolo11n_head_only")
    parser.add_argument("--name", default="neu_yolo11n_neck_detect")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO, settings
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.nn.modules import Detect

    settings.update({key: False for key in (
        "sync", "hub", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune",
        "tensorboard", "wandb", "vscode_msg"
    )})
    baseline = args.baseline.resolve()
    head_only = args.head_only.resolve()
    config = yaml.safe_load((baseline / "args.yaml").read_text(encoding="utf-8"))
    pretrained = Path(config["model"])
    data_file = Path(config["data"])
    checkpoints = [
        ("full_fine_tuning", baseline / "weights/best.pt"),
        ("detect_only", head_only / "weights/best.pt"),
    ]
    inputs = [pretrained, data_file, baseline / "results.csv",
              head_only / "freezing_audit.json", *[path for _, path in checkpoints]]
    for path in inputs:
        if not path.is_file():
            parser.error(f"Required input is missing: {path}")
    if not torch.cuda.is_available():
        parser.error("This comparison requires the same CUDA device as the baselines.")

    initial_sha256 = hashlib.sha256(pretrained.read_bytes()).hexdigest()
    previous_audit = json.loads((head_only / "freezing_audit.json").read_text(encoding="utf-8"))
    if initial_sha256 != previous_audit["initial_weights_sha256"]:
        parser.error("Initial pretrained weights differ from the Detect-only experiment.")
    model = YOLO(str(pretrained))
    if len(model.model.model) != 24 or not isinstance(model.model.model[23], Detect):
        parser.error("Expected YOLO11n with Detect at index 23.")
    if model.model.model[10].__class__.__name__ != "C2PSA":
        parser.error("Unexpected Backbone boundary; expected C2PSA at module 10.")

    run_args = {key: value for key, value in config.items() if key in DEFAULT_CFG_DICT}
    run_args.update(freeze=FREEZE_COUNT, name=args.name, exist_ok=False, resume=False)
    run_dir = Path(run_args["project"]) / args.name
    audit = {
        "baseline": str(baseline), "head_only_baseline": str(head_only),
        "initial_weights": str(pretrained), "initial_weights_sha256": initial_sha256,
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "freeze": FREEZE_COUNT,
        "frozen_module_indices": list(range(FREEZE_COUNT)),
        "trainable_module_indices": list(range(FREEZE_COUNT, 24)),
        "changed_arguments": {key: {"baseline": config.get(key), "neck_detect": value}
                              for key, value in run_args.items() if value != config.get(key)},
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
            trainable_names = []
            module_parameters = {"backbone": 0, "neck": 0, "detect": 0}
            for name, parameter in trainer.model.named_parameters():
                index = int(name.split(".")[1])
                expected = index >= FREEZE_COUNT and ".dfl" not in name
                if parameter.requires_grad != expected:
                    raise RuntimeError(f"Unexpected trainable state for {name}")
                if parameter.requires_grad:
                    trainable_names.append(name)
                    part = "neck" if index < 23 else "detect"
                    module_parameters[part] += parameter.numel()
            if not module_parameters["neck"] or not module_parameters["detect"]:
                raise RuntimeError("Both Neck and Detect must have trainable parameters.")
            state["frozen_sha256"] = frozen_digest(trainer.model)
            audit.update(
                trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                frozen_parameters=sum(p.numel() for p in trainer.model.parameters() if not p.requires_grad),
                trainable_parameters_by_part=module_parameters,
                trainable_tensor_names=trainable_names,
                frozen_initial_sha256=state["frozen_sha256"],
            )
            write_json(trainer.save_dir / "freezing_audit.json", audit)
            print("FREEZE_VERIFIED " + json.dumps({key: audit[key] for key in (
                "freeze", "trainable_parameters", "frozen_parameters", "trainable_parameters_by_part"
            )}), flush=True)

        def record_epoch(trainer):
            epoch = trainer.epoch + 1
            if epoch == state["last_epoch"]:
                return
            if frozen_digest(trainer.model) != state["frozen_sha256"]:
                raise RuntimeError("Frozen Backbone parameters or BatchNorm buffers changed.")
            state["last_epoch"] = epoch
            progress = {
                "epoch": epoch, "epochs": trainer.epochs,
                "elapsed_seconds": round(time.time() - state["started"], 1),
                "frozen_state_unchanged": True,
                "metrics": {key: float(value) for key, value in trainer.metrics.items()},
                "completed": False,
            }
            write_json(trainer.save_dir / "progress.json", progress)
            print("EPOCH_RESULT " + json.dumps(progress), flush=True)

        model.add_callback("on_train_start", verify_start)
        model.add_callback("on_fit_epoch_end", record_epoch)
        model.train(**run_args)
        run_dir = Path(model.trainer.save_dir)
        audit["frozen_final_sha256"] = frozen_digest(model.trainer.model)
        audit["frozen_state_unchanged"] = audit["frozen_final_sha256"] == audit["frozen_initial_sha256"]
        if not audit["frozen_state_unchanged"]:
            raise RuntimeError("Final Backbone freeze verification failed.")
        write_json(run_dir / "freezing_audit.json", audit)
    else:
        audit = json.loads((run_dir / "freezing_audit.json").read_text(encoding="utf-8"))

    evaluation = dict(
        data=str(data_file), split="val", imgsz=config["imgsz"], batch=config["batch"],
        device=config["device"], workers=config["workers"], seed=config["seed"],
        conf=0.001, iou=0.7, max_det=300, half=False, plots=False, verbose=False,
        project=str(run_dir / "evaluation"),
    )
    rows = []
    for label, weights in [*checkpoints, ("neck_detect", run_dir / "weights/best.pt")]:
        checkpoint = YOLO(str(weights))
        metrics = checkpoint.val(**evaluation, name=label, exist_ok=False)
        values = {
            "method": label, "weights": str(weights), "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr), "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        }
        values["per_class"] = {
            metrics.names[int(class_id)]: dict(zip(
                ("precision", "recall", "mAP50", "mAP50_95"),
                (float(v) for v in metrics.box.class_result(index)),
            )) for index, class_id in enumerate(metrics.box.ap_class_index)
        }
        rows.append(values)
        print("VALIDATION_RESULT " + json.dumps(values), flush=True)

    with (run_dir / "results.csv").open(newline="", encoding="utf-8-sig") as file:
        completed_epochs = len(list(csv.DictReader(file)))
    comparison = {
        "split": "val", "single_seed": True, "completed_epochs": completed_epochs,
        "evaluation_settings": evaluation, "freezing": audit, "results": rows,
        "neck_detect_minus_full_percentage_points": {
            key: (rows[2][key] - rows[0][key]) * 100
            for key in ("precision", "recall", "mAP50", "mAP50_95")
        },
        "neck_detect_minus_detect_only_percentage_points": {
            key: (rows[2][key] - rows[1][key]) * 100
            for key in ("precision", "recall", "mAP50", "mAP50_95")
        },
    }
    write_json(run_dir / "comparison.json", comparison)
    names = {
        "full_fine_tuning": "全参数微调", "detect_only": "只训练 Detect",
        "neck_detect": "训练 Neck + Detect",
    }
    markdown = [
        "# YOLO11n 三种微调方案对比", "",
        "同一初始权重、数据划分、随机种子和训练设置；只改变冻结范围。", "",
        "| 方案 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append("| " + names[row["method"]] + " | " + " | ".join(
            f"{row[key] * 100:.2f}%" for key in ("precision", "recall", "mAP50", "mAP50_95")
        ) + " |")
    markdown += [
        "", f"Neck + Detect 方案实际完成 {completed_epochs} 轮，最多 100 轮，patience=30。",
        "冻结 Backbone 模块 0–10，训练 Neck 模块 11–22 和 Detect 模块 23；DFL 投影保持固定。",
        f"Backbone 参数和 BatchNorm 缓冲区保持不变：{audit.get('frozen_state_unchanged')}。",
        "", "指标来自同一验证集的重新评估，尚未使用保留测试集；单个随机种子的结果不代表普遍结论。",
    ]
    (run_dir / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_json(run_dir / "progress.json", {
        "completed": True, "epoch": completed_epochs,
        "comparison": str(run_dir / "comparison.json"),
    })
    print("COMPARISON_SAVED " + str(run_dir / "comparison.md"), flush=True)


if __name__ == "__main__":
    main()
