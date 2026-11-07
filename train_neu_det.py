"""Train YOLO11n on the converted NEU-DET data, or evaluate trained weights."""

import argparse
import importlib.metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def train_detection(model, common, *, epochs, name, **options):
    return model.train(**common, epochs=epochs, patience=30, seed=42,
                       name=name, exist_ok=False, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/NEU-DET-v1/neu_det.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "ptfiles/yolo11n.pt")
    parser.add_argument("--device", default="0", help="0 for NVIDIA GPU, cpu for CPU")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="neu_yolo11n")
    parser.add_argument("--check", action="store_true", help="Check inputs/environment without training")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate --model instead of training")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    for path in (args.data, args.model):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    import torch
    import yaml

    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {importlib.metadata.version('ultralytics')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.data.resolve()}")
    print(f"Classes: {config['names']}")
    data_root = Path(config.get("path") or args.data.resolve().parent)
    for split in ("train", "val", "test"):
        if split == "test" and not config.get(split) and not (args.evaluate and args.split == "test"):
            continue
        if not config.get(split):
            parser.error(f"Dataset does not specify the {split} split.")
        image_dir = data_root / config[split]
        if not image_dir.is_dir():
            parser.error(f"Dataset directory missing: {image_dir}")
        print(f"{split}: {sum(p.is_file() for p in image_dir.iterdir())} images")
    if args.device != "cpu" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Install CUDA PyTorch, or explicitly use --device cpu.")
    if args.check:
        print("Checks passed. No training was started.")
        return

    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    common = dict(data=str(args.data.resolve()), imgsz=args.imgsz, batch=args.batch,
                  device=args.device, workers=args.workers, project=str(ROOT / "runs/detect"))
    if args.evaluate:
        metrics = model.val(**common, split=args.split, name=f"{args.name}_{args.split}")
        print(f"mAP50: {metrics.box.map50:.4f}; mAP50-95: {metrics.box.map:.4f}")
    else:
        train_detection(model, common, epochs=args.epochs, name=args.name)
        print(f"Best weights: {model.trainer.best}")
        print("For the GUI, copy best.pt to ptfiles/yolo11n_neu_det.pt.")


if __name__ == "__main__":
    # Required for multiprocessing on Windows.
    main()
