"""Reuse YOLOSHOW's model, preprocessing and box postprocessing on captured frames."""

from time import perf_counter

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


def load_image(path):
    frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("无法读取所选图片。")
    return frame


def as_image(frame):
    height, width = frame.shape[:2]
    return QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888).copy()


class Detector:
    def __init__(self, model, device="cpu", confidence=0.25, iou=0.45):
        from yolocode.yolov11.YOLOv11Thread import YOLOv11Thread

        self.backend = YOLOv11Thread()
        # An explicit device bypasses the legacy selector's process-wide CUDA mask.
        self.backend.device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        if self.backend.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用。可在检测页改选 CPU。")
        self.backend.imgsz = (model["imgsz"], model["imgsz"])
        self.backend.conf_thres = confidence
        self.backend.iou_thres = iou
        self.backend.setup_model(str(model["weights"]), verbose=False)
        names = self.backend.model.names
        names = [names[index] for index in range(len(names))]
        if names != model["classes"]:
            raise ValueError("权重的类别顺序与模型记录不一致。")

    @torch.inference_mode()
    def predict(self, frame, name):
        started = perf_counter()
        backend = self.backend
        backend.batch = ([str(name)], [frame], None)
        image = backend.preprocess([frame])
        prediction = backend.inference(image)
        result = backend.postprocess(prediction, image, [frame])[0]
        boxes = result.boxes
        return result.plot(), [
            {"class_id": int(cls), "confidence": float(conf), "xyxy": xyxy}
            for cls, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())
        ], (perf_counter() - started) * 1000


class TrialWorker(QThread):
    result = Signal(object, list, float)
    failed = Signal(str)

    def __init__(self, model, path, parent=None):
        super().__init__(parent)
        self.model = model
        self.path = path

    def run(self):
        try:
            detector = Detector(self.model)
            frame, boxes, elapsed = detector.predict(load_image(self.path), self.path)
            self.result.emit(as_image(frame), boxes, elapsed)
        except Exception as error:
            self.failed.emit(str(error))
