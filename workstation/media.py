"""Camera and video reading with a one-frame preview buffer."""

from threading import Lock
from time import perf_counter

import cv2
from PySide6.QtCore import QThread, Signal

from workstation.inference import Detector
from workstation.store import now


class MediaWorker(QThread):
    failed = Signal(str)

    def __init__(self, source, model=None, device="cpu", confidence=0.25, iou=0.45, record=None, parent=None):
        super().__init__(parent)
        self.source = source
        self.model = model
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.record = record
        self.frame_lock = Lock()
        self.latest = None

    def take_latest(self):
        with self.frame_lock:
            latest, self.latest = self.latest, None
            return latest

    def run(self):
        cap = None
        index = 0
        camera = isinstance(self.source, int)
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise RuntimeError("无法打开相机或视频。")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_period = 1 / fps if fps > 0 else 1 / 25
            detector = Detector(self.model, self.device, self.confidence, self.iou) if self.model else None
            while not self.isInterruptionRequested():
                started = perf_counter()
                ok, frame = cap.read()
                if not ok:
                    if camera or index == 0:
                        raise RuntimeError("相机断开或无法读取有效画面。")
                    return
                index += 1
                metadata = {"source": f"camera:{self.source}" if camera else self.source,
                            "frame": index, "position_ms": None if camera else cap.get(cv2.CAP_PROP_POS_MSEC),
                            "captured_at": now()}
                if detector:
                    plotted, boxes, inference_ms = detector.predict(frame, metadata["source"])
                    packet = {**metadata, "status": "defect" if boxes else "pass", "boxes": boxes,
                              "inference_ms": round(inference_ms, 2), "elapsed_ms": round((perf_counter() - started) * 1000, 2)}
                    self.record(packet)
                else:
                    plotted, packet = frame, metadata
                with self.frame_lock:
                    self.latest = (frame, plotted, packet)
                if not camera:
                    remaining = frame_period - (perf_counter() - started)
                    if remaining > 0:
                        self.msleep(min(100, int(remaining * 1000)))
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            if cap is not None:
                cap.release()
