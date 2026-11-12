"""Operate the approved model against a camera or video and persist the outcomes."""

import json
from datetime import datetime
from uuid import uuid4

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from workstation.i18n import bind, button, label, msg
from workstation.media_view import MediaView
from workstation.store import now, write_json


class InspectionSession:
    def __init__(self, store, source, model, device, confidence, iou, part):
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        self.directory = store.root / "inspections" / session_id
        self.directory.mkdir(parents=True)
        self.record = {"id": session_id, "state": "running", "started_at": now(), "source": source,
                       "model_id": model["id"], "weights": str(model["weights"]), "classes": model["classes"],
                       "device": device, "confidence": confidence, "iou": iou, "part": part}
        self.frames = 0
        self.defects = 0
        write_json(self.directory / "session.json", self.record)

    def log(self, packet):
        line = json.dumps({**packet, "model_id": self.record["model_id"], "part": self.record["part"]}, ensure_ascii=False)
        with (self.directory / "frames.jsonl").open("a", encoding="utf-8") as file:
            file.write(line + "\n")
        if packet["status"] != "not_inspected":
            self.frames += 1
            self.defects += packet["status"] == "defect"

    def finish(self, state, error):
        self.record.update(state=state, error=error, ended_at=now(), inspected_frames=self.frames, defect_frames=self.defects)
        write_json(self.directory / "session.json", self.record)


class InspectionPage(QWidget):
    capture_requested = Signal(object, dict)

    def __init__(self, store, compute_available):
        super().__init__()
        self.store = store
        self.compute_available = compute_available
        self.session = None
        self.error = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        live = QFrame(objectName="card")
        live_layout = QVBoxLayout(live)
        live_title = label("检测画面")
        live_title.setObjectName("cardTitle")
        live_layout.addWidget(live_title)

        controls = QFrame(objectName="card")
        controls.setMinimumWidth(245)
        controls.setMaximumWidth(300)
        control_layout = QVBoxLayout(controls)
        control_title = label("检测设置")
        control_title.setObjectName("cardTitle")
        control_layout.addWidget(control_title)
        self.active = QLabel()
        self.active.setObjectName("activeModel")
        self.active.setWordWrap(True)
        control_layout.addWidget(self.active)
        options = QFormLayout()
        self.device = QComboBox()
        self.device.addItem("CPU", "cpu")
        self.device.addItem("GPU 0", "0")
        self.confidence = self.threshold(0.25)
        self.iou = self.threshold(0.45)
        self.part = QLineEdit()
        bind(self.part, "工件编号（可选）", setter="setPlaceholderText")
        for name, widget in (("设备", self.device), ("置信度", self.confidence), ("NMS IoU", self.iou)):
            options.addRow(label(name), widget)
        options.addRow(self.part)
        control_layout.addLayout(options)
        self.media = MediaView()
        self.media.source_selected.connect(self.start)
        self.media.updated.connect(self.update_result)
        self.media.failed.connect(self.failed)
        self.media.finished.connect(self.finished)
        live_layout.addWidget(self.media, 1)
        self.status = label("打开摄像头或视频后，使用已启用的模型检测。")
        self.status.setWordWrap(True)
        self.status.setObjectName("statusText")
        live_layout.addWidget(self.status)
        capture = button("送原图到标注页")
        capture.clicked.connect(self.capture)
        logs = button("打开检测记录")
        logs.clicked.connect(self.open_logs)
        control_layout.addWidget(capture)
        control_layout.addWidget(logs)
        control_layout.addStretch()
        layout.addWidget(live, 1)
        layout.addWidget(controls)
        self.refresh_model()

    @staticmethod
    def threshold(value):
        widget = QDoubleSpinBox()
        widget.setRange(0.01, 1)
        widget.setSingleStep(0.05)
        widget.setValue(value)
        return widget

    def refresh_model(self):
        model = self.store.active_model()
        bind(self.active, "当前模型\n{model}", model=msg(model["name"] if model else "无，请先在模型库启用"))

    def start(self, source):
        model = self.store.active_model()
        if not model or not self.compute_available():
            bind(self.status, "请启用模型，并结束正在运行的训练、试跑或采集。")
            return
        device = self.device.currentData()
        confidence, iou = self.confidence.value(), self.iou.value()
        try:
            self.session = InspectionSession(self.store, source, model, device, confidence, iou, self.part.text().strip())
            self.error = None
            self.media.start(source, model=model, device=device, confidence=confidence, iou=iou, record=self.session.log)
            bind(self.status, "正在加载模型和视频源…")
        except (OSError, ValueError) as error:
            bind(self.status, str(error))

    def update_result(self, packet):
        if self.error:
            return
        state = "发现缺陷" if packet["status"] == "defect" else "本帧未检出缺陷"
        bind(self.status, "第 {frame} 帧  {state}  {count} 个框  {elapsed:.1f} ms",
             frame=packet["frame"], state=msg(state), count=len(packet["boxes"]), elapsed=packet["elapsed_ms"])

    def failed(self, error):
        self.error = error
        bind(self.status, "未完成检测  {error}", error=msg(error))

    def finished(self):
        try:
            if self.error:
                self.session.log({"status": "not_inspected", "captured_at": now(), "error": self.error})
            self.session.finish("failed" if self.error else "finished", self.error)
            if not self.error:
                bind(self.status, "本次检测结束  {frames} 帧  {defects} 帧发现缺陷", frames=self.session.frames, defects=self.session.defects)
        except OSError as error:
            bind(self.status, "无法保存检测记录  {error}", error=str(error))

    def capture(self):
        if self.media.latest is None:
            bind(self.status, "等待检测画面后可送去标注。")
            return
        frame, _, packet = self.media.latest
        group = self.session.record["part"] or (self.session.record["id"] if isinstance(self.session.record["source"], int) else packet["source"])
        self.capture_requested.emit(frame, {**packet, "group": group})

    def open_logs(self):
        directory = self.session.directory if self.session else self.store.root / "inspections"
        if directory.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))
        else:
            bind(self.status, "暂无检测记录。")
