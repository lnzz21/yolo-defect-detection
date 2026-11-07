"""Capture raw video frames, review boxes and create datasets for training."""

from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QPointF, QRectF, QSignalBlocker, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLineEdit, QMessageBox,
    QSplitter, QVBoxLayout, QWidget,
)

from workstation.annotations import Annotations
from workstation.i18n import bind, button, label, language, msg, tr
from workstation.inference import TrialWorker, as_image, load_image
from workstation.media_view import MediaView
from workstation.store import PROJECT, STEEL_CLASSES


class BoxCanvas(QWidget):
    changed = Signal()
    selected = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(310, 235)
        self.image = None
        self.boxes = []
        self.index = -1
        self.class_id = 0
        self.origin = self.endpoint = None

    def set_frame(self, frame, boxes):
        self.image = as_image(frame)
        self.boxes = [{"class_id": box["class_id"], "xyxy": list(box["xyxy"])} for box in boxes]
        self.index = -1
        self.origin = self.endpoint = None
        self.update()

    def image_rect(self):
        width, height = self.image.width(), self.image.height()
        scale = min(self.width() / width, self.height() / height)
        width, height = width * scale, height * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height), scale

    def original_point(self, point):
        rect, scale = self.image_rect()
        return QPointF(max(0, min(self.image.width(), (point.x() - rect.x()) / scale)),
                       max(0, min(self.image.height(), (point.y() - rect.y()) / scale)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#17212c"))
        if self.image is None:
            painter.setPen(QColor("#edf3f9"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr("抓取原始帧后，在这里拖动画框"))
            return
        rect, scale = self.image_rect()
        painter.drawImage(rect, self.image)
        for index, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box["xyxy"]
            painter.setPen(QPen(QColor("#ffd64a" if index == self.index else "#36e8ad"), 2))
            draw = QRectF(rect.x() + x1 * scale, rect.y() + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale)
            painter.drawRect(draw)
            painter.drawText(draw.topLeft() + QPointF(2, 14), tr(STEEL_CLASSES[box["class_id"]]))
        if self.origin is not None and self.endpoint is not None:
            painter.setPen(QPen(QColor("#ffe778"), 2, Qt.PenStyle.DashLine))
            start = rect.topLeft() + self.origin * scale
            end = rect.topLeft() + self.endpoint * scale
            painter.drawRect(QRectF(start, end).normalized())

    def mousePressEvent(self, event):
        if self.image is not None and event.button() == Qt.MouseButton.LeftButton and self.image_rect()[0].contains(event.position()):
            self.origin = self.endpoint = self.original_point(event.position())
            self.update()

    def mouseMoveEvent(self, event):
        if self.origin is not None:
            self.endpoint = self.original_point(event.position())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.origin is None:
            return
        target = self.original_point(event.position())
        rect = QRectF(self.origin, target).normalized()
        self.origin = self.endpoint = None
        if min(rect.width(), rect.height()) >= 2:
            self.boxes.append({"class_id": self.class_id, "xyxy": [rect.left(), rect.top(), rect.right(), rect.bottom()]})
            self.pick(len(self.boxes) - 1)
            self.changed.emit()
        else:
            index = next((i for i in reversed(range(len(self.boxes)))
                          if QRectF(QPointF(*self.boxes[i]["xyxy"][:2]), QPointF(*self.boxes[i]["xyxy"][2:])).contains(target)), -1)
            self.pick(index)

    def pick(self, index):
        self.index = index
        self.selected.emit(index)
        self.update()

    def delete_selected(self):
        if self.index >= 0:
            self.boxes.pop(self.index)
            self.pick(-1)
            self.changed.emit()


class ExportWorker(QThread):
    status = Signal(str)
    failed = Signal(str)
    created = Signal(dict)

    def __init__(self, annotations, base, parent):
        super().__init__(parent)
        self.annotations = annotations
        self.base = base

    def run(self):
        try:
            self.created.emit(self.annotations.export(self.base, self.status.emit))
        except Exception as error:
            self.failed.emit(str(error))


class AnnotationPage(QWidget):
    dataset_created = Signal(str)

    def __init__(self, store, compute_available, capture_available):
        super().__init__()
        self.store = store
        self.compute_available = compute_available
        self.capture_available = capture_available
        self.annotations = Annotations(store)
        self.item = None
        self.recording_group = None
        self.prelabel = None
        self.exporter = None
        layout = QVBoxLayout(self)
        self.media = MediaView()
        self.media.source_selected.connect(self.start_capture)
        self.capture = button("抓帧到标注区")
        self.capture.clicked.connect(self.capture_latest)
        self.group = QLineEdit()
        bind(self.group, "工件或采集批次（可选）", setter="setPlaceholderText")
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.media, 1)
        left_layout.addWidget(self.group)
        left_layout.addWidget(self.capture)

        right = QWidget()
        self.editor = right
        right_layout = QVBoxLayout(right)
        self.items = QComboBox()
        self.items.currentIndexChanged.connect(self.load_selected)
        self.canvas = BoxCanvas()
        self.canvas.changed.connect(self.save_draft)
        self.canvas.selected.connect(self.selected)
        self.classes = QComboBox()
        self.classes.addItems([tr(name) for name in STEEL_CLASSES])
        self.classes.currentIndexChanged.connect(self.class_changed)
        delete = button("删除所选框")
        delete.clicked.connect(self.canvas.delete_selected)
        suggest = button("模型预标注")
        suggest.clicked.connect(self.run_prelabel)
        review = button("审核并保存")
        review.clicked.connect(lambda: self.save("reviewed"))
        normal = button("确认无缺陷")
        normal.clicked.connect(lambda: self.save("negative"))
        actions = QHBoxLayout()
        for widget in (self.classes, delete, suggest, review, normal):
            actions.addWidget(widget)
        right_layout.addWidget(self.items)
        right_layout.addWidget(self.canvas, 1)
        right_layout.addLayout(actions)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        layout.addWidget(splitter, 1)
        base_row = QHBoxLayout()
        self.base = QLineEdit()
        bind(self.base, "可留空，按采集来源独立划分", setter="setPlaceholderText")
        default = PROJECT / "datasets" / "NEU-DET-v1" / "neu_det.yaml"
        if default.is_file():
            self.base.setText(str(default))
        pick_base = button("选择基础 YAML")
        pick_base.clicked.connect(self.pick_base)
        export = button("生成数据集版本")
        export.clicked.connect(self.export_dataset)
        base_row.addWidget(label("基础数据集"))
        base_row.addWidget(self.base, 1)
        base_row.addWidget(pick_base)
        base_row.addWidget(export)
        layout.addLayout(base_row)
        self.status = label("拖动新增框，单击选框后可换类别或删除。")
        self.status.setWordWrap(True)
        self.media.failed.connect(self.set_status)
        layout.addWidget(self.status)
        self.refresh_items()
        language().changed.connect(self.translate_items)

    def set_status(self, text):
        bind(self.status, text)

    @staticmethod
    def item_title(record):
        state = {"pending": "待审核", "reviewed": "有缺陷", "negative": "无缺陷"}[record["state"]]
        return tr("{state} · {source} · 第 {frame} 帧", state=tr(state),
                  source=Path(record["source"]).name, frame=record["frame"])

    def translate_items(self):
        with QSignalBlocker(self.classes), QSignalBlocker(self.items):
            for index in range(self.classes.count()):
                self.classes.setItemText(index, tr(STEEL_CLASSES[index]))
            for index in range(self.items.count()):
                self.items.setItemText(index, self.item_title(self.items.itemData(index)))
        self.canvas.update()

    def start_capture(self, source):
        if not self.capture_available():
            self.set_status("请先结束检测，再打开另一个视频源。")
            return
        self.recording_group = f"camera-session:{uuid4().hex}" if isinstance(source, int) else None
        try:
            self.media.start(source)
        except ValueError as error:
            self.set_status(str(error))

    def capture_latest(self):
        if self.media.latest is None:
            self.set_status("请先打开视频或相机并等待一帧画面。")
            return
        frame, _, metadata = self.media.latest
        self.accept_frame(frame, {**metadata, "group": self.group.text().strip() or self.recording_group or metadata["source"]})

    def accept_frame(self, frame, metadata):
        if self.prelabel or self.exporter:
            self.set_status("请等待当前预标注或导出任务完成。")
            return
        try:
            record = self.annotations.capture(frame, metadata)
            self.refresh_items(record["id"])
            bind(self.status, "已抓取第 {frame} 帧，可开始标注。", frame=metadata["frame"])
        except (OSError, ValueError, RuntimeError) as error:
            self.set_status(str(error))

    def refresh_items(self, preferred=None):
        preferred = preferred or (self.item or {}).get("id")
        self.items.blockSignals(True)
        self.items.clear()
        for record in self.annotations.items():
            self.items.addItem(self.item_title(record), record)
        index = next((i for i in range(self.items.count()) if self.items.itemData(i)["id"] == preferred), 0)
        self.items.setCurrentIndex(index)
        self.items.blockSignals(False)
        self.load_selected()

    def load_selected(self):
        record = self.items.currentData()
        if record:
            self.item = record
            self.canvas.set_frame(load_image(self.annotations.image(record["id"])), record["boxes"])

    def selected(self, index):
        if index >= 0:
            self.classes.blockSignals(True)
            self.classes.setCurrentIndex(self.canvas.boxes[index]["class_id"])
            self.classes.blockSignals(False)
            self.canvas.class_id = self.classes.currentIndex()

    def class_changed(self, class_id):
        self.canvas.class_id = class_id
        if self.canvas.index >= 0:
            self.canvas.boxes[self.canvas.index]["class_id"] = class_id
            self.canvas.update()
            self.save_draft()

    def save_draft(self):
        self.save("pending")

    def save(self, state):
        if not self.item or self.exporter:
            self.set_status("请选择抓帧样本，或等待导出完成。")
            return
        try:
            self.item = self.annotations.save(self.item["id"], self.canvas.boxes, state)
            if state != "pending":
                self.refresh_items()
            else:
                index = self.items.currentIndex()
                self.items.setItemData(index, self.item)
                self.items.setItemText(index, self.item_title(self.item))
            self.set_status("草稿已保存。" if state == "pending" else "审核完成，图片与 YOLO 标签已保存。")
        except (OSError, ValueError) as error:
            self.set_status(str(error))

    def run_prelabel(self):
        model = self.store.active_model()
        if not self.item or not model or self.prelabel or self.exporter or not self.compute_available():
            self.set_status("请选择抓帧样本与已启用模型，并等待训练或检测结束。")
            return
        if model["classes"] != STEEL_CLASSES:
            self.set_status("当前模型的类别顺序与钢材标注类别不一致。")
            return
        if self.canvas.boxes and QMessageBox.question(self, tr("替换标注草稿"), tr("用模型预测结果替换当前框？")) != QMessageBox.StandardButton.Yes:
            return
        item_id = self.item["id"]
        self.prelabel = TrialWorker(model, self.annotations.image(item_id), self)
        self.prelabel.result.connect(lambda image, boxes, elapsed: self.receive_prelabel(item_id, boxes))
        self.prelabel.failed.connect(self.set_status)
        self.prelabel.finished.connect(self.finish_prelabel)
        self.editor.setEnabled(False)
        self.set_status("正在生成候选框…")
        self.prelabel.start()

    def receive_prelabel(self, item_id, boxes):
        if self.item and self.item["id"] == item_id:
            frame = load_image(self.annotations.image(item_id))
            self.canvas.set_frame(frame, boxes)
            self.save_draft()
            bind(self.status, "生成了 {count} 个候选框。请核对后审核。", count=len(boxes))

    def finish_prelabel(self):
        self.prelabel.deleteLater()
        self.prelabel = None
        self.editor.setEnabled(True)

    def pick_base(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("选择基础数据集"), self.base.text(), "YAML (*.yaml *.yml)")
        if path:
            self.base.setText(path)

    def export_dataset(self):
        if self.exporter or self.prelabel:
            return
        self.exporter = ExportWorker(self.annotations, self.base.text().strip(), self)
        self.exporter.status.connect(self.set_status)
        self.exporter.failed.connect(self.export_failed)
        self.exporter.created.connect(self.exported)
        self.exporter.finished.connect(self.finish_export)
        self.editor.setEnabled(False)
        self.set_status("正在检查审核样本并生成数据集…")
        self.exporter.start()

    def exported(self, dataset):
        bind(self.status, "已生成版本 {version}，已填入训练页。", version=dataset["id"][:12])
        self.dataset_created.emit(dataset["yaml"])

    def export_failed(self, error):
        bind(self.status, "导出失败 {error}", error=msg(error))

    def finish_export(self):
        self.exporter.deleteLater()
        self.exporter = None
        self.editor.setEnabled(True)
