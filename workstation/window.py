"""Training and model pages hosted inside the existing YOLOSHOW application."""

import csv
import json
import math
from pathlib import Path

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar,
    QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from workstation.annotation_page import AnnotationPage
from workstation.dataset import IMAGE_TYPES, dataset_yaml
from workstation.i18n import LanguageSelector, bind, button, label, language, msg, show_pixmap, tr
from workstation.inference import TrialWorker
from workstation.inspection_page import InspectionPage
from workstation.store import PROJECT, WorkstationStore
from workstation.training import TrainingController


STATE_LABELS = {"starting": "启动中", "running": "训练中", "completed": "已完成", "stopped": "已停止", "failed": "失败"}
LOSS_KEYS = {"train/box_loss": "框损失", "train/cls_loss": "分类损失", "train/dfl_loss": "DFL"}
MAP_KEYS = {"metrics/mAP50(B)": "mAP@0.5", "metrics/mAP50-95(B)": "mAP@0.5:0.95"}
IMAGE_EXTENSIONS = " ".join(f"*{suffix}" for suffix in sorted(IMAGE_TYPES))


class Curve(QChartView):
    def __init__(self, title, keys, unit=False):
        chart = QChart()
        super().__init__(chart)
        bind(chart, title, setter="setTitle")
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMinimumHeight(200)
        self.unit = unit
        self.series = {}
        self.x = QValueAxis()
        bind(self.x, "轮次", setter="setTitleText")
        self.x.setLabelFormat("%d")
        self.x.setTickType(QValueAxis.TickType.TicksDynamic)
        self.x.setTickAnchor(0)
        self.y = QValueAxis()
        chart.addAxis(self.x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(self.y, Qt.AlignmentFlag.AlignLeft)
        for key, name in keys.items():
            series = QLineSeries()
            bind(series, name, setter="setName")
            series.setPointsVisible(True)
            chart.addSeries(series)
            series.attachAxis(self.x)
            series.attachAxis(self.y)
            self.series[key] = series
        self.clear()

    def clear(self):
        for series in self.series.values():
            series.clear()
        self.maximum = 1
        self.x.setRange(0, 2)
        self.x.setTickInterval(1)
        self.y.setRange(0, 1)

    def add_epoch(self, epoch, metrics):
        for key, series in self.series.items():
            if key in metrics:
                value = float(metrics[key])
                series.append(epoch, value)
                self.maximum = max(self.maximum, value)
        self.x.setRange(0, max(2, epoch))
        self.x.setTickInterval(max(1, math.ceil(epoch / 6)))
        if not self.unit:
            self.y.setRange(0, self.maximum * 1.06)


class TrainingPage(QWidget):
    def __init__(self, controller, compute_available):
        super().__init__()
        self.controller = controller
        self.compute_available = compute_available
        layout = QVBoxLayout(self)
        form = QGridLayout()
        self.data = QLineEdit()
        bind(self.data, "选择 YOLO 数据集的 YAML 或所在目录", setter="setPlaceholderText")
        self.data.textChanged.connect(self.data.setToolTip)
        default_data = PROJECT / "datasets" / "NEU-DET-v1" / "neu_det.yaml"
        if default_data.is_file():
            self.data.setText(str(default_data))
        self.weights = QLineEdit(str(PROJECT / "ptfiles" / "yolo11n.pt"))
        yaml_button = button("选 YAML")
        yaml_button.clicked.connect(self.pick_yaml)
        directory_button = button("选目录")
        directory_button.clicked.connect(self.pick_directory)
        weights_button = button("选权重")
        weights_button.clicked.connect(self.pick_weights)
        form.addWidget(label("数据集"), 0, 0)
        form.addWidget(self.data, 0, 1)
        form.addWidget(yaml_button, 0, 2)
        form.addWidget(directory_button, 0, 3)
        form.addWidget(label("初始权重"), 1, 0)
        form.addWidget(self.weights, 1, 1)
        form.addWidget(weights_button, 1, 2)
        layout.addLayout(form)
        options = QHBoxLayout()
        self.epochs = self.spin(1, 1000, 100)
        self.batch = self.spin(1, 128, 4)
        self.imgsz = self.spin(64, 2048, 640, 32)
        self.device = QComboBox()
        self.device.addItem("GPU 0", "0")
        self.device.addItem("CPU", "cpu")
        for title, widget in (("轮数", self.epochs), ("批大小", self.batch), ("输入尺寸", self.imgsz), ("设备", self.device)):
            options.addWidget(label(title))
            options.addWidget(widget)
        options.addStretch()
        self.start_button = button("开始训练")
        self.start_button.clicked.connect(self.start)
        self.stop_button = button("本轮后停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        options.addWidget(self.start_button)
        options.addWidget(self.stop_button)
        layout.addLayout(options)
        self.status = label("请选择数据集。短轮次可用于检查训练链路。")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        charts = QSplitter(Qt.Orientation.Horizontal)
        self.losses = Curve("训练损失", LOSS_KEYS)
        self.validation = Curve("验证 mAP", MAP_KEYS, unit=True)
        charts.addWidget(self.losses)
        charts.addWidget(self.validation)
        layout.addWidget(charts, 2)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        bind(self.log, "训练日志会出现在这里", setter="setPlaceholderText")
        layout.addWidget(self.log, 1)
        controller.message.connect(self.log_message)
        controller.event.connect(self.handle_event)
        controller.completed.connect(self.finished)

    @staticmethod
    def spin(minimum, maximum, value, step=1):
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    def pick_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("选择 YOLO 数据集"), self.data.text(), "YAML (*.yaml *.yml)")
        if path:
            self.data.setText(path)

    def pick_directory(self):
        path = QFileDialog.getExistingDirectory(self, tr("选择数据集目录"), str(PROJECT / "datasets"))
        if path:
            self.data.setText(path)

    def pick_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("选择初始权重"), str(PROJECT / "ptfiles"), tr("PyTorch 权重 (*.pt)"))
        if path:
            self.weights.setText(path)

    def start(self):
        try:
            if not self.compute_available():
                raise ValueError("请先结束当前检测或试跑，再开始训练。")
            data = dataset_yaml(self.data.text().strip())
            weights = Path(self.weights.text().strip()).resolve()
            if not weights.is_file() or weights.suffix.lower() != ".pt":
                raise ValueError("请选择本地 YOLO11 检测权重 .pt。")
            if self.imgsz.value() % 32:
                raise ValueError("输入尺寸须为 32 的整数倍。")
            settings = {"data": str(data), "weights": str(weights), "epochs": self.epochs.value(),
                        "batch": self.batch.value(), "imgsz": self.imgsz.value(), "device": self.device.currentData()}
            self.log.clear()
            self.losses.clear()
            self.validation.clear()
            self.progress.setRange(0, settings["epochs"])
            self.progress.setValue(0)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            directory = self.controller.start(settings)
            if self.controller.busy:
                bind(self.status, "正在启动任务 {job}", job=directory.name)
        except (OSError, ValueError, RuntimeError) as error:
            self.start_button.setEnabled(not self.controller.busy)
            self.stop_button.setEnabled(self.controller.busy)
            bind(self.status, str(error))

    def stop(self):
        try:
            self.controller.stop()
            self.stop_button.setEnabled(False)
        except OSError as error:
            bind(self.status, str(error))

    def log_message(self, text):
        self.log.appendPlainText(tr(text))

    def handle_event(self, event):
        if event["kind"] == "epoch":
            epoch, metrics = event["epoch"], event["metrics"]
            self.progress.setValue(epoch)
            self.losses.add_epoch(epoch, metrics)
            self.validation.add_epoch(epoch, metrics)
            values = "  ".join(f"{name} {metrics[key]:.4f}" for key, name in MAP_KEYS.items() if key in metrics)
            bind(self.status, "第 {epoch} / {epochs} 轮  {metrics}", epoch=epoch, epochs=event["epochs"], metrics=values)
        elif event["kind"] == "stage":
            bind(self.status, event["message"])
        elif event["kind"] == "dataset":
            self.log.appendPlainText(tr("训练数据版本 {version}  {counts}", version=event["id"][:12], counts=event["counts"]))

    def finished(self, record):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        state = record["state"]
        bind(self.status, "任务 {state}  {error}", state=msg(STATE_LABELS.get(state, state)), error=msg(record.get("error", "")))


class ModelPage(QWidget):
    changed = Signal()

    def __init__(self, store, compute_available):
        super().__init__()
        self.store = store
        self.compute_available = compute_available
        self.trial = None
        self.preview_image = None
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        self.jobs = QComboBox()
        self.jobs.currentIndexChanged.connect(self.show_job)
        self.job_info = QPlainTextEdit()
        self.job_info.setReadOnly(True)
        self.job_loss = Curve("历史训练损失", LOSS_KEYS)
        self.job_map = Curve("历史验证 mAP", MAP_KEYS, unit=True)
        open_folder = button("打开任务目录")
        open_folder.clicked.connect(self.open_job)
        left.addWidget(label("训练历史"))
        left.addWidget(self.jobs)
        left.addWidget(self.job_info, 1)
        left.addWidget(self.job_loss, 2)
        left.addWidget(self.job_map, 2)
        left.addWidget(open_folder)
        right = QVBoxLayout()
        self.models = QComboBox()
        self.active = QLabel()
        self.active.setWordWrap(True)
        self.preview = label("选择模型，再用一张图试跑。")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(300, 240)
        self.preview.setStyleSheet("background: #17212c; color: #edf3f9; border-radius: 5px;")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.try_button = button("用样图试跑")
        self.try_button.clicked.connect(self.pick_trial)
        enable = button("启用所选模型")
        enable.clicked.connect(self.activate)
        rollback = button("切回上一版")
        rollback.clicked.connect(self.rollback)
        actions = QHBoxLayout()
        for action in (self.try_button, enable, rollback):
            actions.addWidget(action)
        right.addWidget(label("模型库"))
        right.addWidget(self.models)
        right.addWidget(self.active)
        right.addLayout(actions)
        right.addWidget(self.preview, 1)
        right.addWidget(self.status)
        layout.addLayout(left, 1)
        layout.addLayout(right, 1)
        self.refresh()
        language().changed.connect(self.refresh)

    def refresh(self):
        current_job = self.jobs.currentData()
        current_model = self.models.currentData()
        self.jobs.blockSignals(True)
        self.jobs.clear()
        for path, record in self.store.history():
            state = record["state"]
            self.jobs.addItem(f"{path.name}  {tr(STATE_LABELS.get(state, state))}", str(path))
        index = self.jobs.findData(current_job)
        self.jobs.setCurrentIndex(max(0, index))
        self.jobs.blockSignals(False)
        self.show_job()
        models = self.store.models()
        selection = self.store.selection()
        self.models.clear()
        for model_id, model in models.items():
            self.models.addItem(tr(model["name"]), model_id)
        index = self.models.findData(current_model or selection["active"])
        self.models.setCurrentIndex(max(0, index))
        active = models.get(selection["active"])
        previous = models.get(selection["previous"])
        bind(self.active, "当前启用  {active}\n上一版本  {previous}",
             active=msg(active["name"] if active else "无"), previous=msg(previous["name"] if previous else "无"))

    def show_job(self):
        self.job_loss.clear()
        self.job_map.clear()
        path = self.jobs.currentData()
        if not path:
            bind(self.job_info, "暂无训练记录。", setter="setPlainText")
            return
        directory = Path(path)
        try:
            record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            dataset = record.get("dataset", {})
            fields = record["settings"]
            best = record.get("best_metrics", {})
            best_line = "  ".join(f"{name} {best[key]:.4f}" for key, name in MAP_KEYS.items() if key in best)
            bind(self.job_info,
                 "数据源  {data}\n数据版本  {version}\n初始权重  {weights}\n轮数 {epochs}  批大小 {batch}  尺寸 {imgsz}  设备 {device}\n{best}\n{error}",
                 setter="setPlainText", version=msg(dataset.get("id", "待生成")), best=best_line,
                 error=msg(record.get("error", "")), **fields)
            results = directory / "training" / "results.csv"
            if results.is_file():
                with results.open(encoding="utf-8-sig", newline="") as file:
                    for row in csv.DictReader(file, skipinitialspace=True):
                        row = {k.strip(): v for k, v in row.items() if k and v}
                        epoch = int(float(row["epoch"]))
                        self.job_loss.add_epoch(epoch, row)
                        self.job_map.add_epoch(epoch, row)
        except (OSError, KeyError, ValueError) as error:
            bind(self.job_info, "读取训练记录失败 {error}", setter="setPlainText", error=str(error))

    def open_job(self):
        if self.jobs.currentData():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.jobs.currentData()))

    def pick_trial(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("选择试跑图片"), str(PROJECT), tr("图片 ({extensions})", extensions=IMAGE_EXTENSIONS))
        if path:
            self.start_trial(path)

    def start_trial(self, path):
        model = self.store.models().get(self.models.currentData())
        if not model or self.trial is not None or not self.compute_available():
            bind(self.status, "请选择模型，并先结束正在运行的训练或检测。")
            return
        self.try_button.setEnabled(False)
        bind(self.status, "正在试跑 {model}…", model=msg(model["name"]))
        self.trial = TrialWorker(model, path, self)
        self.trial.result.connect(self.show_trial)
        self.trial.failed.connect(self.trial_failed)
        self.trial.finished.connect(self.finish_trial)
        self.trial.start()

    def show_trial(self, image, boxes, elapsed):
        self.preview_image = image
        self.update_preview()
        bind(self.status, "试跑完成  {count} 个检测框  {elapsed:.1f} ms", count=len(boxes), elapsed=elapsed)

    def trial_failed(self, error):
        bind(self.status, error)

    def update_preview(self):
        if self.preview_image is not None:
            show_pixmap(self.preview, QPixmap.fromImage(self.preview_image).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview()

    def finish_trial(self):
        self.trial.deleteLater()
        self.trial = None
        self.try_button.setEnabled(True)

    def activate(self):
        try:
            if not self.compute_available():
                raise ValueError("请先结束训练或检测，再切换模型。")
            self.store.activate(self.models.currentData())
            self.refresh()
            self.changed.emit()
            bind(self.status, "已保存启用版本。")
        except (OSError, ValueError) as error:
            bind(self.status, str(error))

    def rollback(self):
        try:
            if not self.compute_available():
                raise ValueError("请先结束训练或检测，再切换模型。")
            self.store.rollback()
            self.refresh()
            self.changed.emit()
            bind(self.status, "已切回上一版。")
        except (OSError, ValueError) as error:
            bind(self.status, str(error))


class WorkstationWindow(QDialog):
    model_changed = Signal()

    def __init__(self, parent=None, store=None, legacy_busy=None):
        super().__init__(parent)
        bind(self, "YOLOSHOW · 钢材缺陷工作台", setter="setWindowTitle")
        self.resize(1080, 760)
        self.store = store or WorkstationStore()
        self.legacy_busy = legacy_busy or (lambda: False)
        self.controller = TrainingController(self.store, self)
        self.training = TrainingPage(self.controller, self.compute_available)
        self.library = ModelPage(self.store, self.compute_available)
        self.annotation = AnnotationPage(self.store, self.compute_available,
                                         lambda: not (self.inspection.media.busy or self.legacy_busy()))
        self.inspection = InspectionPage(self.store, lambda: self.compute_available() and not self.annotation.media.busy)
        self.library.changed.connect(self.model_changed)
        self.library.changed.connect(self.inspection.refresh_model)
        self.inspection.capture_requested.connect(self.capture_for_annotation)
        self.annotation.dataset_created.connect(self.training.data.setText)
        self.controller.completed.connect(lambda _: self.library.refresh())
        self.tabs = QTabWidget()
        self.tab_names = ("现场检测", "采集与标注", "可视化训练", "训练历史与模型")
        for page, name in zip((self.inspection, self.annotation, self.training, self.library), self.tab_names):
            self.tabs.addTab(page, tr(name))
        language().changed.connect(self.translate_tabs)
        self.language_selector = LanguageSelector()
        self.tabs.setCornerWidget(self.language_selector, Qt.Corner.TopRightCorner)
        self.tabs.currentChanged.connect(self.refresh_page)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

    def translate_tabs(self):
        for index, name in enumerate(self.tab_names):
            self.tabs.setTabText(index, tr(name))

    def compute_available(self):
        return not (self.controller.busy or self.library.trial or self.annotation.prelabel
                    or self.annotation.exporter or self.inspection.media.busy or self.legacy_busy())

    def busy(self):
        return (self.controller.busy or self.library.trial is not None or self.annotation.prelabel is not None
                or self.annotation.exporter is not None or self.annotation.media.busy or self.inspection.media.busy)

    def refresh_page(self):
        if self.tabs.currentWidget() is self.library:
            self.library.refresh()
        if self.tabs.currentWidget() is self.inspection:
            self.inspection.refresh_model()

    def capture_for_annotation(self, frame, metadata):
        self.annotation.accept_frame(frame, metadata)
        self.tabs.setCurrentWidget(self.annotation)

    def closeEvent(self, event):
        if self.busy():
            QMessageBox.information(self, tr("任务正在运行"), tr("结束当前任务后再关闭工作台。"))
            event.ignore()
        else:
            super().closeEvent(event)
