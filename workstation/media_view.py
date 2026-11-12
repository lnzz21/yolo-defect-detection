"""Preview camera or video frames and keep the latest original for capture."""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QSizePolicy, QSpinBox, QVBoxLayout, QWidget

from workstation.i18n import bind, button, label, show_pixmap, tr
from workstation.inference import as_image
from workstation.media import MediaWorker
from workstation.store import PROJECT


class MediaView(QWidget):
    source_selected = Signal(object)
    updated = Signal(dict)
    failed = Signal(str)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.worker = None
        self.latest = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.video_button = button("打开视频")
        self.video_button.setProperty("primary", True)
        self.camera_button = button("打开相机")
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 15)
        bind(self.camera_index, "本地相机编号", setter="setToolTip")
        self.stop_button = button("停止")
        self.stop_button.setEnabled(False)
        self.video_button.clicked.connect(self.pick_video)
        self.camera_button.clicked.connect(lambda: self.source_selected.emit(self.camera_index.value()))
        self.stop_button.clicked.connect(self.stop)
        for widget in (self.video_button, self.camera_button, self.camera_index, self.stop_button):
            toolbar.addWidget(widget)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.preview = label("请选择摄像头或视频")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview.setMinimumHeight(210)
        self.preview.setObjectName("preview")
        layout.addWidget(self.preview, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.poll)

    @property
    def busy(self):
        return self.worker is not None

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("选择本地视频"), str(PROJECT),
                                              tr("视频 (*.mp4 *.avi *.mkv *.mov *.wmv *.m4v);;所有文件 (*)"))
        if path:
            self.source_selected.emit(path)

    def start(self, source, **options):
        if self.busy:
            raise ValueError("请先停止当前视频源。")
        self.latest = None
        bind(self.preview, "正在读取视频源…")
        self.worker = MediaWorker(source, parent=self, **options)
        self.worker.failed.connect(self.show_failure)
        self.worker.finished.connect(self.finish)
        self.video_button.setEnabled(False)
        self.camera_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker.start()
        self.timer.start()

    def stop(self):
        if self.worker is not None:
            self.stop_button.setEnabled(False)
            self.worker.requestInterruption()

    def poll(self):
        if self.worker is not None:
            latest = self.worker.take_latest()
            if latest is not None:
                self.latest = latest
                self.redraw()
                self.updated.emit(latest[2])

    def redraw(self):
        if self.latest is not None:
            show_pixmap(self.preview, QPixmap.fromImage(as_image(self.latest[1])).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.redraw()

    def show_failure(self, error):
        self.failed.emit(error)

    def finish(self):
        self.timer.stop()
        self.poll()
        self.worker.deleteLater()
        self.worker = None
        self.video_button.setEnabled(True)
        self.camera_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.finished.emit()
