"""Lay out the original inference controls without replacing their handlers."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from workstation.i18n import bind, label


def reuse(widget, caption=None):
    widget.setStyleSheet("")
    widget.setFont(QFont())
    widget.setMinimumSize(0, 0)
    widget.setMaximumSize(16777215, 16777215)
    widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    if caption:
        bind(widget, caption)
    return widget


def card(title):
    frame = QFrame(objectName="card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 10)
    heading = label(title)
    heading.setObjectName("cardTitle")
    layout.addWidget(heading)
    return frame, layout


class InferenceView(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = next((url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()), None)
        if path:
            self.controller.load_dropped_source(path)
            event.acceptProposedAction()


def install_inference_view(controller, *, compare):
    ui = controller.ui
    old_content = controller.takeCentralWidget()
    old_content.setParent(controller)
    old_content.hide()

    view = InferenceView(controller)
    layout = QVBoxLayout(view)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)

    toolbar, toolbar_layout = card("输入与运行")
    row = QHBoxLayout()
    sources = [(ui.src_img, "打开媒体"), (ui.src_folder, "打开文件夹"), (ui.src_camera, "网络相机")]
    if not compare:
        sources.insert(1, (ui.src_webcam, "本地相机"))
    for control, caption in sources:
        control.setShortcut("")
        row.addWidget(reuse(control, caption))
    row.addStretch()
    ui.run_button.setIcon(QIcon())
    ui.run_button.setProperty("primary", True)
    bind(ui.run_button, "开始推理")
    ui.run_button.toggled.connect(lambda running: bind(ui.run_button, "暂停推理" if running else "开始推理"))
    ui.stop_button.setIcon(QIcon())
    row.addWidget(reuse(ui.run_button))
    row.addWidget(reuse(ui.stop_button, "停止"))
    toolbar_layout.addLayout(row)
    layout.addWidget(toolbar)

    panels = QHBoxLayout()
    model_fields = (ui.model_box1, ui.model_box2) if compare else (None, ui.model_box)
    metrics = (
        ((ui.Class_num1, ui.Target_num1, ui.fps_label1), (ui.Class_num2, ui.Target_num2, ui.fps_label2))
        if compare else ((), (ui.Class_num, ui.Target_num, ui.fps_label))
    )
    titles = ("模型 A", "模型 B") if compare else ("原始画面", "推理结果")
    for title, preview, model, values in zip(titles, (ui.main_leftbox, ui.main_rightbox), model_fields, metrics):
        frame, column = card(title)
        if model is not None:
            column.addWidget(reuse(model))
        reuse(preview)
        preview.setObjectName("preview")
        preview.setMinimumHeight(175)
        preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        column.addWidget(preview, 1)
        if values:
            counts = QHBoxLayout()
            for caption, value in zip(("类别数", "目标数", "FPS"), values):
                counts.addWidget(label(caption))
                counts.addWidget(reuse(value), 1)
            column.addLayout(counts)
        panels.addWidget(frame, 1)
        ui.rightbox_main = frame
    layout.addLayout(panels, 1)

    settings, settings_layout = card("推理参数与输出")
    ui.main_rightbox_play = settings
    fields = QGridLayout()
    for column, (caption, spin) in enumerate((
        ("置信度", ui.conf_spinbox), ("NMS IoU", ui.iou_spinbox),
        ("帧间延时 (ms)", ui.speed_spinbox), ("框线宽度", ui.line_spinbox),
    )):
        fields.addWidget(label(caption), 0, column)
        fields.addWidget(reuse(spin), 1, column)
    settings_layout.addLayout(fields)
    actions = QHBoxLayout()
    actions.addWidget(reuse(ui.import_button, "导入权重"))
    if not compare:
        actions.addWidget(reuse(ui.src_result, "类别统计"))
        actions.addWidget(reuse(ui.src_table, "结果表格"))
    actions.addStretch()
    actions.addWidget(reuse(ui.save_status_button, "保存推理输出"))
    actions.addWidget(reuse(ui.save_button, "导出结果"))
    settings_layout.addLayout(actions)
    layout.addWidget(settings)

    status = QHBoxLayout()
    message = reuse(ui.message_bar)
    message.setObjectName("statusText")
    message.setWordWrap(True)
    status.addWidget(message, 1)
    progress = reuse(ui.progress_bar)
    progress.setFixedWidth(124)
    status.addWidget(progress)
    layout.addLayout(status)
    controller.setCentralWidget(view)
