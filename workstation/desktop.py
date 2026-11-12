"""Single desktop shell for steel inspection and the original YOLOSHOW tools."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QStackedWidget, QVBoxLayout, QWidget,
)

from utils import glo
from workstation.i18n import LanguageSelector, bind, button, label, tr
from workstation.legacy_view import install_inference_view
from workstation.store import PROJECT
from workstation.window import WorkstationWindow
from yoloshow.YOLOSHOW import YOLOSHOW
from yoloshow.YOLOSHOWVS import YOLOSHOWVS


LOGO = PROJECT / "images" / "steelscope.svg"


class DesktopWindow(QMainWindow):
    def __init__(self, store=None):
        super().__init__()
        bind(self, "SteelScope · 钢材缺陷检测", setter="setWindowTitle")
        self.setWindowIcon(QIcon(str(LOGO)))
        self.resize(1380, 840)
        self.setStyleSheet((PROJECT / "ui" / "steelscope.qss").read_text(encoding="utf-8"))
        root = QWidget()
        root.setObjectName("desktop")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tools = {}
        self.entries = {}
        self.current = None
        self.last_tool = None
        self.navigation = QButtonGroup(self)
        self.workstation = WorkstationWindow(self, store=store, legacy_busy=self.legacy_busy, embedded=True)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(188)
        rail = QVBoxLayout(sidebar)
        rail.setContentsMargins(14, 20, 14, 16)
        rail.setSpacing(5)
        brand = QHBoxLayout()
        mark = QLabel()
        mark.setPixmap(QPixmap(str(LOGO)).scaled(38, 38, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        brand.addWidget(mark)
        brand.addWidget(QLabel("SteelScope", objectName="brand"), 1)
        rail.addLayout(brand)
        rail.addSpacing(22)
        self.add_group(rail, "钢材检测流程", (
            ("inspection", "现场检测", "使用已启用模型检测视频或相机，并记录每帧结果。"),
            ("annotation", "采集与标注", "抓取原始帧、审核缺陷框并生成数据集版本。"),
            ("training", "可视化训练", "选择数据与初始权重，跟踪训练曲线和验证指标。"),
            ("library", "训练历史与模型", "查看训练记录，试跑、启用或回退模型。"),
        ))
        rail.addSpacing(16)
        self.add_group(rail, "通用推理工具", (
            ("single", "单模型检测", "使用本地权重检测图片、视频、文件夹或摄像头。"),
            ("compare", "双模型对比", "对同一输入并排查看两个模型的预测。"),
        ))
        rail.addStretch()
        footer = label("本地视觉工作站")
        footer.setObjectName("navGroup")
        rail.addWidget(footer)
        layout.addWidget(sidebar)

        content = QVBoxLayout()
        content.setContentsMargins(22, 15, 22, 15)
        content.setSpacing(12)
        header = QHBoxLayout()
        text = QVBoxLayout()
        text.setSpacing(3)
        self.heading = QLabel(objectName="pageTitle")
        self.description = QLabel(objectName="pageDescription", wordWrap=True)
        text.addWidget(self.heading)
        text.addWidget(self.description)
        header.addLayout(text, 1)
        self.language_selector = LanguageSelector()
        header.addWidget(self.language_selector, 0, Qt.AlignTop)
        content.addLayout(header)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.workstation)
        content.addWidget(self.stack, 1)
        layout.addLayout(content, 1)
        self.workstation.tabs.currentChanged.connect(self.workstation_page_changed)
        self.navigate("inspection")

    def add_group(self, layout, name, entries):
        heading = label(name)
        heading.setObjectName("navGroup")
        layout.addWidget(heading)
        for key, title, description in entries:
            item = button(title)
            item.setObjectName("navItem")
            item.setCheckable(True)
            item.clicked.connect(lambda checked, page=key: self.navigate(page))
            self.navigation.addButton(item)
            self.entries[key] = item, title, description
            layout.addWidget(item)

    def update_navigation(self, key):
        self.current = key
        item, title, description = self.entries[key]
        item.setChecked(True)
        bind(self.heading, title)
        bind(self.description, description)

    def legacy_busy(self):
        return any(view.detection_running() for view in self.tools.values())

    def navigate(self, key):
        if self.current and key != self.current and (
            self.legacy_busy() or (key in {"single", "compare"} and self.workstation.busy())
        ):
            self.entries[self.current][0].setChecked(True)
            QMessageBox.information(self, tr("任务正在运行"), tr("结束当前任务后再切换推理工具。"))
            return
        if key in {"single", "compare"}:
            if key not in self.tools:
                controller = YOLOSHOW(embedded=True) if key == "single" else YOLOSHOWVS(embedded=True)
                glo.set_value("yoloshow" if key == "single" else "yoloshowvs", controller)
                controller._workstation = self.workstation
                install_inference_view(controller, compare=key == "compare")
                self.tools[key] = controller
                self.stack.addWidget(controller)
            self.stack.setCurrentWidget(self.tools[key])
            self.last_tool = self.tools[key]
        else:
            self.workstation.tabs.setCurrentWidget(getattr(self.workstation, key))
            self.stack.setCurrentWidget(self.workstation)
        self.update_navigation(key)

    def workstation_page_changed(self):
        if self.stack.currentWidget() is self.workstation:
            current = self.workstation.tabs.currentWidget()
            for key in ("inspection", "annotation", "training", "library"):
                if current is getattr(self.workstation, key):
                    self.update_navigation(key)
                    return

    def closeEvent(self, event):
        if self.workstation.busy() or self.legacy_busy():
            QMessageBox.information(self, tr("任务正在运行"), tr("结束当前任务后再关闭工作台。"))
            event.ignore()
            return
        if self.last_tool is not None:
            self.last_tool.save_config()
        super().closeEvent(event)
