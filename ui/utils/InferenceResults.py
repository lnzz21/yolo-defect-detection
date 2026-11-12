"""Table and category distribution for the same inference report."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QGridLayout, QHeaderView, QLabel, QProgressBar, QScrollArea,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from workstation.i18n import bind, label, language, msg, tr


class InferenceResults(QDialog):
    def __init__(self, report, chart, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.report = report
        bind(self, "类别统计" if chart else "结果表格", setter="setWindowTitle")
        self.resize(670 if chart else 830, 440)
        layout = QVBoxLayout(self)
        if "state" in report:
            state = {"completed": "已完成", "stopped": "已停止", "failed": "失败"}[report["state"]]
            layout.addWidget(bind(QLabel(), "{state} · {files} 个文件或来源 · {frames} 张图或帧 · {boxes} 个预测框",
                                  state=msg(state), files=len(report["files"]),
                                  frames=report["processed_frames"], boxes=report["total_boxes"]))
            layout.addWidget(label("视频和相机的框数按已处理帧累计。"))
        if report.get("error"):
            error = bind(QLabel(), "推理失败：{error}", error=report["error"])
            error.setWordWrap(True)
            layout.addWidget(error)
        if report.get("summary_path"):
            path = QLabel(report["summary_path"])
            path.setTextFormat(Qt.TextFormat.PlainText)
            path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            path.setWordWrap(True)
            layout.addWidget(path)
        if chart:
            self.add_chart(layout)
        else:
            self.add_table(layout)

    def add_chart(self, layout):
        total = self.report["total_boxes"]
        if total == 0:
            layout.addWidget(label("本次任务没有处理图像。" if self.report.get("files") == [] else "已处理的图像中未检出目标。"))
            layout.addStretch()
            return
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(len(self.report["counts"]), 1)
        for row, (name, count) in enumerate(sorted(self.report["counts"].items(), key=lambda entry: (-entry[1], entry[0]))):
            grid.addWidget(label(name), row, 0)
            bar = QProgressBar()
            bar.setRange(0, total)
            bar.setValue(count)
            bar.setFormat(f"{100 * count / total:.1f}%")
            grid.addWidget(bar, row, 1)
            grid.addWidget(bind(QLabel(), "{count} 个框", count=count), row, 2)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

    def add_table(self, layout):
        self.table = QTableWidget(len(self.report["files"]), 4)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        header = self.table.horizontalHeader()
        for column in (0, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        for index, row in enumerate(self.report["files"]):
            path = row["path"]
            title = path if "://" in path else Path(path).name
            item = QTableWidgetItem(title)
            item.setToolTip(path)
            self.table.setItem(index, 0, item)
            self.table.setItem(index, 1, QTableWidgetItem(str(row["frames"])))
            self.table.setItem(index, 2, QTableWidgetItem(str(sum(row["counts"].values()))))
            self.table.setItem(index, 3, QTableWidgetItem())
        self.translate_table()
        language().changed.connect(self.translate_table)
        layout.addWidget(self.table, 1)

    def translate_table(self):
        self.table.setHorizontalHeaderLabels([tr(name) for name in ("文件或来源", "图像或帧", "预测框数", "各类别框数")])
        for index, row in enumerate(self.report["files"]):
            counts = ", ".join(f"{tr(name)} × {count}" for name, count in sorted(row["counts"].items()))
            item = self.table.item(index, 3)
            item.setText(counts or tr("未检出目标"))
            item.setToolTip(item.text())
