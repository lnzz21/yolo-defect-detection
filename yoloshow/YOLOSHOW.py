from utils import glo
import json
import os
from uuid import uuid4
from ui.YOLOSHOWUI import Ui_MainWindow
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox
from yoloshow.YOLOThreadPool import YOLOThreadPool
from PySide6.QtCore import QTimer, Qt
from PySide6 import QtCore, QtGui
from yoloshow.YOLOSHOWBASE import YOLOSHOWBASE, MODEL_THREAD_CLASSES
from workstation.i18n import button, localize_legacy_window, tr

GLOBAL_WINDOW_STATE = True
WIDTH_LEFT_BOX_STANDARD = 80
WIDTH_LEFT_BOX_EXTENDED = 200
WIDTH_LOGO = 60
UI_FILE_PATH = "ui/YOLOSHOWUI.ui"
KEYS_LEFT_BOX_MENU = ['src_menu', 'src_setting', 'src_webcam', 'src_folder', 'src_camera', 'src_vsmode', 'src_setting']


# YOLOSHOW窗口类 动态加载UI文件 和 Ui_mainWindow
class YOLOSHOW(QMainWindow, YOLOSHOWBASE):
    def __init__(self, *, embedded=False):
        super().__init__()
        self.current_model = None
        self.current_workpath = os.getcwd()
        self.inputPath = None
        self.result_statistic = None
        self.detect_result = None
        self.inference_result = None
        self.report_run_id = None
        self.result_dialog = None

        # --- 加载UI --- #
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._workstation = None
        if not embedded:
            self.workstation_button = button("钢材缺陷工作台", self.ui.topbox)
            self.workstation_button.setStyleSheet("QPushButton { background: #164b63; color: white; padding: 7px 14px; border-radius: 5px; }")
            self.ui.topBox.insertWidget(1, self.workstation_button)
            localize_legacy_window(self)
            self.workstation_button.clicked.connect(self.open_workstation)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setWindowFlags(Qt.FramelessWindowHint)
            self.initSiderWidget()
        # --- 加载UI --- #

        # --- 最大化 最小化 关闭 --- #
        self.ui.maximizeButton.clicked.connect(self.maxorRestore)
        self.ui.minimizeButton.clicked.connect(self.showMinimized)
        self.ui.closeButton.clicked.connect(self.close)
        self.ui.topbox.doubleClickFrame.connect(self.maxorRestore)
        # --- 最大化 最小化 关闭 --- #

        # --- 播放 暂停 停止 --- #
        self.playIcon = QtGui.QIcon()
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/play.png"), QtGui.QIcon.Normal,
                                QtGui.QIcon.Off)
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/pause.png"), QtGui.QIcon.Active,
                                QtGui.QIcon.On)
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/pause.png"),
                                QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.ui.run_button.setCheckable(True)
        self.ui.run_button.setIcon(self.playIcon)
        # --- 播放 暂停 停止 --- #

        # --- 侧边栏缩放 --- #
        self.ui.src_menu.clicked.connect(self.scaleMenu)  # hide menu button
        self.ui.src_setting.clicked.connect(self.scalSetting)  # setting button
        # --- 侧边栏缩放 --- #

        # --- 自动加载/动态改变 PT 模型 --- #
        self.pt_Path = f"{self.current_workpath}/ptfiles/"
        os.makedirs(self.pt_Path, exist_ok=True)
        self.pt_list = self.getModelFiles()
        self.ui.model_box.clear()
        self.ui.model_box.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.loadModels())
        self.qtimer_search.start(2000)
        self.ui.model_box.currentTextChanged.connect(self.changeModel)
        # --- 自动加载/动态改变 PT 模型 --- #

        # --- 导入 图片/视频、调用摄像头、导入文件夹（批量处理）、调用网络摄像头、结果统计图片、结果统计表格 --- #
        self.ui.src_img.clicked.connect(self.selectFile)
        self.ui.src_webcam.clicked.connect(self.selectWebcam)
        self.ui.src_folder.clicked.connect(self.selectFolder)
        self.ui.src_camera.clicked.connect(self.selectRtsp)
        self.ui.src_result.clicked.connect(self.showResultStatics)
        self.ui.src_table.clicked.connect(self.showTableResult)
        # --- 导入 图片/视频、调用摄像头、导入文件夹（批量处理）、调用网络摄像头 --- #

        # --- 导入模型、 导出结果 --- #
        self.ui.import_button.clicked.connect(self.importModel)
        self.ui.save_status_button.clicked.connect(self.saveStatus)
        self.ui.save_button.clicked.connect(self.saveResult)
        self.ui.save_button.setEnabled(False)
        # --- 导入模型、 导出结果 --- #

        # --- 视频、图片 预览 --- #
        self.ui.main_leftbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.ui.main_rightbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        # --- 视频、图片 预览 --- #

        # --- 状态栏 初始化 --- #
        # 状态栏阴影效果
        self.shadowStyle(self.ui.mainBody, QColor(0, 0, 0, 38), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Class_QF, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.classesLabel, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Target_QF, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.targetLabel, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Fps_QF, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.fpsLabel, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Model_QF, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.modelLabel, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        # 状态栏默认显示
        self.model_name = self.ui.model_box.currentText()  # 获取默认 model
        self.ui.Class_num.setText('--')
        self.ui.Target_num.setText('--')
        self.ui.fps_label.setText('--')
        self.ui.Model_label.setText(str(self.model_name).replace(".pt", ""))
        # --- 状态栏 初始化 --- #

        self.initThreads()

        # --- 超参数调整 --- #
        self.ui.iou_spinbox.valueChanged.connect(
            lambda x: self.changeValue(x, 'iou_spinbox'))  # iou box
        self.ui.iou_slider.valueChanged.connect(lambda x: self.changeValue(x, 'iou_slider'))  # iou scroll bar
        self.ui.conf_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'conf_spinbox'))  # conf box
        self.ui.conf_slider.valueChanged.connect(lambda x: self.changeValue(x, 'conf_slider'))  # conf scroll bar
        self.ui.speed_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'speed_spinbox'))  # speed box
        self.ui.speed_slider.valueChanged.connect(lambda x: self.changeValue(x, 'speed_slider'))  # speed scroll bar
        self.ui.line_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'line_spinbox'))  # line box
        self.ui.line_slider.valueChanged.connect(lambda x: self.changeValue(x, 'line_slider'))  # line slider
        # --- 超参数调整 --- #

        # --- 开始 / 停止 --- #
        self.ui.run_button.clicked.connect(self.runorContinue)
        self.ui.stop_button.clicked.connect(self.stopDetect)
        # --- 开始 / 停止 --- #

        # --- Setting栏 初始化 --- #
        self.loadConfig()
        # --- Setting栏 初始化 --- #

        # --- MessageBar Init --- #
        self.showStatus("Welcome to YOLOSHOW")
        # --- MessageBar Init --- #

    def initThreads(self):
        self.yolo_threads = YOLOThreadPool()
        # 获取当前Model
        model_name = self.checkCurrentModel()
        if model_name:
            self.yolo_threads.set(model_name, MODEL_THREAD_CLASSES[model_name]())
            self.initModel(yoloname=model_name)

    def open_workstation(self):
        if self._workstation is None:
            from workstation.window import WorkstationWindow

            try:
                self._workstation = WorkstationWindow(self, legacy_busy=self.legacy_detection_running)
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, tr("无法打开钢材缺陷工作台"), str(error))
                return
        self._workstation.show()
        self._workstation.raise_()
        self._workstation.activateWindow()

    def legacy_detection_running(self):
        for window in (self, glo.get_value("yoloshowvs")):
            if window and window.detection_running():
                return True
        return False


    # 导出结果
    def saveResult(self):
        if not any(thread.res_status for thread in self.yolo_threads.threads_pool.values()):
            self.showStatus("Please select the Image/Video before starting detection...")
            return
        config_file = f'{self.current_workpath}/config/save.json'
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        save_path = config.get('save_path', os.getcwd())
        is_folder = isinstance(self.inputPath, list)
        if is_folder:
            self.OutputDir = QFileDialog.getExistingDirectory(
                self,  # 父窗口对象
                tr("Save Results in new Folder"),  # 标题
                save_path,  # 起始目录
            )
            current_model_name = self.checkCurrentModel()
            self.saveResultProcess(self.OutputDir, current_model_name, folder=True)
        else:
            self.OutputDir, _ = QFileDialog.getSaveFileName(
                self,  # 父窗口对象
                tr("Save Image/Video"),  # 标题
                save_path,  # 起始目录
                tr("Image/Vide Type (*.jpg *.jpeg *.png *.bmp *.dib  *.jpe  *.jp2 *.mp4)")
            )
            current_model_name = self.checkCurrentModel()
            self.saveResultProcess(self.OutputDir, current_model_name, folder=False)

        config['save_path'] = self.OutputDir
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)

    # 加载 pt 模型到 model_box
    def loadModels(self):
        if self.report_run_id is not None:
            return
        pt_list = self.getModelFiles()

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.refreshModelBox(self.ui.model_box, self.pt_list)

    def stopOtherModelProcess(self, model_name, current_yoloname):
        yolo_thread = self.yolo_threads.get(model_name)
        yolo_thread.finished.connect(lambda: self.resignModel(current_yoloname))
        yolo_thread.stop_dtc = True
        self.yolo_threads.stop_thread(model_name)

    # 停止其他模型
    def stopOtherModel(self, current_yoloname=None):
        for model_name in self.yolo_threads.threads_pool.keys():
            if not current_yoloname or model_name == current_yoloname:
                continue
            if self.yolo_threads.get(model_name).isRunning():
                self.stopOtherModelProcess(model_name, current_yoloname)

    # 重新加载模型
    def resignModel(self, model_name):
        # 重载 common 和 yolo 模块
        glo.set_value('yoloname', model_name)
        self.reloadModel()
        self.yolo_threads.set(model_name, MODEL_THREAD_CLASSES[model_name]())
        self.initModel(yoloname=model_name)
        self.runModel(True)

    # Model 变化
    def changeModel(self):
        self.model_name = self.ui.model_box.currentText()
        self.ui.Model_label.setText(str(self.model_name).replace(".pt", ""))  # 修改状态栏显示
        model_name = self.checkCurrentModel()
        if not model_name:
            self.showStatus(self.unsupportedModelMessage(self.model_name))
            return
        # 停止其他模型
        self.stopOtherModel(model_name)
        yolo_thread = self.yolo_threads.get(model_name)
        if yolo_thread is not None:
            yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box.currentText()
        else:
            self.yolo_threads.set(model_name, MODEL_THREAD_CLASSES[model_name]())
            self.initModel(yoloname=model_name)
            self.loadConfig()
        now_model_name = str(self.model_name).replace(".pt", "")
        self.showStatus(f"Change Model to {now_model_name} Successfully")

    def runModelProcess(self, yolo_name):
        yolo_thread = self.yolo_threads.get(yolo_name)
        if self.ui.run_button.isChecked():
            if not yolo_thread.isRunning():
                if self.report_run_id is not None:
                    self.ui.run_button.setChecked(False)
                    return
                self.inference_result = self.result_statistic = self.detect_result = None
                if self.result_dialog is not None:
                    self.result_dialog.close()
                yolo_thread.source = self.inputPath.copy() if isinstance(self.inputPath, list) else self.inputPath
                yolo_thread.stop_dtc = False
                if hasattr(yolo_thread, "send_report"):
                    self.report_run_id = yolo_thread.report_id = uuid4().hex
                    yolo_thread.save_res = self.ui.save_status_button.isChecked()
                    self.enable_report_inputs(False)
            yolo_thread.is_continue = True
            self.yolo_threads.start_thread(yolo_name)
        else:
            yolo_thread.is_continue = False
            self.showStatus('Pause Detection')

    def runModel(self, runbuttonStatus=None):
        if self._workstation is not None and self._workstation.busy():
            self.ui.run_button.setChecked(False)
            self.showStatus("请先结束钢材缺陷工作台中的任务。")
            return
        self.ui.save_status_button.setEnabled(False)
        if runbuttonStatus:
            self.ui.run_button.setChecked(True)
        current_model_name = self.checkCurrentModel()
        if current_model_name is not None:
            self.runModelProcess(current_model_name)
        else:
            self.showStatus(self.unsupportedModelMessage(self.model_name))
            self.ui.save_status_button.setEnabled(True)
            if self.ui.run_button.isChecked():
                self.ui.run_button.setChecked(False)

    # 开始/暂停 预测
    def runorContinue(self):
        if self.inputPath is not None:
            if self.report_run_id is None:
                self.changeModel()
            self.runModel()
        else:
            self.showStatus("Please select the Image/Video before starting detection...")
            self.ui.run_button.setChecked(False)

    def enable_report_inputs(self, enabled):
        for widget in (self.ui.model_box, self.ui.src_img, self.ui.src_folder, self.ui.src_webcam, self.ui.src_camera):
            widget.setEnabled(enabled)

    def setInferenceReport(self, report):
        if report["id"] != self.report_run_id:
            return
        self.inference_result = report
        self.report_run_id = None
        self.enable_report_inputs(True)
        self.showStatus("Stop Detection" if report["state"] == "stopped" else "Finish Detection")
        if report["state"] == "failed":
            bind(self.ui.message_bar, "推理失败：{error}", error=report["error"])
        if "save_error" in report:
            bind(self.ui.message_bar, "无法保存统计文件：{error}", error=report["save_error"])

    # 停止识别
    def stopDetect(self):
        self.quitRunningModel(stop_status=True)
        self.ui.run_button.setChecked(False)
        self.ui.save_status_button.setEnabled(True)
        self.ui.progress_bar.setValue(0)
        self.ui.main_leftbox.clear()  # clear image display
        self.ui.main_rightbox.clear()
        self.ui.Class_num.setText('--')
        self.ui.Target_num.setText('--')
        self.ui.fps_label.setText('--')
