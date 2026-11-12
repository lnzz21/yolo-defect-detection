import sys
import os
import logging
# 将ui目录添加到系统路径中
sys.path.append(os.path.join(os.getcwd(), "ui"))
# 禁止标准输出
sys.stdout = open(os.devnull, 'w')
logging.disable(logging.CRITICAL)  # 禁用所有级别的日志
from PySide6.QtWidgets import QApplication
from utils import glo
from workstation.desktop import DesktopWindow
from workstation.i18n import language

if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName("SteelScope")
    app.setStyle("Fusion")
    glo._init()
    language()
    window = DesktopWindow()
    app.setWindowIcon(window.windowIcon())
    window.show()
    sys.exit(app.exec())
