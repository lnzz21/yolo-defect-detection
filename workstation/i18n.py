"""Live UI translations and the workstation's saved language preference."""

from dataclasses import dataclass
from pathlib import Path
import re
from string import Formatter

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QObject, QSignalBlocker, QTranslator, Signal
from PySide6.QtWidgets import QComboBox, QLabel, QMessageBox, QPushButton

from workstation.store import STORAGE, read_json, write_json
from workstation.translations import ENGLISH, ENGLISH_MESSAGES, LEGACY_PREFIXES, SIMPLIFIED_CHINESE, SIMPLIFIED_CHINESE_MESSAGES


def compile_messages(messages):
    patterns = []
    for template, translation in messages.items():
        parts = []
        for literal, field, _, _ in Formatter().parse(template):
            parts.append(re.escape(literal))
            if field:
                parts.append(f"(?P<{field}>.+?)")
        patterns.append((re.compile("".join(parts)), translation))
    return patterns


FORMATTED_MESSAGES = {
    "en": compile_messages(ENGLISH_MESSAGES),
    "zh": compile_messages(SIMPLIFIED_CHINESE_MESSAGES),
}


@dataclass
class Message:
    source: str
    fields: dict


def msg(source, /, **fields):
    return Message(source, fields)


class Language(QObject):
    changed = Signal()

    def __init__(self, app, preferences=STORAGE / "ui_preferences.json"):
        super().__init__(app)
        self.app = app
        self.preferences = preferences
        try:
            self.code = read_json(preferences).get("language", "zh")
        except (OSError, ValueError, AttributeError):
            self.code = "zh"
        if self.code not in {"zh", "en"}:
            self.code = "zh"
        self.qt_chinese = QTranslator(self)
        directory = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        self.qt_chinese.load(str(Path(directory) / "qtbase_zh_CN.qm"))
        if self.code == "zh":
            app.installTranslator(self.qt_chinese)

    def select(self, code):
        if code not in {"zh", "en"} or code == self.code:
            return
        write_json(self.preferences, {"language": code})
        self.code = code
        if code == "zh":
            self.app.installTranslator(self.qt_chinese)
        else:
            self.app.removeTranslator(self.qt_chinese)
        self.changed.emit()


def language():
    app = QCoreApplication.instance()
    if not hasattr(app, "workstation_language"):
        app.workstation_language = Language(app)
    return app.workstation_language


def render(message):
    code = language().code
    catalog = ENGLISH if code == "en" else SIMPLIFIED_CHINESE
    source = message.source
    translated = catalog.get(source, catalog.get(source.strip(), source))
    if not message.fields:
        if translated == source:
            for pattern, result in FORMATTED_MESSAGES[code]:
                match = pattern.fullmatch(source)
                if match:
                    return result.format(**match.groupdict())
            if code == "zh":
                for prefix, chinese in LEGACY_PREFIXES.items():
                    if source.startswith(prefix):
                        return chinese + source[len(prefix):]
        return translated
    fields = {key: render(value) if isinstance(value, Message) else value for key, value in message.fields.items()}
    return translated.format(**fields)


def tr(source, /, **fields):
    return render(msg(source, **fields))


class TextBindings(QObject):
    def __init__(self, owner):
        super().__init__(owner)
        self.messages = {}
        language().changed.connect(self.refresh)

    def refresh(self):
        for setter, message in self.messages.items():
            getattr(self.parent(), setter)(render(message))


def bind(widget, source, /, *, setter="setText", **fields):
    if not hasattr(widget, "workstation_texts"):
        widget.workstation_texts = TextBindings(widget)
    message = msg(source, **fields)
    widget.workstation_texts.messages[setter] = message
    getattr(widget, setter)(render(message))
    return widget


def show_pixmap(widget, pixmap):
    if hasattr(widget, "workstation_texts"):
        widget.workstation_texts.messages.pop("setText", None)
    widget.setPixmap(pixmap)


def label(source):
    return bind(QLabel(), source)


def button(source, parent=None):
    return bind(QPushButton(parent), source)


class LanguageSelector(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.addItem("简体中文", "zh")
        self.addItem("English", "en")
        bind(self, "界面语言", setter="setToolTip")
        self.update_choice()
        self.currentIndexChanged.connect(self.change_language)
        language().changed.connect(self.update_choice)

    def update_choice(self):
        with QSignalBlocker(self):
            self.setCurrentIndex(self.findData(language().code))

    def change_language(self):
        try:
            language().select(self.currentData())
        except OSError as error:
            self.update_choice()
            QMessageBox.warning(self, tr("无法保存语言设置"), str(error))


def localize_legacy_window(window):
    """Bind the generated captions without re-running retranslateUi, which clears previews."""
    names = ("title", "src_menu", "src_img", "src_webcam", "src_folder", "src_camera", "src_result",
             "src_table", "src_vsmode", "src_singlemode", "src_setting", "label_2", "label_5", "label_6",
             "label_7", "label_8", "label_9", "label_10", "label_11", "label_12", "ToggleBotton_2",
             "ToggleBotton_3", "ToggleBotton_4", "ToggleBotton_5", "ToggleBotton_6", "ToggleBotton_7",
             "import_button", "save_status_button", "save_button")
    for name in names:
        widget = getattr(window.ui, name, None)
        if widget is not None and widget.text().strip() in SIMPLIFIED_CHINESE:
            bind(widget, widget.text().strip())
    bind(window, "YOLOSHOW", setter="setWindowTitle")
    window.language_selector = LanguageSelector(window.ui.topbox)
    window.ui.topBox.addWidget(window.language_selector)
