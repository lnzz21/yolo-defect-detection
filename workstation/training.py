"""Qt process controller for workbench training."""

import json
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from workstation.store import PROJECT, now, read_json


class TrainingController(QObject):
    message = Signal(str)
    event = Signal(dict)
    completed = Signal(dict)

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self.process = None
        self.directory = None
        self.stdout_tail = b""

    @property
    def busy(self):
        return self.process is not None

    def start(self, settings):
        if self.busy:
            raise RuntimeError("当前已有训练任务。")
        self.directory = self.store.create_job(settings)
        self.stdout_tail = b""
        process = QProcess(self)
        self.process = process
        process.setProgram(sys.executable)
        process.setArguments(["-u", "-m", "workstation.train_worker", "--job", str(self.directory)])
        process.setWorkingDirectory(str(PROJECT))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self.read_stdout)
        process.readyReadStandardError.connect(self.read_stderr)
        process.finished.connect(self.finished)
        process.errorOccurred.connect(self.error)
        process.start()
        return self.directory

    def stop(self):
        if self.busy:
            (self.directory / "stop.request").touch()
            self.message.emit("已请求停止，当前轮次验证和权重保存后退出。")

    def log(self, data):
        with (self.directory / "train.log").open("ab") as file:
            file.write(data)

    def read_stdout(self):
        data = bytes(self.process.readAllStandardOutput())
        self.log(data)
        lines = (self.stdout_tail + data).split(b"\n")
        self.stdout_tail = lines.pop()
        for raw in lines:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line.startswith("@workstation "):
                try:
                    self.event.emit(json.loads(line[len("@workstation "):]))
                    continue
                except ValueError:
                    pass
            if line:
                self.message.emit(line)

    def read_stderr(self):
        data = bytes(self.process.readAllStandardError())
        self.log(data)
        self.message.emit(data.decode("utf-8", errors="replace").rstrip())

    def error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.finished(-1, QProcess.ExitStatus.CrashExit)

    def finished(self, code, status):
        process = self.process
        self.read_stdout()
        self.read_stderr()
        record = read_json(self.directory / "run.json")
        if record["state"] in {"starting", "running"} or (
                record["state"] == "completed" and (code != 0 or status != QProcess.ExitStatus.NormalExit)):
            record = self.store.update_job(self.directory, state="failed", ended_at=now(),
                                           error=f"训练进程退出 ({code})。{process.errorString()}")
        self.process = None
        process.deleteLater()
        self.completed.emit(record)
