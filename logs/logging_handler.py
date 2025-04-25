from PyQt5.QtCore import QObject, pyqtSignal
import logging

class QtLogHandler(logging.Handler, QObject):
    log_message = pyqtSignal(str)

    def __init__(self):
        QObject.__init__(self)
        logging.Handler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_message.emit(msg)
