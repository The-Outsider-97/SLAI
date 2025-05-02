
from PyQt5.QtWidgets import QTextEdit, QSizePolicy
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import Qt, pyqtSignal

class TextEditor(QTextEdit):
    heightChanged = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.setMaximumHeight(150)
        self.document().documentLayout().documentSizeChanged.connect(self.updateHeight)
        self.textChanged.connect(self.keep_cursor_bottom)

    def updateHeight(self):
        doc_height = self.document().size().height()
        margins = self.contentsMargins()
        new_height = int(doc_height + margins.top() + margins.bottom() + 5)  # Basic padding
        min_h = 200
        max_h = 1000
        # max_h = min(350, self.parent().parent().maximumHeight())
        final_height = min(new_height, max_h)

        self.setFixedHeight(final_height)
        self.heightChanged.emit(final_height)

    def keep_cursor_bottom(self):
        """Ensures text stays anchored to bottom"""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)
