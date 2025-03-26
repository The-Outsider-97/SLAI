from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QMovie
import os, sys

class StartupScreen(QWidget):
    def __init__(self, on_ready_to_proceed):
        super().__init__()
        self.on_ready_to_proceed = on_ready_to_proceed
        self.ready_to_show = False
        self.minimum_time_elapsed = False
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QVBoxLayout()
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        gif_path = os.path.join(os.path.dirname(__file__), "assets", "startup.png")
        if os.path.exists(gif_path):
            self.movie = QMovie(gif_path)
            self.label.setMovie(self.movie)
            self.movie.start()
        else:
            self.label.setText("Loading SLAI...")
            self.label.setStyleSheet("color: white; font-size: 18px;")

        layout.addWidget(self.label)
        self.setLayout(layout)
        self.resize(2560, 1440)
        self.center_on_screen()

        QTimer.singleShot(10000, self.allow_proceed)  # Wait 10s before allowing transition

    def center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def allow_proceed(self):
        self.minimum_time_elapsed = True
        if self.ready_to_show:
            self.proceed()

    def notify_launcher_ready(self):
        self.ready_to_show = True
        if self.minimum_time_elapsed:
            self.proceed()

    def proceed(self):
        self.close()
        self.on_ready_to_proceed()
