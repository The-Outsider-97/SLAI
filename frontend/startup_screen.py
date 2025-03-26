from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QMovie
import sys

class StartupScreen(QWidget):
    def __init__(self, on_finish_callback):
        super().__init__()
        self.on_finish_callback = on_finish_callback
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QVBoxLayout()
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        # Load GIF animation
        self.movie = QMovie("Frontend/assets/launch_animation.gif")  # Place your gif here
        self.label.setMovie(self.movie)
        self.movie.start()

        layout.addWidget(self.label)
        self.setLayout(layout)
        self.resize(600, 400)
        self.center_on_screen()

        # Timer: change to your desired splash duration (ms)
        QTimer.singleShot(3500, self.finish)

    def center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def finish(self):
        self.close()
        self.on_finish_callback()
