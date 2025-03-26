from PyQt5.QtWidgets import QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt, QTimer
import os
import json

class MainWindow(QWidget):
    def __init__(self, log_queue=None, metric_queue=None):
        super().__init__()

        self.log_queue = log_queue
        self.metric_queue = metric_queue
        self.latest_metrics = {}

        self.setWindowTitle("SLAI Launcher")
        self.setStyleSheet("background-color: #101010; color: white;")
        self.setGeometry(200, 100, 1280, 720)

        self.initUI()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_loop)
        self.update_timer.start(500)

    def initUI(self):
        header_font = QFont("Courier", 14, QFont.Bold)
        text_font = QFont("Courier", 10)

        # === Top Controls ===
        self.module_select = QComboBox()
        self.module_select.addItems([
            "EvolutionaryAgent", "BasicRLAgent", "EvolutionaryDQNAgent",
            "MultiTaskAgent", "MetaLearning", "RSI", "RL", "SafeAI", "Collaborative"
        ])

        self.launch_btn = QPushButton("Launch")
        self.launch_btn.setStyleSheet("background-color: #1e90ff; color: white; padding: 5px;")

        self.stop_btn = QPushButton("Stop Agent")
        self.stop_btn.setStyleSheet("background-color: #660000; color: white; padding: 5px;")

        self.save_logs_btn = QPushButton("Save Logs")
        self.save_metrics_btn = QPushButton("Save Metrics")

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Module:"))
        top_layout.addWidget(self.module_select)
        top_layout.addWidget(self.launch_btn)
        top_layout.addStretch()
        top_layout.addWidget(self.stop_btn)
        top_layout.addWidget(self.save_logs_btn)
        top_layout.addWidget(self.save_metrics_btn)

        # === Log Output Panel ===
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setFont(text_font)
        self.log_panel.setStyleSheet("background-color: black;")

        # === Visual Output Panel (Right) ===
        self.reward_img = QLabel("Right Visual Output")
        self.reward_img.setAlignment(Qt.AlignCenter)
        self.reward_img.setStyleSheet("background-color: black;")
        self.reward_img.setFixedWidth(640)

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.log_panel)
        output_layout.addWidget(self.reward_img)

        # === Main Layout ===
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(output_layout)
        self.setLayout(main_layout)

        # Connect actions
        self.stop_btn.clicked.connect(self.stop_agent)
        self.save_logs_btn.clicked.connect(self.save_logs)
        self.save_metrics_btn.clicked.connect(self.save_metrics)

    def stop_agent(self):
        if self.log_queue:
            self.log_queue.put("[USER] Stop requested.")
        # Add shared memory signal if needed

    def save_logs(self):
        os.makedirs("logs", exist_ok=True)
        with open("logs/ui_run_log.txt", "w") as f:
            f.write(self.log_panel.toPlainText())
        self.log_queue.put("[✓] Logs saved to logs/ui_run_log.txt")

    def save_metrics(self):
        os.makedirs("logs", exist_ok=True)
        with open("logs/ui_metrics.json", "w") as f:
            json.dump(self.latest_metrics, f, indent=2)
        self.log_queue.put("[✓] Metrics saved to logs/ui_metrics.json")

    def update_loop(self):
        if self.log_queue:
            while not self.log_queue.empty():
                try:
                    line = self.log_queue.get_nowait()
                    self.log_panel.append(line)
                except Exception:
                    pass

        if self.metric_queue:
            while not self.metric_queue.empty():
                try:
                    metrics = self.metric_queue.get_nowait()
                    self.latest_metrics = metrics
                    self.update_reward_plot()
                except Exception:
                    pass

    def update_reward_plot(self):
        if os.path.exists("outputs/reward_trend.png"):
            pixmap = QPixmap("outputs/reward_trend.png")
            self.reward_img.setPixmap(pixmap.scaled(self.reward_img.width(), self.reward_img.height(), Qt.KeepAspectRatio))
