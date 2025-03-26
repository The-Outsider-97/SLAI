from PyQt5.QtWidgets import QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
import os
import sys
import json
import threading
import subprocess

def launch_module(self):
    module_map = {
        "EvolutionaryAgent": "main.py",
        "BasicRLAgent": "main_cartpole.py",
        "EvolutionaryDQNAgent": "main_cartpole_evolve.py",
        "MultiTaskAgent": "main_multitask.py",
        "MetaLearning": "main_maml.py",
        "RSI": "main_rsi.py",
        "RL": "main_autotune.py",
        "SafeAI": "main_safe_ai.py",
        "Collaborative": "collaborative/main_collaborative.py"
    }

    module_key = self.module_select.currentText()
    script_path = module_map.get(module_key)

    if not script_path or not os.path.exists(script_path):
        self.log_signal.emit(f"[ERROR] Cannot find script for module: {module_key}")
        self.log_panel.moveCursor(self.log_panel.textCursor().End)
        return

    def run_agent():
        self.log_signal.emit(f"[✓] Launching {script_path}...\n")
        self.log_panel.moveCursor(self.log_panel.textCursor().End)
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        for line in process.stdout:
            self.log_signal.emit(line.strip())
            self.log_panel.moveCursor(self.log_panel.textCursor().End)
        process.wait()
        self.log_signal.emit(f"\n[✓] Execution finished.\n")
        self.log_panel.moveCursor(self.log_panel.textCursor().End)

    threading.Thread(target=run_agent, daemon=True).start()

class MainWindow(QWidget):
    log_signal = pyqtSignal(str)

    def __init__(self, log_queue=None, metric_queue=None):
        super().__init__()

        self.log_queue = log_queue
        self.metric_queue = metric_queue
        self.latest_metrics = {}

        self.setWindowTitle("SLAI Launcher")
        self.setStyleSheet("background-color: #0e0e0e; color: #eaedec;")
        self.setGeometry(200, 100, 1280, 720)

        self.initUI()
        self.log_signal.connect(self.safe_append_log)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_loop)
        self.update_timer.start(500)

    def safe_append_log(self, line):
        self.log_signal.emit(line)
        self.log_panel.moveCursor(self.log_panel.textCursor().End)

    def initUI(self):
        header_font = QFont("Times New Roman", 14, QFont.Bold)
        text_font = QFont("Times New Roman", 10)

        # === Top Controls ===
        self.module_select = QComboBox()
        self.module_select.addItems([
            "EvolutionaryAgent", "BasicRLAgent", "EvolutionaryDQNAgent",
            "MultiTaskAgent", "MetaLearning", "RSI", "RL", "SafeAI", "Collaborative"
        ])

        self.launch_btn = QPushButton("Launch")
        self.launch_btn.setStyleSheet("background-color: #b99b00; color: #eaedec; padding: 5px;")
        self.launch_btn.clicked.connect(lambda: launch_module(self))

        self.stop_btn = QPushButton("Stop Agent")
        self.stop_btn.setStyleSheet("background-color: #431600; color: #eaedec; padding: 5px;")

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
                    self.log_signal.emit(line)
                    self.log_panel.moveCursor(self.log_panel.textCursor().End)
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
