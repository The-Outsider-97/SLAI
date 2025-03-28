from PyQt5.QtWidgets import QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QFrame, QSizePolicy
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
import os
import sys
import json
import threading
import subprocess
import psutil
import GPUtil

class MainWindow(QWidget):
    log_signal = pyqtSignal(str)

    def __init__(self, log_queue=None, metric_queue=None):
        super().__init__()

        self.log_queue = log_queue
        self.metric_queue = metric_queue
        self.latest_metrics = {}
        self.last_image_timestamp = None

        self.setWindowTitle("SLAI Launcher")
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "assets", "logo.ico")))
        self.setStyleSheet("background-color: #0e0e0e; color: #eaedec;")
        self.setGeometry(200, 100, 1280, 720)

        self.initUI()
        self.log_signal.connect(self.safe_append_log)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_loop)
        self.update_timer.start(500)

    def safe_append_log(self, line):
        self.log_panel.append(line)
        self.log_panel.moveCursor(self.log_panel.textCursor().End)

    def initUI(self):
        header_font = QFont("Times New Roman", 14, QFont.Bold)
        text_font = QFont("Times New Roman", 10)

        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.ico")
        logo = QLabel()
        logo.setPixmap(QPixmap(logo_path).scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        title_label = QLabel()
        self.typing_texts = ["SLAI", "Self-Learning Autonomous Intelligence"]
        self.typing_index = 0
        self.char_index = 0
        self.display_text = ""
        self.is_pausing = False
        self.pause_counter = 0

        def update_typing():
            if self.is_pausing:
                self.pause_counter += 100
                if self.pause_counter >= 5000:  # 5 seconds pause
                    self.is_pausing = False
                    self.typing_index = (self.typing_index + 1) % len(self.typing_texts)
                    self.char_index = 0
                    self.display_text = ""
                return

            full_text = self.typing_texts[self.typing_index]
            if self.char_index < len(full_text):
                self.display_text += full_text[self.char_index]
                title_label.setText(self.display_text)
                self.char_index += 1
            else:
                self.is_pausing = True
                self.pause_counter = 0

        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(update_typing)
        self.typing_timer.start(100)

        title_label.setFont(QFont("Times New Roman", 14, QFont.Bold))
        title_label.setStyleSheet("color: #eaedec;")


        title_layout = QHBoxLayout()
        title_layout.addWidget(logo)
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        self.module_select = QComboBox()
        self.module_select.addItems([
            "EvolutionaryAgent", "BasicRLAgent", "EvolutionaryDQNAgent",
            "MultiTaskAgent", "MetaLearning", "RSI", "RL", "SafeAI", "Collaborative"
        ])

        self.launch_btn = QPushButton("Launch")
        self.launch_btn.setStyleSheet("background-color: #b99b00; color: #eaedec; padding: 5px;")
        self.launch_btn.clicked.connect(self.launch_module)

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

        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setFont(text_font)
        self.log_panel.setStyleSheet("background-color: black;")

        self.reward_img = QLabel("Right Visual Output")
        self.reward_img.setAlignment(Qt.AlignCenter)
        self.reward_img.setStyleSheet("background-color: black;")
        self.reward_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.hardware_info = QLabel("Hardware Stats Loading...")
        self.hardware_info.setStyleSheet("color: #888888; padding: 4px;")
        self.hardware_info.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.hardware_info.setWordWrap(True)

        right_panel = QVBoxLayout()
        right_panel.addWidget(self.reward_img)
        right_panel.addWidget(self.hardware_info)

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.log_panel, 2)
        output_layout.addLayout(right_panel, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(title_layout)
        main_layout.addLayout(top_layout)
        main_layout.addLayout(output_layout)
        self.setLayout(main_layout)

        self.stop_btn.clicked.connect(self.stop_agent)
        self.save_logs_btn.clicked.connect(self.save_logs)
        self.save_metrics_btn.clicked.connect(self.save_metrics)

    def stop_agent(self):
        if self.log_queue:
            self.log_queue.put("[USER] Stop requested.")

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
                except Exception:
                    pass

        self.update_reward_plot()
        self.update_hardware_info()

        if self.metric_queue:
            while not self.metric_queue.empty():
                try:
                    metrics = self.metric_queue.get_nowait()
                    self.latest_metrics = metrics
                except Exception:
                    pass

    def update_reward_plot(self):
        for path in ["outputs/reward_trend.png", "logs/reward_trend.png"]:
            if os.path.exists(path):
                timestamp = os.path.getmtime(path)
                if timestamp != self.last_image_timestamp:
                    self.last_image_timestamp = timestamp
                    image = QImage(path)
                    image.invertPixels()
                    self.reward_img.setPixmap(QPixmap.fromImage(image).scaled(
                        self.reward_img.width(), self.reward_img.height(), Qt.KeepAspectRatio))
                return

    def update_hardware_info(self):
        cpu_freq = psutil.cpu_freq()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        processes = len(psutil.pids())

        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_info = f"GPU: {gpu.name} | Usage: {gpu.load*100:.1f}% | Temp: {gpu.temperature}°C"
        else:
            gpu_info = "GPU: N/A"

        ram_stats = ram._asdict()
        cached_mb = ram_stats.get('cached', 0) // (1024 ** 2)

        text = (
            f"CPU: {cpu_freq.current:.1f}MHz | Usage: {cpu_usage:.1f}% | "
            f"Threads: {psutil.cpu_count(logical=True)} | Processes: {processes}\n"
            f"RAM: {ram.percent:.1f}% Used, {ram.available // (1024 ** 2)}MB Available, {cached_mb}MB Cached\n"
            f"{gpu_info}"
        )
        self.hardware_info.setText(text)

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
            return

        def run_agent():
            self.log_signal.emit(f"[✓] Launching {script_path}...")
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            for line in process.stdout:
                self.log_signal.emit(line.strip())
            process.wait()
            self.log_signal.emit(f"[✓] Execution finished.")

        threading.Thread(target=run_agent, daemon=True).start()

# ========== External Panel Update Hooks ==========
def update_visual_output_panel(image_paths):
    """
    Allows external agents to update the right visual panel with reward plots, etc.
    """
    if not image_paths:
        return
    for path in image_paths:
        if os.path.exists(path):
            try:
                with open("logs/ui_display.txt", "w") as f:
                    f.write(path)
            except Exception as e:
                print(f"[ERROR] Failed to update visual output: {e}")
            break  # Only show one image (first found)

def update_text_output_panel(summary):
    """
    Allows external agents to update the left log panel with performance summaries.
    """
    try:
        with open("logs/ui_text.txt", "w") as f:
            f.write(summary)
    except Exception as e:
        print(f"[ERROR] Failed to update text output: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
