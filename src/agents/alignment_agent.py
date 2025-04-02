import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import json
import threading
import subprocess
import psutil
import GPUtil

class MainWindow(QWidget):
    def __init__(self, log_queue=None, metric_queue=None):  # Add parameters
        super().__init__()
        self.log_queue = log_queue
        self.metric_queue = metric_queue
        self.setWindowTitle("SLAI Launcher")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(80)
        sidebar.setStyleSheet("background-color: #0a0a0a;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        sidebar_layout.setContentsMargins(0, 20, 0, 20)
        sidebar_layout.setSpacing(20)

        # Sidebar buttons
        for i in range(4):
            btn = QPushButton()
            btn.setFixedSize(50, 50)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #FFD700;
                    border-radius: 25px;
                    border: 2px solid #333;
                }
                QPushButton:hover {
                    background-color: #FFE44D;
                }
            """)
            sidebar_layout.addWidget(btn)

        # Bottom system button
        system_btn = QPushButton()
        system_btn.setFixedSize(50, 50)
        system_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                border-radius: 25px;
                border: 2px solid #333;
            }
        """)
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(system_btn)

        main_layout.addWidget(sidebar)

        # Center Panel
        center_panel = QWidget()
        center_panel.setStyleSheet("background-color: black;")
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(20, 20, 20, 20)
        
        # Output label
        output_label = QLabel("Prompt output here...")
        output_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-family: 'Times New Roman';
                font-size: 18px;
                font-style: italic;
            }
        """)
        center_layout.addWidget(output_label)
        
        # Output content (placeholder)
        output_content = QTextEdit()
        output_content.setStyleSheet("""
            QTextEdit {
                background-color: black;
                color: #fff;
                border: 1px solid #333;
                font-family: Arial;
                font-size: 14px;
            }
        """)
        center_layout.addWidget(output_content)
        
        # Input panel
        input_panel = QLineEdit()
        input_panel.setPlaceholderText("Type something here...")
        input_panel.setStyleSheet("""
            QLineEdit {
                background-color: #111;
                color: #fff;
                border: 1px solid #333;
                border-radius: 5px;
                padding: 10px;
                font-family: Arial;
                font-size: 14px;
            }
        """)
        center_layout.addWidget(input_panel)
        
        main_layout.addWidget(center_panel, 1)

        # Right Panel
        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_panel.setStyleSheet("background-color: #0a0a0a;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 20, 10, 20)
        right_layout.setSpacing(10)
        
        # Logs label
        logs_label = QLabel("System process logs...")
        logs_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-family: 'Times New Roman';
                font-size: 16px;
                font-style: italic;
            }
        """)
        right_layout.addWidget(logs_label)
        
        # Logs content
        logs_content = QTextEdit()
        logs_content.setStyleSheet("""
            QTextEdit {
                background-color: black;
                color: #fff;
                border: 1px solid #333;
                font-family: Monospace;
                font-size: 12px;
            }
        """)
        logs_content.setReadOnly(True)
        right_layout.addWidget(logs_content)
        
        # System monitor
        self.system_monitor = QWidget()
        self.system_monitor.setStyleSheet("background-color: #111;")
        monitor_layout = QVBoxLayout(self.system_monitor)
        monitor_layout.setContentsMargins(10, 10, 10, 10)
        
        self.cpu_label = QLabel()
        self.ram_label = QLabel()
        self.gpu_label = QLabel()
        
        for label in [self.cpu_label, self.ram_label, self.gpu_label]:
            label.setStyleSheet("color: #fcfcfc; font-family: Times New Roman; font-size: 12px;")
            monitor_layout.addWidget(label)
        
        right_layout.addWidget(self.system_monitor)
        
        main_layout.addWidget(right_panel)

        # Update system stats
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)
        self.update_stats()

        # Log handling setup
        if self.log_queue:
            self.log_timer = QTimer()
            self.log_timer.timeout.connect(self.update_log_display)
            self.log_timer.start(100)  # Update every 100ms

    def update_log_display(self):
        """Update log display from queue"""
        while not self.log_queue.empty():
            log_entry = self.log_queue.get_nowait()
            self.logs_content.append(log_entry)  # Assuming logs_content is your QTextEdit

    def update_stats(self):
        # CPU
        cpu_percent = psutil.cpu_percent()
        cpu_freq = psutil.cpu_freq().current
        cpu_text = f"CPU: {cpu_freq:.1f} MHz\nUsage: {cpu_percent}%"
        self.cpu_label.setText(cpu_text)

        # RAM
        ram = psutil.virtual_memory()
        ram_text = f"RAM: {ram.percent}% used\n{ram.available/1024/1024:.0f} MB available"
        self.ram_label.setText(ram_text)

        # GPU
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_text = f"GPU: {gpu.name}\nUsage: {gpu.load*100:.1f}%\nTemp: {gpu.temperature}°C"
                self.gpu_label.setText(gpu_text)
        except:
            self.gpu_label.setText("GPU: Not available")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set dark theme palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(16, 16, 16))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
