from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from torch.utils.data import DataLoader, TensorDataset

import os, sys
import yaml
import torch
import queue
import logging
import threading
import subprocess

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# === Logger Setup ===
from logs.logger import get_logger, get_log_queue
from utils.logger import setup_logger
logger = setup_logger("SLAI", level=logging.DEBUG)

log_queue = get_log_queue()
metric_queue = queue.Queue()

# === UI Setup ===
from frontend.startup_screen import StartupScreen
from frontend.main_window import MainWindow

def launch_ui():
    app = QApplication(sys.argv)

    def show_main_window():
        app.main_window = MainWindow(log_queue=log_queue, metric_queue=metric_queue)
        app.main_window.show()

    app.splash_screen = StartupScreen(on_ready_to_proceed=show_main_window)
    app.splash_screen.show()

    # Simulate preload (adjust as needed)
    QTimer.singleShot(2000, app.splash_screen.notify_launcher_ready)

    sys.exit(app.exec_())

# === Entry Point ===
if __name__ == "__main__":
    logger.info("Launching SLAI UI...")
    launch_ui()
