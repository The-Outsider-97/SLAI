from PyQt5.QtWidgets import QApplication
from torch.utils.data import DataLoader, TensorDataset

import os
import sys
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
    qt_app = QApplication(sys.argv)

    def launch_main_window():
        # Keep a persistent reference to prevent garbage collection
        qt_app.main_win = MainWindow(log_queue=log_queue, metric_queue=metric_queue)
        qt_app.main_win.show()

    splash = StartupScreen(on_finish_callback=launch_main_window)
    splash.show()

    qt_app.exec_()

# === Entry Point ===
if __name__ == "__main__":
    logger.info("Launching SLAI UI...")
    launch_ui()
