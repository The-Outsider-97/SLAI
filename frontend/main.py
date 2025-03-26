import os
import torch
import logging
import sys, queue
import tkinter as tk
from PyQt5.QtWidgets import QApplication
from startup_screen import StartupScreen
from main_window import MainWindow

# Communication queues (if you're using multiprocessing)
log_queue = queue.Queue()
metric_queue = queue.Queue()

def launch_main_ui():
    main_win = MainWindow(log_queue=log_queue, metric_queue=metric_queue)
    main_win.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Splash screen with animation
    splash = StartupScreen(on_finish_callback=launch_main_ui)
    splash.show()

    sys.exit(app.exec_())
