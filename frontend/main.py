import os
import sys
import torch
import queue
import logging
import tkinter as tk
from tkinter import font

class SLAIInterface:
    def __init__(self, root, log_queue, metric_queue):
        self.root = root
        self.log_queue = log_queue
        self.metric_queue = metric_queue

        root.title("SLAI Launcher")
        root.configure(bg="#101010")
        root.geometry("1280x720")
        root.resizable(False, False)

        # Fonts
        header_font = font.Font(family="Courier", size=12, weight="bold")
        text_font = font.Font(family="Courier", size=10)

        # Terminal Panel (Left)
        self.terminal = tk.Text(root, bg="black", fg="white", insertbackground="white",
                                font=text_font, wrap="word", borderwidth=0, width=80)
        self.terminal.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.terminal.configure(state="disabled")

        # Metrics Panel (Right)
        self.metrics_label = tk.Label(root, text="[Results Visualized]",
                                      bg="black", fg="white", font=header_font, width=50, height=40, anchor="center")
        self.metrics_label.pack(side="right", fill="both", expand=True)

        self._insert_initial_text()
        self._update_loop()

    def _insert_initial_text(self):
        self._append_terminal(\"\"\"\n==============================\n    SLAI Main Launcher Menu\n==============================\n
Select a module to run:

1 - Evolutionary Agent (Current main.py logic)
2 - Basic RL Agent (CartPole DQN)                       --> main_cartpole.py
3 - Evolutionary DQN Agent                              --> main_cartpole_evolve.py
4 - Multi-Task RL Agent                                 --> main_multitask.py
5 - Meta-Learning Agent (MAML)                          --> main_maml.py
6 - Recursive Self-Improvement (RSI)                    --> main_rsi.py
7 - RL Agent                                            --> main_autotune.py
8 - Safe AI Agent                                       --> main_safe_ai.py
9 - Collaborative Agents (Task Routing, Shared Memory)  --> collaborative.main_collaborative.py

0 - Exit
        \"\"\")

    def _append_terminal(self, text):
        self.terminal.configure(state=\"normal\")
        self.terminal.insert(\"end\", text + \"\\n\")
        self.terminal.see(\"end\")
        self.terminal.configure(state=\"disabled\")

    def _update_loop(self):
        try:
            while not self.log_queue.empty():
                line = self.log_queue.get_nowait()
                self._append_terminal(line)

            while not self.metric_queue.empty():
                metrics = self.metric_queue.get_nowait()
                formatted = \"\\n\".join(f\"{k}: {v:.2f}\" for k, v in metrics.items())
                self.metrics_label.config(text=formatted)

        except queue.Empty:
            pass

        self.root.after(500, self._update_loop)

def launch_ui(log_queue, metric_queue):
    root = tk.Tk()
    app = SLAIInterface(root, log_queue, metric_queue)
    root.mainloop()
