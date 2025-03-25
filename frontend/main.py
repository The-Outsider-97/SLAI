import os
import sys
import torch
import logging
import tkinter as tk
from tkinter import font

class SLAIInterface:
    def __init__(self, root):
        root.title("SLAI Launcher")
        root.configure(bg="#101010")
        root.geometry("1280x720")
        root.resizable(False, False)

        # Fonts
        header_font = font.Font(family="Times New Roman", size=14, weight="bold")
        text_font = font.Font(family="Times New Roman", size=12)

        # Left Panel (Terminal-like)
        terminal = tk.Frame(root, bg="black", width=640, height=720)
        terminal.pack(side="left", fill="both")

        term_text = tk.Text(terminal, bg="black", fg="white", insertbackground="white",
                            font=text_font, wrap="word", borderwidth=0)
        term_text.pack(fill="both", expand=True, padx=10, pady=10)

        term_text.insert("end", "==============================\n")
        term_text.insert("end", "    SLAI Main Launcher Menu\n")
        term_text.insert("end", "==============================\n\n")
        term_text.insert("end", "Select a module to run:\n\n")
        term_text.insert("end", "1 - Evolutionary Agent (Current main.py logic)\n")
        term_text.insert("end", "2 - Basic RL Agent (CartPole DQN)                       --> main_cartpole.py\n")
        term_text.insert("end", "3 - Evolutionary DQN Agent                              --> main_cartpole_evolve.py\n")
        term_text.insert("end", "4 - Multi-Task RL Agent                                 --> main_multitask.py\n")
        term_text.insert("end", "5 - Meta-Learning Agent (MAML)                          --> main_maml.py\n")
        term_text.insert("end", "6 - Recursive Self-Improvement (RSI)                    --> main_rsi.py\n")
        term_text.insert("end", "7 - RL Agent                                            --> main_autotune.py\n")
        term_text.insert("end", "8 - Safe AI Agent                                       --> main_safe_ai.py\n\n")
        term_text.insert("end", "9 - Collaborative Agents (Task Routing, Shared Memory)  --> collaborative.main_collaborative.py
        term_text.insert("end", "0 - Exit\n\n")
        term_text.configure(state="disabled")

        # Right Panel
        results = tk.Frame(root, bg="black", width=640, height=720)
        results.pack(side="right", fill="both")

        label = tk.Label(results, text="[Results Visualized]", bg="black", fg="white",
                         font=header_font)
        label.place(relx=0.5, rely=0.5, anchor="center")

def launch_ui():
    root = tk.Tk()
    app = SLAIInterface(root)
    root.mainloop()
