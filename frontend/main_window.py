import os, sys
import psutil
import GPUtil
import torch
import logging
import subprocess
import json
import random   
import getpass
import queue
from datetime import datetime

from torch.nn.functional import cosine_similarity # Assuming this and REFERENCE_EMBEDDINGS/QUERIES are defined elsewhere if needed
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout,
                             QComboBox, QFrame, QSizePolicy, QFileDialog, QStackedWidget,
                             QSplitter, QApplication) # Added QStackedWidget, QSplitter, QApplication
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon, QTextCursor, QColor, QPainter
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect, QVariantAnimation, QSize, QThread

class PromptThread(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, agent, prompt, task_data):
        super().__init__()
        self.agent = agent
        self.prompt = prompt
        self.task_data = task_data

    def run(self):
        result = self.agent.generate(self.prompt, self.task_data)
        self.result_ready.emit(result)

from src.utils.agent_factory import AgentFactory
from src.utils.system_optimizer import SystemOptimizer
try:
    from src.agents.collaborative_agent import CollaborativeAgent
    try:
        from models.slai_lm import SLAILM # Needed by CollaborativeAgent
    except ImportError:
        print("Warning: Could not import SLAILM. Ensure models/slai_lm.py is accessible.")
        # Define a dummy SLAILM if needed for the script to run without the actual model
        class SLAILM: pass
except ImportError as e:
    print(f"Error importing CollaborativeAgent: {e}")
    print("Please ensure 'collaborative_agent.py' exists in the 'src/agents/' directory relative to main.py.")
    print("Also check dependencies like 'SLAILM', 'AgentFactory', 'SharedMemory', 'GrammarProcessor'.")
    # Define a placeholder if import fails, so the rest of the GUI can load
    class CollaborativeAgent:
        def __init__(self, agent_network=None, risk_threshold=0.2, **kwargs):
            print("--- Using Placeholder CollaborativeAgent ---")
            pass
        def generate(self, prompt):
            # Simulate response
            import time
            time.sleep(0.5)
            return f"Placeholder response (Import Failed) to: {prompt}"

class DynamicTextEdit(QTextEdit):
    heightChanged = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.setMaximumHeight(150)
        self.document().documentLayout().documentSizeChanged.connect(self.updateHeight)
        self.textChanged.connect(self.keep_cursor_bottom)

    def updateHeight(self):
        doc_height = self.document().size().height()
        margins = self.contentsMargins()
        new_height = int(doc_height + margins.top() + margins.bottom() + 5)  # Basic padding
        min_h = 50
        max_h = 300
        # max_h = min(350, self.parent().parent().maximumHeight())
        final_height = min(new_height, max_h)

        # self.setMinimumHeight(final_height)
        # self.setMaximumHeight(final_height)
        self.setFixedHeight(final_height)
        self.heightChanged.emit(final_height)

    def keep_cursor_bottom(self):
        """Ensures text stays anchored to bottom"""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)

class StatusIndicator(QLabel):
    """ Simple circle indicator """
    def __init__(self, color="grey", size=15, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._color = QColor(color)
        self.set_status("grey") # Default to off

    def set_status(self, color_name):
        self._color = QColor(color_name)
        self.update() # Trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self._color)
        painter.setPen(Qt.NoPen)
        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) // 2 - 1 # Small padding
        painter.drawEllipse(center, radius, radius)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, collaborative_agent, shared_memory, log_queue=None, metric_queue=None, shared_resources=None, optimizer=None):
        super().__init__()
        self.collaborative_agent = collaborative_agent
        self.shared_memory = shared_memory
        self.log_queue = log_queue or []
        self.metric_queue = metric_queue or []
        self.session_start_time = datetime.now()
        self.response_times = []
        self.current_response_start = None
        self.memory_peak = 0
        self.error_count = 0
        self.risk_trigger_count = 0
        self.safe_ai_count = 0
        shared_resources = {"log_path": "logs/", "memory_limit": 1000}
        optimizer = SystemOptimizer()

        factory = AgentFactory(shared_resources=shared_resources, optimizer=optimizer)

        self.agent = self.collaborative_agent
        self.setWindowTitle("SLAI Launcher")
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "..", "frontend", "assets", "logo1.ico")))

        # Geometry fix for large DPI / screen
        screen_geometry = QtWidgets.QDesktopWidget().availableGeometry()
        safe_width = min(screen_geometry.width(), 1920)
        safe_height = min(screen_geometry.height(), 1080)
        self.setGeometry(100, 100, safe_width, safe_height)
        self.setStyleSheet("""
            QMainWindow {
                background-color: black;
            }
            QWidget {
                color: white;
                font-family: 'Times New Roman', Times, serif; /* Added fallback */
            }
            QTextEdit, QLineEdit {
                background-color: #1e1e1e; /* Slightly lighter background for inputs */
                border: 1px solid gray;
                border-radius: 6px;
                padding: 5px;
            }
            QPushButton {
                padding: 5px 10px;
                border-radius: 6px;
                font-weight: bold;
                background-color: gold;
                color: black;
            }            
            QPushButton#uploadButton, QPushButton#downloadButton { /* Use object names for specific styling */
                background-color: #38b6ff;
                color: #0e0e0e;
                font-weight: bold;
                min-width: 30px; /* Make them squarish */
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
                padding: 0; /* Remove padding for icon-like buttons */
            }
             QPushButton#uploadButton {
                 background-color: #38b6ff;
                 icon: url(frontend/assets/upload.png);
             }
             QPushButton#downloadButton {
                 background-color: #fefefe;
                 icon: url(frontend/assets/download.png);
             }
             QPushButton#clearButton {
                 background-color: #fefefe;
                 color: #0e0e0e;
             }
             QPushButton#saveButton {
                 background-color: #FFD700; /* Gold */
                 color: black;
             }
             QPushButton#loadButton {
                  background-color: #FFD700; /* Gold */
                  color: black;
             }
            QLabel#headerLabel {
                font-size: 22px;
                font-weight: bold;
                color: gold;
                padding-left: 10px; /* Add some padding */
            }
            QLabel#footerLabel {
                font-size: 11px;
                color: lightgray;
                padding: 5px; /* Add padding */
            }
            QTextEdit#logPanel { /* Style for log panel */
                 font-family: Consolas, monospace; /* Monospace for logs */
                 font-size: 12px;
                 background-color: #111; /* Darker bg for logs */
            }
            QStackedWidget > QWidget { /* Style widgets inside stacked widget */
                 background-color: #1e1e1e;
                 border: 1px solid #444;
                 border-radius: 4px;
            }
            /* Style for the status indicators container */
             #statusIndicatorPanel {
                 padding: 5px;
                 spacing: 10px; /* Spacing between indicators */
             }
        """)
        self.status_message = QtWidgets.QLabel()
        self.loading_animation = None
        self.color_animation = None

        self.typing_texts = ["Scalable Learning Autonomous Intelligence ", "SLAI"]
        self.typing_index = 0
        self.char_index = 0
        self.display_text = ""
        self.is_pausing = False
        self.pause_counter = 0

        # Main Splitter for 60/40 layout
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.main_splitter)

        # --- Left Panel (60%) ---
        self.left_widget = QWidget()
        self.left_panel_layout = QVBoxLayout(self.left_widget)
        self.left_panel_layout.setContentsMargins(20, 32, 10, 20)
        self.left_panel_layout.setSpacing(1)
        self.main_splitter.addWidget(self.left_widget)

        # --- Right Panel (40%) ---
        self.right_widget = QWidget()
        self.right_panel_layout = QVBoxLayout(self.right_widget)
        self.right_panel_layout.setContentsMargins(10, 20, 20, 20)
        self.right_panel_layout.setSpacing(1)
        self.main_splitter.addWidget(self.right_widget)

        self.initUI()
        self.initTimer()

        # Set initial split sizes (60/40)
        total_width = self.width()
        self.main_splitter.setSizes([int(total_width * 0.6), int(total_width * 0.4)])

        self.greetUser()

    def initUI(self):
        # === Left Panel Content ===
        # Header with typing animation and control buttons
        header_layout = QHBoxLayout()
        self.header_label = QtWidgets.QLabel("SLAI")
        self.header_label.setObjectName("headerLabel") # For styling
        header_layout.addWidget(self.header_label)
        header_layout.addStretch()

        # Buttons (moved from center panel originally)
        self.save_button = QtWidgets.QPushButton("Save") # Shortened label
        self.save_button.setObjectName("saveButton")
        self.save_button.setToolTip("Save Chat Log")
        self.save_button.clicked.connect(self.save_logs)
        header_layout.addWidget(self.save_button)

        self.load_button = QtWidgets.QPushButton("Load") # Shortened label
        self.load_button.setObjectName("loadButton")
        self.load_button.setToolTip("Load Chat Log")
        self.load_button.clicked.connect(self.load_chat)
        header_layout.addWidget(self.load_button)

        self.clear_button = QtWidgets.QPushButton("Clear") # Shortened label
        self.clear_button.setObjectName("clearButton")
        self.clear_button.setToolTip("Clear Current Chat")
        self.clear_button.clicked.connect(self.clear_chat)
        header_layout.addWidget(self.clear_button)

        self.left_panel_layout.addLayout(header_layout)

        # --- Output Area (Main part of left panel) ---
        self.output_area = QtWidgets.QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setObjectName("outputArea") # For potential styling
        self.output_area.setStyleSheet("font-size: 14px;") # Style specific to output
        self.left_panel_layout.addWidget(self.output_area, 1) # Takes expanding space

        # --- Input Area (Bottom of left panel) ---
        input_area_layout = QHBoxLayout()
        input_area_layout.setSpacing(5)

        self.input_field = DynamicTextEdit()
        self.input_field.setPlaceholderText("Type your prompt here... (Shift+Enter for newline)")
        self.input_field.setObjectName("inputField")
        input_area_layout.addWidget(self.input_field, 1) # Input takes most space

        # Upload Button (Moved next to input)
        self.upload_media_button = QtWidgets.QPushButton() # Icon preferred
        self.upload_media_button.setObjectName("uploadButton") # For styling
        self.upload_media_button.setToolTip("Upload Media")
        self.upload_media_button.clicked.connect(self.upload_media)
        input_area_layout.addWidget(self.upload_media_button)

        # Download Button (New)
        self.download_button = QtWidgets.QPushButton() # Icon preferred
        self.download_button.setObjectName("downloadButton") # For styling
        self.download_button.setToolTip("Download Content/Logs") # Define action
        self.download_button.clicked.connect(self.download_content) # Create this method
        input_area_layout.addWidget(self.download_button)

        self.left_panel_layout.addLayout(input_area_layout)

        # Typing Timer for Header
        self.typing_timer = QtCore.QTimer()
        self.typing_timer.timeout.connect(self.updateTyping)
        self.typing_timer.start(100)

        # Override key press for input field
        self.input_field.keyPressEvent = self.handleInputKeyPress

        # === Right Panel Content ===
        # --- Status Indicators ---
        status_indicator_layout = QHBoxLayout()
        status_indicator_layout.setObjectName("statusIndicatorPanel")
        status_indicator_layout.addStretch() # Push lights to the right
        self.indicator_red = StatusIndicator("grey") # Default grey/off
        self.indicator_red.setToolTip("SLAI Busy (Loading/Generating)")
        self.indicator_orange = StatusIndicator("grey")
        self.indicator_orange.setToolTip("Critical Issue Detected")
        self.indicator_yellow = StatusIndicator("grey")
        self.indicator_yellow.setToolTip("Warning/Failed Request")
        self.indicator_green = StatusIndicator("green") # Start green (standby)
        self.indicator_green.setToolTip("SLAI Standby - OK")

        status_indicator_layout.addWidget(self.indicator_red)
        status_indicator_layout.addWidget(self.indicator_orange)
        status_indicator_layout.addWidget(self.indicator_yellow)
        status_indicator_layout.addWidget(self.indicator_green)
        self.right_panel_layout.addLayout(status_indicator_layout)

        # --- Switchable Panels ---
        self.right_tab_widget = QStackedWidget()
        self.right_panel_layout.addWidget(self.right_tab_widget, 1) # Takes expanding space

        # Panel 1: Real-Time Logs
        self.log_panel = QTextEdit()
        self.log_panel.setObjectName("logPanel") # For styling
        self.log_panel.setReadOnly(True)
        self.log_panel.setText("Real-time system logs will appear here...")
        self.right_tab_widget.addWidget(self.log_panel)

        # Panel 2: Live Score Graph (Placeholder)
        self.graph_panel = QLabel("Live Score Graph Placeholder")
        self.graph_panel.setAlignment(Qt.AlignCenter)
        self.graph_panel.setStyleSheet("background-color: #2a2a2a; border-radius: 4px;")
        self.right_tab_widget.addWidget(self.graph_panel)

        # Panel 3: Media Panel (Placeholder)
        self.media_panel = QLabel("Media Panel Placeholder (e.g., for images)")
        self.media_panel.setAlignment(Qt.AlignCenter)
        self.media_panel.setStyleSheet("background-color: #2a2a2a; border-radius: 4px;")
        self.right_tab_widget.addWidget(self.media_panel)

        # --- Footer (System Stats) ---
        self.footer = QtWidgets.QLabel("System Stats Initializing...")
        self.footer.setObjectName("footerLabel") # For styling
        self.right_panel_layout.addWidget(self.footer)

        # --- Status Message Label (Bottom Right) ---
        self.status_message = QLabel()
        self.status_message.setObjectName("statusMessage")
        self.status_message.setStyleSheet("color: gold; font-size: 12px; padding: 2px 5px;")
        self.status_message.hide()
        self.right_panel_layout.addWidget(self.status_message)

        # --- Example: Add buttons to switch right panels ---
        switcher_layout = QHBoxLayout()
        log_btn = QPushButton("Logs")
        graph_btn = QPushButton("Graph")
        media_btn = QPushButton("Media")
        log_btn.clicked.connect(lambda: self.right_tab_widget.setCurrentIndex(0))
        graph_btn.clicked.connect(lambda: self.right_tab_widget.setCurrentIndex(1))
        media_btn.clicked.connect(lambda: self.right_tab_widget.setCurrentIndex(2))
        switcher_layout.addWidget(log_btn)
        switcher_layout.addWidget(graph_btn)
        switcher_layout.addWidget(media_btn)
        switcher_layout.addStretch()
        self.right_panel_layout.insertLayout(1, switcher_layout) # Insert above the stack

    def handleInputKeyPress(self, event):
        """ Handle key presses in the input field """
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if event.modifiers() & Qt.ShiftModifier:
                # Insert newline if Shift is pressed
                self.input_field.textCursor().insertText("\n")
            else:
                # Submit prompt if only Enter is pressed
                self.submitPrompt()
                return # Consume the event
        # Default handling for other keys
        super(DynamicTextEdit, self.input_field).keyPressEvent(event)

    def greetUser(self):
        """Greets the user with a time-specific message, including the username."""
        hour = datetime.now().hour

        if 2 <= hour <= 11: greeting = "Good morning, "
        elif 12 <= hour <= 16: greeting = "Good afternoon, "
        elif 17 <= hour <= 22: greeting = "Good evening, "
        else: greeting = "Good night, "

        # Get the username
        
        try: username = getpass.getuser()
        except Exception: username = "User"
        greeting += username + ","

        # Load random responses from JSON
        try:
            json_path = os.path.join(os.path.dirname(__file__), "templates/nlg_templates_en.json")
            with open(json_path, "r") as f:
                responses = json.load(f)
            random_response = random.choice(responses["responses"])
            greeting_message = f"{greeting} {random_response}"
        except FileNotFoundError:
            greeting_message = greeting + " I hope you're having a good day."
        except json.JSONDecodeError:
            greeting_message = greeting + " There was an error loading additional responses."

        greeting_message = f"{greeting} {random_response}"
        # Post greeting to the main output area
        self.output_area.append(f"<font color='gold'>SLAI:</font> {greeting_message}<br>{'-'*50}<br>")

    def play_audio(self, audio_path):
        """Plays the selected audio file."""
        # Basic audio playback (Needs more robust implementation)
        try:
            os.system(f"start {audio_path}")  # Simple play (Windows only)
            self.status_message.setText(f"Playing: {os.path.basename(audio_path)}")
            self.status_message.show()
            QTimer.singleShot(3000, self.status_message.hide)
        except Exception as e:
            self.status_message.setText(f"Failed to play audio: {e}")
            self.status_message.show()
            QTimer.singleShot(3000, self.status_message.hide)

    def handle_response(self, text):
        self.stop_loading_animation()
        current_output = self.tabs[self.tab_stack.currentIndex()]
        
        user_embedding = encode_sentence(text)
        similarities = cosine_similarity(user_embedding, REFERENCE_EMBEDDINGS)
        
        max_score = similarities.max().item()
        threshold = 0.75  # Semantic confidence threshold
        
        if max_score >= threshold:
            matched_idx = torch.argmax(similarities).item()
            matched_reference = REFERENCE_QUERIES[matched_idx]
            current_output.append(f"\nUser: {text}\nSLAI: Detected similarity to '{matched_reference}' (score = {max_score:.3f})\n[Academic response generated here...]")
        else:
            self.status_message.setText("SLAI was unable to find a relevant academic response.")
            self.status_message.show()
            QtCore.QTimer.singleShot(3000, self.status_message.hide)

        self.input_field.clear()

    def encode_sentence(text):
        # needs to expanded upon
            return

    def save_logs(self):
        chat_text = self.output_area.toHtml() # Save rich text if needed
        system_text = self.log_panel.toPlainText() # Get text from log panel

        logs_dir = os.path.join("logs", "chat_logs")
        os.makedirs(logs_dir, exist_ok=True)
        default_filename = os.path.join(logs_dir, f"chatlog_{datetime.now():%Y%m%d_%H%M%S}.html") # Changed to html

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Chat Log", default_filename, "HTML Files (*.html);;Text Files (*.txt);;All Files (*)")

        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as file:
                    # Simple HTML structure
                    file.write("<html><head><title>Chat Log</title></head><body>")
                    file.write("<h1>Chat Log</h1><hr>")
                    file.write(chat_text) # Write HTML content
                    file.write("<br><hr><h1>System Logs</h1><hr><pre>") # Use pre for plain text logs
                    file.write(system_text)
                    file.write("</pre></body></html>")
                self.show_status_message("Logs saved successfully.", 3000)
            except Exception as e:
                self.show_status_message(f"Failed to save logs: {e}", 5000)

    def load_chat(self):
        logs_dir = os.path.join("logs", "chat_logs")
        os.makedirs(logs_dir, exist_ok=True)

        file_path, _ = QFileDialog.getOpenFileName(self, "Load Chat Log", logs_dir, "HTML Files (*.html);;Text Files (*.txt);;All Files (*)")

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                # Check if it's likely HTML or plain text
                if file_path.lower().endswith(".html"):
                     # Basic extraction if needed, or just load as HTML
                     # For simplicity, just load it, might need parsing for clean display
                     self.output_area.setHtml(content)
                else:
                     self.output_area.setPlainText(content) # Load as plain text

                self.show_status_message(f"Loaded: {os.path.basename(file_path)}", 3000)
            except Exception as e:
                 self.show_status_message(f"Failed to load file: {e}", 5000)

    def clear_chat(self):
        self.output_area.clear()
        self.greetUser()
        self.show_status_message("Chat cleared.", 2000)

    def upload_media(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Media File", "",
            "Image Files (*.png *.jpg *.jpeg);;Audio Files (*.mp3 *.wav);;All Files (*.*)"
        )

        if file_path:
            self.show_status_message(f"Selected media: {os.path.basename(file_path)}", 3000)
            # Add logic here to process/display the media
            # For example, display image in the media panel:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in (".png", ".jpg", ".jpeg"):
                self.display_image_in_media_panel(file_path)
            elif file_extension in (".mp3", ".wav"):
                 # Add audio handling logic
                 self.show_status_message("Audio file selected (playback not implemented yet).", 3000)
            else:
                 self.show_status_message("File selected (handling not implemented yet).", 3000)

    def display_image_in_media_panel(self, image_path):
        """ Displays image in the dedicated media panel """
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.show_status_message("Failed to load image.", 3000)
            return

        # Scale pixmap to fit the media panel label
        # Ensure media_panel is visible before getting size
        if self.media_panel.isVisible():
            target_size = self.media_panel.size()
        else:
            target_size = QSize(200, 200) # Default size if not visible yet

        scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.media_panel.setPixmap(scaled_pixmap)
        self.right_tab_widget.setCurrentWidget(self.media_panel) # Switch to media panel
        self.show_status_message(f"Displayed image: {os.path.basename(image_path)}", 3000)

    def download_content(self):
        # Calculate session duration
        session_duration = datetime.now() - self.session_start_time
        hours, remainder = divmod(session_duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        """
        Exports chat and system session data into a styled PDF using generate_pdf_report.
        """
        from frontend.templates.generate_pdf_report import generate_pdf_report

        logs_dir = os.path.join("reports", "session_reports")
        os.makedirs(logs_dir, exist_ok=True)
        default_filename = os.path.join(logs_dir, f"SLAI_Report_{datetime.now():%Y%m%d_%H%M%S}.pdf")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save SLAI Report",
            default_filename,
            "PDF Files (*.pdf);;All Files (*)"
        )

        if not save_path:
            return

        try:
            # Collect dynamic chat logs
            chat_text = self.output_area.toPlainText()

            # Build data structure expected by generate_pdf_report
            data = {
                "chat_history": chat_text,
                "executive_summary": "This report summarizes the SLAI session performance and system activity.",
                "user_id": getpass.getuser(),
                "interaction_count": len(self.response_times),
                "total_duration": f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}",
                "avg_response_time": f"{sum(self.response_times)/len(self.response_times):.2f}s" if self.response_times else "N/A",
                "min_response_time": f"{min(self.response_times):.2f}s" if self.response_times else "N/A",
                "max_response_time": f"{max(self.response_times):.2f}s" if self.response_times else "N/A",
                "session_uptime": str(datetime.now() - self.session_start_time).split('.')[0],
                "memory_peak": f"{self.memory_peak:.2f} GB",
                "risk_triggered": self.risk_trigger_count,
                "safe_ai": self.safe_ai_count,
                "errors": self.error_count,
                "modules_used": ", ".join(
                    set(agent.__class__.__name__ for agent in self.agent.agent_network.values())) 
                        if hasattr(
                            self.agent, 
                            "agent_network") 
                            else 
                            "CollaborativeAgent",
                "export_format": "PDF",
                "risk_logs": "No critical risks were triggered during the session.",
                "recommendations": "Maintain active monitoring of SLAI response time.",
                "log_file_path": "N/A",
                "chat_export_path": "N/A",
                "notes": "Auto-generated report from SLAI desktop session.",
                "date_range": "Last 7 days",
                "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generated_by": getpass.getuser(),
                "country": "Netherlands",
                "watermark_logo_path": "frontend/assets/logo1.png",
                "cover_logo_path": "frontend/assets/logo.png",
            }

            generate_pdf_report(save_path, data)
            self.show_status_message(f"PDF report saved: {os.path.basename(save_path)}", 4000)

        except Exception as e:
            logging.exception("Failed to generate PDF report")
            self.set_status_indicator("error") # Set YELLOW light on error
            self.show_status_message(f"Error: {str(e)}", 5000)

    def set_status_indicator(self, state):
        """ Updates status lights based on state """
        # Reset all to grey first
        self.indicator_red.set_status("grey")
        self.indicator_orange.set_status("grey")
        self.indicator_yellow.set_status("grey")
        self.indicator_green.set_status("grey")

        if state == "busy":
            self.indicator_red.set_status("#e51e25")
        elif state == "error":
            self.indicator_yellow.set_status("#ffd051")
        elif state == "critical":
            self.indicator_orange.set_status("#ff914d")
        elif state == "standby":
            self.indicator_green.set_status("#00bf63")
        else: # Default to standby / OK
            self.indicator_green.set_status("#00bf63")

    def submitPrompt(self):
        text = self.input_field.toPlainText().strip()
        if text:
            self.set_status_indicator("busy") # Set RED light
            self.output_area.append(f"<font color='lightblue'>User:</font> {text}<br>") # Display user prompt
            self.input_field.clear() # Clear input after sending
            self.show_status_message("Sending prompt to SLAI...", 2000)

            # Create task_data with default context
            task_data = {"context": "default", "prompt": text}
            QtCore.QTimer.singleShot(100, lambda: self.call_slai_pipeline(task_data, text))

    def call_slai_pipeline(self, task_data: dict, prompt: str):
        thread = PromptThread(self.agent, prompt, task_data)
        thread.result_ready.connect(lambda result: self.output_area.setPlainText(result))
        thread.start()

        try:
            # Start response timer
            self.current_response_start = datetime.now()

            # Call the generate method of the imported CollaborativeAgent instance
            slai_response = self.agent.generate(prompt, task_data)
            
            # Calculate response time
            response_time = (datetime.now() - self.current_response_start).total_seconds()
            self.response_times.append(response_time)
            
            # Display response
            self.output_area.append(f"<font color='gold'>SLAI:</font> {slai_response}<br>")
            self.set_status_indicator("standby") # Set GREEN light on success
            self.show_status_message("Response received.", 3000)
            
            # Track SafeAI interventions
            if "SafeAI intervention" in slai_response:
                self.safe_ai_count += 1

        except Exception as e:
            self.error_count += 1
            # Log the full exception for debugging
            logging.exception(f"Error during SLAI pipeline execution for prompt: {prompt}")
            error_message = f"Error processing prompt: {e}"
            self.output_area.append(f"<font color='red'>[ERROR]:</font> {error_message}<br>")
            self.set_status_indicator("error") # Set YELLOW light on error
            self.show_status_message(error_message, 5000)
        finally:
            self.current_response_start = None
        # Ensure cursor is visible
        self.output_area.moveCursor(QTextCursor.End)

    def updateTyping(self):
        if self.is_pausing:
            self.pause_counter += 100
            if self.pause_counter >= 5000:
                self.is_pausing = False
                self.typing_index = (self.typing_index + 1) % len(self.typing_texts)
                self.char_index = 0
                self.display_text = ""
            return

        full_text = self.typing_texts[self.typing_index]
        if self.char_index < len(full_text):
            self.display_text += full_text[self.char_index]
            self.header_label.setText(self.display_text + "...")
            self.char_index += 1
        else:
            self.header_label.setText(self.display_text)
            self.is_pausing = True
            self.pause_counter = 0

    def initTimer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateSystemStats)
        self.timer.timeout.connect(self.update_log_panel)
        self.timer.start(1000)

    def updateSystemStats(self):
        
        # System stats update logic (minor formatting change)
        try:
            current_mem = psutil.virtual_memory().used / (1024**3)
            if current_mem > self.memory_peak:
                self.memory_peak = current_mem
            cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0
            cpu_usage = psutil.cpu_percent()
            threads = psutil.cpu_count(logical=True)
            processes = len(psutil.pids())
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / (1024**3)
            ram_total_gb = ram.total / (1024**3)
            ram_percent = ram.percent
            ram_available = ram.available // (1024 * 1024)
            gpu_name = "N/A"
            gpu_usage = 0
            gpu_temp = 0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_usage = gpu.load * 100
                    gpu_temp = gpu.temperature
                    gpu_name_full = gpu.name.strip()
                    gpu_name = gpu_name_full[:20] + '...' if len(gpu_name_full) > 20 else gpu_name_full # Shorten name
            except Exception:
                gpu_name = "GPU Error" # Indicate if GPUtil fails

            # Rearranged slightly, added RAM GB
            text = f"CPU: {cpu_freq:.0f}MHz ({cpu_usage:.1f}%) | RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f} GB ({ram_percent}%) | "
            text += f"GPU: {gpu_name} ({gpu_usage:.1f}%, {gpu_temp:.0f}°C)"
        except Exception as e:
             text = f"Error updating stats: {e}"

        self.footer.setText(text)

    def update_log_panel(self):
        """ Placeholder to update log panel """
        # In a real app, you'd fetch logs from a queue or source
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Append vs SetText: Append keeps history
        self.log_panel.append(f"[{timestamp}] System heartbeat...")
        # Auto-scroll to bottom
        self.log_panel.moveCursor(QTextCursor.End)

    def show_status_message(self, message, duration_ms):
        """ Displays a status message for a specified duration """
        self.status_message.setText(message)
        self.status_message.show()
        QTimer.singleShot(duration_ms, self.status_message.hide)

    # --- Layout Handling for Vertical/Horizontal ---
    def resizeEvent(self, event):
        """ Handle window resizing to adjust layout """
        super().resizeEvent(event)
        self.adjustLayoutOrientation()
        # Also rescale media panel image on resize
        if self.media_panel.pixmap() and not self.media_panel.pixmap().isNull():
             # Need the original path to reload and scale properly
             # This requires storing the path when the image is loaded
             # For now, just re-apply current pixmap scaled
             current_pixmap = self.media_panel.pixmap()
             scaled_pixmap = current_pixmap.scaled(self.media_panel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
             self.media_panel.setPixmap(scaled_pixmap)

    def adjustLayoutOrientation(self):
         """ Switch splitter orientation based on window aspect ratio """
         size = self.size()
         if size.height() > size.width() * 1.2: # If significantly taller than wide
             if self.main_splitter.orientation() != Qt.Vertical:
                 self.main_splitter.setOrientation(Qt.Vertical)
                 # Adjust vertical sizes (e.g., top 40%, bottom 60%)
                 total_height = self.height()
                 # Ensure sizes are positive integers
                 size1 = max(1, int(total_height * 0.4))
                 size2 = max(1, int(total_height * 0.6))
                 self.main_splitter.setSizes([size1, size2]) # Right(top) 40%, Left(bottom) 60%
         else: # Horizontal layout
             if self.main_splitter.orientation() != Qt.Horizontal:
                 self.main_splitter.setOrientation(Qt.Horizontal)
                 # Adjust horizontal sizes (60/40)
                 total_width = self.width()
                 # Ensure sizes are positive integers
                 size1 = max(1, int(total_width * 0.6))
                 size2 = max(1, int(total_width * 0.4))
                 self.main_splitter.setSizes([size1, size2]) # Left 60%, Right 40%

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    launcher = MainWindow()
    launcher.showFullScreen()
    sys.exit(app.exec_())
