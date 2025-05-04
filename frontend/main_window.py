import os, sys
import psutil, GPUtil
import torch
import json, yaml
import logging
import random
import getpass
from datetime import datetime

from typing import Optional
from collections import deque
from PyQt5.QtWidgets import QMessageBox, QInputDialog, QLineEdit
from torch.nn.functional import cosine_similarity
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout,
                             QComboBox, QFrame, QSizePolicy, QFileDialog, QStackedWidget, QDialog,
                             QSplitter, QApplication,) # Added QStackedWidget, QSplitter, QApplication
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon, QTextCursor, QColor, QPainter
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QVariantAnimation, QSize, QThread

from src.utils.async_task_report import run_in_thread
from src.agents.collaborative_agent import CollaborativeAgent
from logs.logging_handler import QtLogHandler
from logs.logger import get_logger
from src.utils.agent_factory import AgentFactory
from src.utils.system_optimizer import SystemOptimizer
from models.music.music_editor import MusicEditor
from models.training.language_editor import LanguageEditor
from models.musician import Musician
from models.auditor import IdleAuditManager
from frontend.utils.text_editor import TextEditor
# from frontend.utils.audio_editor import AudioEditor
# from frontend.utils.visual_editor import VisualEditor

logger = get_logger(__name__)

# === Configuration ===
REFERENCE_QUERIES = [
    "What is the capital of France?",
    "Explain Newton's second law.",
    "Describe the process of photosynthesis."
]

def encode_sentence(text):
    import hashlib
    # Simulated embedding using a deterministic hash to tensor conversion
    hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (10**8)
    torch.manual_seed(hash_val)
    return torch.rand(1, 768)

REFERENCE_EMBEDDINGS = torch.stack([encode_sentence(q) for q in REFERENCE_QUERIES])  


class PromptThread(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, agent, prompt, task_data):
        super().__init__()
        self.agent = agent
        self.prompt = prompt
        self.task_data = task_data

    def run(self):
        result = self.agent.generate(self.agent.shared_memory, self.task_data)
        self.result_ready.emit(result)

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

def check_nlg_trigger(prompt: str, nlg_templates: dict) -> Optional[str]:
    tokens = prompt.strip().lower().split()
    if not (1 <= len(tokens) <= 2):
        return None  # Too long

    for category, content in nlg_templates.items():
        for trigger in content.get("triggers", []):
            if trigger.lower() in prompt.lower():
                return random.choice(content.get("responses", []))

    return None  # No match

def handle_user_prompt(self, prompt: str):
    response = check_nlg_trigger(prompt, self.nlg_templates)
    if response:
        self.display_response(response)
    else:
        self.process_language_agent(prompt)

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
        self.shared_memory = shared_memory
        if not self.shared_memory.get('evaluation_metrics'):
            self.shared_memory.set('evaluation_metrics', {
                'hazards': [],
                'operational_time': 0,
                'successes': 0,
                'failures': 0
            })
        with open("frontend/templates/nlg_templates_en.json", "r", encoding="utf-8") as f:
            self.nlg_templates = json.load(f)
        self.collaborative_agent = collaborative_agent
        self.optimizer = optimizer
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.factory = AgentFactory(config=config, shared_resources=shared_resources)
        self.current_musician_model = None

        # Setup QtLogHandler
        self.qt_log_handler = QtLogHandler()
        self.qt_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(self.qt_log_handler)
        logging.getLogger().setLevel(logging.INFO)

        self._init_base_ui()
        self._deferred_ui_setup()
        self.log_queue = log_queue or []
        self.metric_queue = metric_queue or []
        self.session_start_time = datetime.now()
        self.response_times = []
        self.current_response_start = None
        self.memory_peak = 0
        self.error_count = 0
        self.risk_trigger_count = 0
        self.safe_ai_count = 0
        shared_resources = {"log_path": "logs/", "memory_limit": 1000, "shared_memory": self.shared_memory, "agent_factory": self.factory}
        optimizer = SystemOptimizer()
        self.agent = self.collaborative_agent
        self.setWindowTitle("SLAI Launcher")
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "..", "frontend", "assets", "logo.ico")))

        QTimer.singleShot(0, self._deferred_ui_setup)  # Defer heavy UI loading

        from src.agents.evaluators.report import PerformanceVisualizer
        self.visualizer = PerformanceVisualizer(max_points=200)
        self.last_visual_update = datetime.now()

        # Connect to evaluation agent updates
        shared_memory.register_callback('evaluation_metrics', self.handle_evaluation_update)

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
                background-color: #0e0e0e; /* Slightly lighter background for inputs */
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
                 font-size: 15px;
                 /* background-color: #111; Darker bg for logs */
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
        self.left_panel_layout.setContentsMargins(75, 32, 10, 20)
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

        from models.training.shared_signals import training_signals
        # Access embedding data through:
        training_signals.embedding_update

        self.synonym_controller = None # Add placeholder for controller

    def handle_evaluation_update(self, metrics):
        """Process evaluation metrics from EvaluationAgent"""
        self.visualizer.update_metrics(metrics)
        # Throttle visual updates to 15 FPS
        if (datetime.now() - self.last_visual_update).total_seconds() > 0.066:
            self.update_visualizations()
            self.last_visual_update = datetime.now()

        approvals = metrics.get("approvals", [])
        if not approvals:
            return
        approval_queue = deque(approvals)
        dialog = QDialog(self)
        dialog.setWindowTitle("Approve Generated Terms")
        layout = QVBoxLayout(dialog)

        def process_next():
            nonlocal approval_queue
            if not approval_queue:
                dialog.accept()
                return

            word, entry = approval_queue.popleft()
            current_type, current_item = approval_queue.popleft()
            prompt = QLabel(f"Approve {current_type}: '{current_item}'?")
            input_field = QLineEdit()
            input_field.setPlaceholderText("y/n or enter correction")

            def handle_response():
                decision = input_field.text().strip().lower()
                nonlocal entry, word
                if decision in ('y', ''):
                    pass # Keep original
                elif decision == 'n':
                    # Remove from entry
                    if current_type == 'synonym':
                        entry['synonyms'].remove(current_item)
                    else:
                        entry['related_terms'].remove(current_item)
                else:  # User entered replacement
                    if current_type == 'synonym':
                        entry['synonyms'].append(decision)
                    else:
                        entry['related_terms'].append(decision)

                # Clear current widgets
                for w in [prompt, input_field, btn]:
                    layout.removeWidget(w)
                    w.deleteLater()

                process_next()

            btn = QPushButton("Submit")
            btn.clicked.connect(handle_response)

            layout.addWidget(prompt)
            layout.addWidget(input_field)
            layout.addWidget(btn)
        
        def handle_final_approval():
            from models.training.shared_signals import training_signals
            training_signals.batch_approved.emit(True)  # Signal approval
            dialog.accept()

        word, entry = approval_queue.popleft()
        final_btn = QPushButton("Finish Approval")
        final_btn.clicked.connect(handle_final_approval)
        layout.addWidget(final_btn)
        
        process_next()
        dialog.exec_()
        self._save_approved_entry(word, entry)

    def show_single_approval_prompt(self, item):
        dialog = QInputDialog(self)
        dialog.setWindowTitle(f"Approve {self.current_approval_queue['current_type']}")
        dialog.setLabelText(
            f"{self.current_approval_queue['word']} - {item}\n"
            "Approve? [y/n] Or enter replacement:"
        )
        dialog.setTextValue("y")
        dialog.accepted.connect(lambda: self.handle_approval_decision(dialog.textValue()))
        dialog.exec_()

    def handle_approval_decision(self, decision):
        decision = decision.strip().lower()
        current_type = self.current_approval_queue['current_type']

        if decision == 'n':
            # Remove item
            pass
        elif decision.startswith('y'):
            # Keep original
            self.current_approval_queue['entry'][current_type].append(
                self.current_approval_queue['current_item']
            )
        else:
            # Use replacement
            self.current_approval_queue['entry'][current_type].append(decision)

        self.process_next_approval_item()

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
        self.output_area.setStyleSheet("""
            font-size: 14px;
            padding: 5px;
        """)
        self.left_panel_layout.addWidget(self.output_area, 1) # Takes expanding space

        # --- Input Area (Bottom of left panel) ---
        input_area_layout = QHBoxLayout()
        input_area_layout.setSpacing(5)

        self.input_field = TextEditor()
        self.input_field.setPlaceholderText("Type your prompt here... (Shift+Enter for newline)")
        self.input_field.setObjectName("inputField")
        input_area_layout.addWidget(self.input_field, 1) # Input takes most space

        # Upload Button (Moved next to input)
        self.upload_media_button = QtWidgets.QPushButton() # Icon preferred
        self.upload_media_button.setObjectName("uploadButton") # For styling
        self.upload_media_button.setToolTip("Upload Media")
        self.upload_media_button.clicked.connect(self.upload_media)
        input_area_layout.addWidget(self.upload_media_button)

        # Download Button
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

        self.indicator_heartbeat = StatusIndicator("grey")
        self.indicator_heartbeat.setToolTip("System Heartbeat")
        status_indicator_layout.addWidget(self.indicator_heartbeat)

        # --- Switchable Panels ---
        self.right_tab_widget = QStackedWidget()
        self.right_panel_layout.addWidget(self.right_tab_widget, 1) # Takes expanding space

        # Panel 1: Real-Time Logs
        self.log_panel = QTextEdit()
        self.log_panel.setObjectName("logPanel") # For styling
        self.log_panel.setReadOnly(True)
        self.log_panel.setText("Real-time system logs will appear here...")
        self.right_tab_widget.addWidget(self.log_panel)

        # Panel 2: Language Editor Panel
        self.training_logs = LanguageEditor()
        self.right_tab_widget.addWidget(self.training_logs)

        # Panel 3: Music Editor Panel
        self.music_editor = MusicEditor()
        self.right_tab_widget.addWidget(self.music_editor)

        # Panel 4: Media Panel (Placeholder)
        self.media_panel = QLabel("Media Panel Placeholder (e.g., for images)")
        self.media_panel.setAlignment(Qt.AlignCenter)
        self.media_panel.setStyleSheet("background-color: #2a2a2a; border-radius: 4px;")
        self.right_tab_widget.addWidget(self.media_panel)

        # Panel 5: Risk & Reward Visualization Panel
        self.visualization_panel = QWidget()
        vis_layout = QVBoxLayout()
        vis_layout.setContentsMargins(5, 5, 5, 5)

        self.vis_tradeoff = QLabel()
        self.vis_tradeoff.setMinimumSize(360, 240)
        self.vis_tradeoff.setAlignment(Qt.AlignCenter)
        
        self.vis_risk_trend = QLabel()
        self.vis_risk_trend.setMinimumSize(360, 240)
        self.vis_risk_trend.setAlignment(Qt.AlignCenter)
        
        self.vis_reward_trend = QLabel()
        self.vis_reward_trend.setMinimumSize(360, 240)
        self.vis_reward_trend.setAlignment(Qt.AlignCenter)
    
        vis_layout.addWidget(self.vis_tradeoff)
        vis_layout.addWidget(self.vis_risk_trend)
        vis_layout.addWidget(self.vis_reward_trend)
        
        self.visualization_panel.setLayout(vis_layout)
        self.right_tab_widget.addWidget(self.visualization_panel)

        # --- Footer (System Stats) ---
        self.log_footer = QLabel("System Stats Initializing...")
        self.log_footer.setObjectName("footerLabel")
        self.log_footer.setAlignment(Qt.AlignRight)

        log_container = QVBoxLayout()
        log_container.addWidget(self.log_panel)
        log_container.addWidget(self.log_footer)

        log_widget = QWidget()
        log_widget.setLayout(log_container)
        self.right_tab_widget.insertWidget(0, log_widget)
        self.footer = self.log_footer

        # --- Status Message Label (Bottom Right) ---
        self.status_message = QLabel()
        self.status_message.setObjectName("statusMessage")
        self.status_message.setStyleSheet("color: gold; font-size: 12px; padding: 2px 5px;")
        self.status_message.hide()
        self.right_panel_layout.addWidget(self.status_message)

        # --- Add buttons to switch right panels ---
        switcher_layout = QHBoxLayout()
        log_btn = QPushButton("Logs")
        training_btn = QPushButton("Training Logs")
        music_editor_btn = QPushButton("Music Editor")
        media_btn = QPushButton("Media")
        viz_btn = QPushButton("Eval Charts")
        log_btn.clicked.connect(lambda: self.right_tab_widget.setCurrentIndex(0))
        training_btn.setText("Training Logs")
        training_btn.clicked.connect(lambda: self.right_tab_widget.setCurrentIndex(1))
        music_editor_btn.clicked.connect(lambda: self.right_tab_widget.setCurrentIndex(2))
        music_editor_index = self.right_tab_widget.indexOf(self.music_editor)
        music_editor_btn.clicked.connect(lambda: self.right_tab_widget.setCurrentIndex(music_editor_index))
        media_btn.clicked.connect(lambda: self.right_tab_widget.setCurrentIndex(3))
        viz_btn.clicked.connect(lambda: self.right_tab_widget.setCurrentIndex(4))
        switcher_layout.addWidget(log_btn)
        switcher_layout.addWidget(training_btn)
        switcher_layout.addWidget(music_editor_btn)
        switcher_layout.addWidget(media_btn)
        switcher_layout.addWidget(viz_btn)
        switcher_layout.addStretch()
        self.right_panel_layout.insertLayout(1, switcher_layout) # Insert above the stack

# === Sidebar Buttons (Vertical, Circular, Flush Left with Unique Icons) ===
        self.sidebar_buttons_container = QWidget(self.left_widget)
        self.sidebar_buttons_container.setGeometry(10, 100, 50, 400)
        self.sidebar_buttons_container.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.sidebar_buttons_container.setStyleSheet("background: transparent;")

        sidebar_layout = QVBoxLayout(self.sidebar_buttons_container)
        sidebar_layout.setSpacing(20)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        buttons = [
            ("Home Page", "home", "frontend/assets/home.png", "frontend/assets/home_active.png"),
            ("User's info", "user", "frontend/assets/user.png", "frontend/assets/user_active.png"),
            ("Researcher Model", "research", "frontend/assets/research.png", "frontend/assets/research_active.png"),
            ("Musician Model", "music", "frontend/assets/musician.png", "frontend/assets/musician_active.png"),
            ("Training Model", "train", "frontend/assets/training.png", "frontend/assets/training_active.png"),
            ("Auditor Model", "audit", "frontend/assets/auditor.png", "frontend/assets/auditor_active.png")
        ]

        self.sidebar_button_map = {}
        self.active_sidebar_button = None

        for label, key, icon_path, checked_icon_path in buttons:
            btn = QPushButton()
            btn.setToolTip(label)
            btn.setFixedSize(50, 50)
            btn.setCheckable(True)
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(QSize(40, 40))
            btn.setStyleSheet("""
                QPushButton {
                    border-radius: 20px;
                    background-color: gold;
                    color: black;
                }
                QPushButton:checked {
                    background-color: black;
                    color: gold;
                    border: 0.5px solid gold;
                }
            """)
            btn.clicked.connect(lambda checked, k=key, l=label, b=btn, i=icon_path, ci=checked_icon_path: self.toggle_sidebar_button(k, l, b, i, ci))
            sidebar_layout.addWidget(btn)
            self.sidebar_button_map[key] = btn

        self.sidebar_buttons_container.raise_()


        # === Slide Animation Setup ===
        self.slide_anim = QPropertyAnimation(self.left_widget, b"pos")
        self.slide_anim.setDuration(400)
        self.slide_anim.setEasingCurve(QEasingCurve.OutCubic)

        from models.audit.audit_scheduler import AuditScheduler
        self.audit_scheduler = AuditScheduler(
            parent=self,
            output_callback=self.output_area.append,
            status_callback=self.show_status_message
        )

    def post_to_output_area(self, message: str, tag: str = "System"):
        """Post a formatted message to the output area."""
        color_map = {
            "user": "#f10000",
            "Musician Model": "gold",
            "Intermediate": "#38b4fe",
            "Related to": "#38b4fe"
        }
        color = color_map.get(tag, "gold")
        formatted = f"<span style='color:{color};'><b>[{tag}]</b></span> {message}<br>"
        self.output_area.append(formatted)

    def update_visualizations(self):
        """Update all visualization elements"""
        if not self.visualization_panel.isVisible():
            return

        # Get current panel size
        panel_size = self.visualization_panel.size()
        chart_width = max(360, panel_size.width() - 20)
        chart_height = max(240, panel_size.height() // 3 - 10)
        chart_size = QtCore.QSize(chart_width, chart_height)

        # Update charts
        try:
            self.vis_tradeoff.setPixmap(
                self.visualizer.render_tradeoff_chart(chart_size))
            self.vis_risk_trend.setPixmap(
                self.visualizer.render_temporal_chart(chart_size, 'hazard_rates'))
            self.vis_reward_trend.setPixmap(
                self.visualizer.render_temporal_chart(chart_size, 'operational_times'))
        except Exception as e:
            logger.error(f"Visualization update failed: {str(e)}")

    # Add resize handler
    def resizeEvent(self, event):
        """Handle window resizing for proper visualization scaling"""
        super().resizeEvent(event)
        self.update_visualizations()

    def handleInputKeyPress(self, event):
        """ Handle key presses in the input field """
        #self.user_input = user_input
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if event.modifiers() & Qt.ShiftModifier:
                # Insert newline if Shift is pressed
                self.input_field.textCursor().insertText("\n")
            else:
                # Submit prompt if only Enter is pressed
                self.submitPrompt()
                return # Consume the event
        # Default handling for other keys
        super(TextEditor, self.input_field).keyPressEvent(event)

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
            json_path = os.path.join(os.path.dirname(__file__), "templates/nlg_greetings_templates_en.json")
            with open(json_path, "r", encoding="utf-8") as f:
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

    def _init_base_ui(self):
        """Set up minimal UI elements that don't depend on agents."""
        self.setWindowTitle("SLAI Interface")
        self.setMinimumSize(1024, 768)
        self.statusBar().showMessage("Loading agents in background...")

        # You can pre-render panels, logs, or splash visuals here
        # Example: set up a placeholder chat or loading widget
        from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

        placeholder = QLabel("🧠 Initializing SLAI Core...\nPlease wait.", self)
        placeholder.setStyleSheet("font-size: 18px; color: gray;")
        placeholder.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(placeholder)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _deferred_ui_setup(self):
        # Load heavy resources here after UI becomes visible
        # self._load_charts()
        self._init_visualization_engine()
        self._connect_analytics()
        run_in_thread(self._load_agents_and_finalize_ui)

    def _load_agents_and_finalize_ui(self):
        try:
            # Load config.yaml
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            self.agent_factory = AgentFactory(
                config=config,
                shared_resources={
                    "shared_memory": self.shared_memory,
                    "optimizer": self.optimizer
                }
            )
            self.collaborative_agent = CollaborativeAgent(
                shared_memory=self.shared_memory,
                agent_factory=self.agent_factory,
                agent_network=self.agent_factory.registry,
                config_path="config.yaml"
            )

            from src.agents.evaluation_agent import EvaluationAgent

            self.evaluation_agent = EvaluationAgent(
                shared_memory=self.shared_memory,
                agent_factory=self.factory,
                config=config
            )
            self.idle_audit = IdleAuditManager(
                shared_memory=self.shared_memory,
                agent_factory=self.agent_factory,
                agent=self.evaluation_agent,
                target_path=".",
                idle_threshold=120
            )
            self.idle_audit.start()

            # Link audit signal to display in log panel
            if hasattr(self.idle_audit.auditor, 'audit_signal'):
                self.idle_audit.auditor.audit_signal.connect(lambda path, msg: self.log_panel.append(msg))
            
            # Optional: Change status light on audit fail
            def audit_status_feedback(path, msg):
                if "Undefined" in msg or "anomaly" in msg or "Parse error" in msg:
                    self.set_status_indicator("critical")
                else:
                    self.set_status_indicator("warning")
            
            self.idle_audit.auditor.audit_signal.connect(audit_status_feedback)

            # If you want to pass these back to the UI thread later, use QMetaObject
            print("[MainWindow] AgentFactory + CollaborativeAgent loaded successfully.")

        except Exception as e:
            logger.error(f"[MainWindow] Failed to load agents in thread: {e}")

#========================Sidebar button ===================================
    def toggle_sidebar_button(self, model_key, label, button, normal_icon_path, checked_icon_path):
        if self.active_sidebar_button and self.active_sidebar_button != button:
            active_key = next((k for k, v in self.sidebar_button_map.items() if v == self.active_sidebar_button), None)
            if active_key == 'train':
                self.stop_active_training()
            self.active_sidebar_button.setChecked(False)

        if model_key != "audit":
            if self.active_sidebar_button and self.active_sidebar_button != button:
                self.active_sidebar_button.setChecked(False)
                self.active_sidebar_button.setIcon(QIcon(normal_icon_path))

            button.setChecked(True)
            button.setIcon(QIcon(checked_icon_path))
            self.active_sidebar_button = button
            self.load_model(model_key, label, button, normal_icon_path, checked_icon_path)  # Pass all arguments
            return
        # Special handling for Auditor Model
        reply = QMessageBox.question(
            self, 'Start Audit?',
            "Do you want to start the audit now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.audit_scheduler.start_audit_now()
        else:
            self.audit_scheduler.schedule_audit()

    def load_model(self, model_key, label, button, normal_icon_path, checked_icon_path):
        """Load the selected model"""
        from models.slai_lm import get_shared_slailm
        base_model = get_shared_slailm(self.shared_memory, agent_factory=self.factory)
        self.current_musician_model = None

        # Home Page
        if model_key == "home":
            # This deactivates any active models and those running in the background.
            # Whe User clicks on the home page button, it will deactivate the current active model automatically,
            # than it will show user the models that are running in the backround and asks: Do you want to close [model(s) name]
            pass

        # Researcher Model
        if model_key == "research":
            from models.research_model import ResearchModel
            model = ResearchModel(shared_memory=self.shared_memory, agent_factory=self.factory)
            self.agent = model
            self.show_status_message(f"{label} loaded.", 3000)

        # Musician Model
        elif model_key == "music":
            from models.musician import Musician
            model = Musician(shared_memory=self.shared_memory, agent_factory=self.factory,
                             expert_level="beginner", instrument="piano", parent=self)
            if getattr(model, 'defer_setup', False):
                model.interactive_setup()

            self.current_musician_model = model # <<< Store the instance
            self.agent = model # <<< Set Musician as the active agent (or handle differently if needed)
            self.show_status_message(f"{label} loaded. Editor connected.", 3000)

            # <<< --- CONNECT SIGNALS --- >>>
            if self.music_editor and self.current_musician_model:
                print("Connecting MusicEditor signals to Musician slots...")
                self.post_to_output_area("Connecting MusicEditor signals to Musician slots...", "Musician Model")
                # Connect editor's generateRequested signal to musician's slot
                self.music_editor.generateRequested.connect(self.current_musician_model.handle_generate_request)
                # Connect editor's tempoChanged signal to musician's slot
                self.music_editor.tempoChanged.connect(self.current_musician_model.handle_tempo_change)
                # Add connections for other signals here
                self.music_editor.keyChanged.connect(self.current_musician_model.handle_key_change)
                print("Connections established.")
                self.post_to_output_area("Connections established.", "Musician Model")
            else:
                print("Warning: MusicEditor or Musician model not available for connection.")
                self.post_to_output_area("Connecting MusicEditor signals to Musician slots...", "Musician Model")
                logger.warning("MusicEditor or Musician model not available for connection.")

            from models.music.music_setup_dialog import MusicianSetupDialog
            setup_dialog = MusicianSetupDialog(self)
            if setup_dialog.exec_() == QDialog.Accepted:
                level, instrument = setup_dialog.get_choices()
            else:
                return
    
            # Create Musician with collected parameters
            model = Musician(
                shared_memory=self.shared_memory,
                agent_factory=self.factory,
                expert_level=level.lower(),
                instrument=instrument.lower(),
                parent=self
            )
            self.current_musician_model = model
            self.agent = model
            self.post_to_output_area(f"{label} loaded. Editor connected.", "Musician Model")
            self.post_music_instructional_message()

        # Training Model
        elif model_key == "train":
            from models.train import SLAITrainer
            from models.training.synonym_trainer import SynonymTrainerController

            self.synonym_controller = SynonymTrainerController(language_editor=self.training_logs)
            self.training_logs.nextBatchClicked.connect(self.synonym_controller.next_batch)
            self.training_logs.approveBatchClicked.connect(self.synonym_controller.approve_batch)
            self.training_logs.editor_signals.term_decision.connect(self.synonym_controller.handle_term_update)

            if self.synonym_controller:
                self.show_status_message(f"Starting Training Model...", 3000)
                self.post_train_instructional_message()
                self.right_tab_widget.setCurrentWidget(self.training_logs) # Switch to the editor tab
                # Start the process (ensure controller and UI are fully initialized)
                QTimer.singleShot(100, self.synonym_controller.start_training)
            else:
                self.show_status_message(f"Error: Synonym Trainer Controller not initialized.", 5000)
                logger.error("Synonym Trainer Controller not initialized.")

            class TrainerThread(QThread):
                def __init__(self, factory, training_logs, config, parent=None):  # Add config param
                    super().__init__(parent)
                    self.factory = factory
                    self.training_logs = training_logs
                    self.config = config 

                def run(self):  # Remove config param from run()
                    # Now access config via self.config
                    from src.collaborative.shared_memory import SharedMemory
                    from src.agents.perception_agent import PerceptionAgent
                    from models.slai_lm import get_shared_slailm
            
                    dummy_memory = SharedMemory(self.config['shared_resources']['memory'])
                    dummy_agent = PerceptionAgent(
                        config={"modalities": ["text"]},
                        shared_memory=dummy_memory,
                        agent_factory=self.factory
                    )
                    dummy_agent.slai_lm = get_shared_slailm(dummy_memory, self.factory)
                    trainer = SLAITrainer(
                        dummy_agent, dummy_memory,
                        "dummy target", "dummy response",
                        self.factory, self.training_logs.editor_signals)
                    trainer.run()

            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            self.trainer_thread = TrainerThread(
                factory=self.factory,
                training_logs=self.training_logs,
                config=config)
            # ✅ Connect correct training signal
            from models.training.shared_signals import training_signals
            training_signals.approval_request.connect(self.training_logs.show_approval_dialog)
            training_signals.batch_continuation.connect(self.training_logs.update_batch_counter)
            training_signals.log_message.connect(self.training_logs.append_log)

            training_signals.batch_continuation.connect(
                lambda p, r: self.training_progress.setValue(int((p/len(words))*100)))
            
            self.trainer_thread.start()
            self.right_tab_widget.setCurrentIndex(1)
            self.show_status_message(f"{label} loaded. Training started in background.", 3000)
            self.post_train_instructional_message()
    
    def stop_active_training(self):
         if self.synonym_controller and self.synonym_controller.is_running:
             self.synonym_controller.stop_training()

    def closeEvent(self, event):
        # Ensure training stops gracefully on close
        self.stop_active_training()
        # ... (any other cleanup) ...
        super().closeEvent(event)

    def post_music_instructional_message(self):
        instructional_text = (
            "<b><hr><br>PROMPT INSTRUCTIONS:</b><br>"
            "Music Theory:"
                "<ul>"
                "<li>Start prompt by typing <i>‘Explain’</i> followed the <i>‘music theory’</i></li><br>"
                "</ul>"
            "Music Generation:"
                "<ul>"
                "<li>To generate an AI music composition prompt should be ordered as follows:</li>"
                "<li><i>Compose | number of bars | time signature as n/n | tempo</i></li>"
                "</ul><hr>"
            ""
        )
        self.output_area.append(instructional_text)

    def post_train_instructional_message(self):
        instructional_text = (
            "<b>TRAINING INSTRUCTIONS:</b>"
                "<ul>"
                "<li>The model loads words from the structured wordlist.</li>"
                "<li>For each word, it updates or generates new <i>synonyms</i> and <i>related_terms</i>.</li>"
                "<li>Batches of words are loaded according to the user's selected batch size.</li>"
                "<li>You review each updated entry and either approve, reject, or edit the suggestions.</li>"
                "<li>The bottom navigation allows you to move to the next, previous, repeat, shuffle, or finalize batch approvals.</li>"
                "<li>Once approved, the updates are saved back into the structured wordlist database.</li>"
                "</ul><hr />"
            "Promts:"
                "<ul>"
                "<li>User can edit a specific word by typeing <i>Edit</i> followed by the desired word</li>"
                "<li>In this way, the user can even edit a word that has already been approved</li>"
                "</ul><hr>"
            ""
        )
        self.output_area.append(instructional_text)

    def update_batch_counter(self, current, total):
        from models.traiing.synonym_trainer import BATCH_SIZE
        self.batch_counter.setText(f"Processing: Batch {current//BATCH_SIZE+1}/{(total//BATCH_SIZE)+1}")

    def handle_approval_prompt(self, word, item, category):
        from models.train import log_emitter
        decision, ok = QInputDialog.getText(self, f"{category.capitalize()} Approval", f"{word}: Approve or correct '{item}'?")
        if ok:
            log_emitter.approval_result.emit(item, decision.strip().lower())

    def handle_batch_continuation(self, continue_info):
        from models.training.shared_signals import training_signals
        msg = (
            f"Processed {continue_info['processed']} words\n"
            f"Remaining: {continue_info['remaining']}\n"
            "Continue to next batch?"
        )
        
        # Show dialog with Continue/Cancel buttons
        choice = QMessageBox.question(
            self, 
            "Continue Training?",
            msg,
            QMessageBox.Yes | QMessageBox.No
        )
        
        # Emit signal back to trainer thread
        training_signals.batch_continuation_result.emit(choice == QMessageBox.Yes)
#===========================================================================================

    def load_audio(self, file_path):
        from models.music.audio_utils import AudioUtils
        return AudioUtils.reshape_for_model(AudioUtils.load_audio(file_path))
    
    def analyze_audio(self, audio_path):
        waveform = self.load_audio(audio_path)
        input_dict = {'audio': waveform}
        if hasattr(self.agent, 'forward'):
            embedding = self.agent.forward(input_dict)
            self.output_area.append(f"<b>Audio Embedding:</b><br>{embedding}")

    def play_audio(self, audio_path):
        """Add buffer size validation"""
        try:
            # Validate file first
            if not os.path.exists(audio_path):
                raise FileNotFoundError
    
            # Use dedicated audio thread
            self.audio_thread = QtCore.QThread()
            self.audio_worker = AudioWorker(audio_path)
            self.audio_worker.moveToThread(self.audio_thread)
            self.audio_thread.started.connect(self.audio_worker.play)
            self.audio_thread.start()
    
        except Exception as e:
            logging.error(f"Audio playback failed: {str(e)}")
            self.show_status_message(f"Audio error: {str(e)}", 3000)
    
    class AudioWorker(QtCore.QObject):
        def __init__(self, path):
            super().__init__()
            self.path = path
    
        def play(self):
            try:
                # Use pydub for better buffer management
                from pydub import AudioSegment
                sound = AudioSegment.from_file(self.path)
                sound.export("temp.wav", format="wav")
                os.system("start temp.wav")
            except Exception as e:
                logging.error(str(e))

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

        # self.input_field.clear()

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
        #self.output_area.append()
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
        """Exports chat and system session data into a styled PDF using generate_pdf_report."""
        from bs4 import BeautifulSoup
        from frontend.templates.generate_pdf_report import generate_pdf_report

        # Calculate session duration
        session_duration = datetime.now() - self.session_start_time
        hours, remainder = divmod(session_duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

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
            from models.research_model import ResearchModel
            if isinstance(self.agent, ResearchModel):
                output = self.agent.run("EXPORT LAST PROMPT")  # Adjust this if prompt is cached differently
                research_data = {
                    "summary": output["summary"],
                    "response": output["response"],
                    "sources": output["sources"],
                    "time_taken": output["time_taken"],
                    "follow_up_question": output["follow_up_question"],
                    "agent_trace": output["agent_trace"],
                    "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "generated_by": getpass.getuser(),
                    "country": "Netherlands",
                    "watermark_logo_path": "frontend/assets/logo1.png",
                    "cover_logo_path": "frontend/assets/logo.png",
                }
                generate_research_pdf_report(save_path, "Research Result", research_data)
            else:
                # Collect dynamic chat logs and sanitize HTML
                chat_text = self.output_area.toHtml()
                
                # Convert HTML to plain text and strip all tags
                soup = BeautifulSoup(chat_text, "html.parser")
                clean_text = soup.get_text('\n')  # Convert <br> to newlines
                
                # Additional cleanup for HTML entities and special characters
                clean_text = clean_text.replace('&nbsp;', ' ').replace('<', '[').replace('>', ']')
                
                # Capitalize first letter of each line
                cleaned_lines = []
                for line in clean_text.split('\n'):
                    if line.strip():
                        cleaned_lines.append(line[0].upper() + line[1:])
                    else:
                        cleaned_lines.append('')

                # Build data structure expected by generate_pdf_report
                data = {
                    "chat_history": '\n'.join(cleaned_lines),  # Use sanitized text
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
                    "date_range": f"{self.session_start_time.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
                    "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "generated_by": getpass.getuser(),
                    "country": "Netherlands",
                    "watermark_logo_path": "frontend/assets/logo1.png",
                    "cover_logo_path": "frontend/assets/logo.png",
                }

                generate_pdf_report(save_path, data)
            self.show_status_message(f"PDF report saved: {os.path.basename(save_path)}", 4000)

        except Exception as e:
            logger.exception("Failed to generate PDF report")
            self.set_status_indicator("error")
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
        #text = self.input_field.toPlainText().strip()
        user_input = self.input_field.toPlainText().strip()
        if not user_input:
            return
    
        self.post_to_output_area(user_input, tag="User")
        self.input_field.clear()

        response = check_nlg_trigger(user_input, self.nlg_templates)
        if response:
            self.post_to_output_area(response, tag="SLAI")
            return

        self.post_to_output_area("Generating response...", tag="SLAI")
        self.current_response_start = datetime.now()

        # Start a prompt thread to get response
        self.prompt_thread = PromptThread(self.agent, user_input, task_data=None)
        self.prompt_thread.result_ready.connect(self.displayResponse)
        self.prompt_thread.start()

        if user_input:
            self.set_status_indicator("busy") # Set RED light
        #    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
        #    self.output_area.append(f"{timestamp}<font color='lightblue'>User:</font> {text}<br>")
        #    #self.output_area.append(f"<font color='lightblue'>User:</font> {text}<br>") # Display user prompt
        #   #self.input_field.clear() # Clear input after sending
        #    self.show_status_message("Sending prompt to SLAI...", 2000)

        #    # Create task_data with default context
        #    task_data = {"context": "default", "prompt": text}
        #    QtCore.QTimer.singleShot(100, lambda: self.call_slai_pipeline(task_data, text))

    def displayResponse(self, response_text):
        self.post_to_output_area(response_text, tag="SLAI")

    def call_slai_pipeline(self, task_data: dict, prompt: str):
        thread = PromptThread(self.agent, prompt, task_data)
        thread.result_ready.connect(lambda result: self.output_area.setPlainText(result))
        thread.start()

        try:
            self.current_response_start = datetime.now() # Start response timer

            slai_response = self.agent.generate(prompt, task_data) # Call the generate method of the imported CollaborativeAgent instance

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
            logger.exception(f"Error during SLAI pipeline execution for prompt: {prompt}")
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
        """Update heartbeat visual and heartbeat count display"""
        if not hasattr(self, "heartbeat_count"):
            self.heartbeat_count = 0
            self.heartbeat_flash = False

        self.heartbeat_count += 1
        self.heartbeat_flash = not self.heartbeat_flash

        # Toggle between green and grey
        target_color = "#00bf63" if self.heartbeat_flash else "grey"
        self.indicator_heartbeat.set_status(target_color)

        # Format uptime
        session_duration = datetime.now() - self.session_start_time
        total_seconds = int(session_duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

        # Compose heartbeat log (live-updating)
        line = f"System Heartbeat | Active: {formatted_time} | Beats: {self.heartbeat_count}"

        # Overwrite last line
        cursor = self.log_panel.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.select(QTextCursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.insertText(line)
        self.log_panel.moveCursor(QTextCursor.End)
    
    def show_status_message(self, message, duration_ms):
        """ Displays a status message for a specified duration """
        self.status_message.setText(message)
        self.status_message.show()
        QTimer.singleShot(duration_ms, self.status_message.hide)

    def _init_visualization_engine(self):
        # Placeholder for future chart setup (e.g., performance graph)
        pass

    def _connect_analytics(self):
        # Placeholder for future chart setup (e.g., performance graph)
        pass

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
