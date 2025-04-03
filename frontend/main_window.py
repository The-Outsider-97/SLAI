import os, sys
import psutil
import GPUtil
import torch
import logging
import subprocess

from torch.nn.functional import cosine_similarity
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QFrame, QSizePolicy, QFileDialog
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon, QTextCursor, QColor, QPainter
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect, QVariantAnimation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from src.agents.collaborative_agent import CollaborativeAgent

class DynamicTextEdit(QTextEdit):
    heightChanged = pyqtSignal(int)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.document().documentLayout().documentSizeChanged.connect(self.updateHeight)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMaximumHeight(150)
        self.textChanged.connect(self.keep_cursor_bottom)
        
    def updateHeight(self):
        doc_height = self.document().size().height()
        new_height = int(doc_height + 45)  # Reduced padding
        
        # Constrain by container's maximum height
        max_h = min(350, self.parent().parent().maximumHeight()) 
        final_height = min(new_height, max_h)
        
        self.setMinimumHeight(final_height)
        self.setMaximumHeight(final_height)
        self.heightChanged.emit(final_height)
        
    def keep_cursor_bottom(self):
        """Ensures text stays anchored to bottom"""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, log_queue, metric_queue):
        super().__init__()
        self.log_queue = log_queue
        self.metric_queue = metric_queue
        self.agent = CollaborativeAgent(
            agent_network={
                'LanguageAgent': {
                    'type': 'nlp',
                    'capabilities': ['text_generation'],
                    'components': ['language_model']
                }
            },
            risk_threshold=0.35
        )
        self.setWindowTitle("SLAI Launcher")
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "..", "frontend", "assets", "logo.ico")))
        self.setGeometry(100, 100, 1366, 768)
        self.setStyleSheet("font-family: Times New Roman; background-color: black; color: white;")
        self.status_message = QtWidgets.QLabel()
        self.loading_animation = None
        self.color_animation = None
        self.bouncing_ball = QtWidgets.QLabel(self)
        self.bouncing_ball.setFixedSize(30, 30)
        self.bouncing_ball.setStyleSheet("""
            background-color: #FFD700;
            border-radius: 15px;
            border: 2px solid #FFFFFF;
        """)
        self.bouncing_ball.hide()
        
        self.typing_texts = ["Scalable Learning Autonomous Intelligence ", "SLAI"]
        self.typing_index = 0
        self.char_index = 0
        self.display_text = ""
        self.is_pausing = False
        self.pause_counter = 0

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        self.left_panel = QtWidgets.QVBoxLayout()
        self.center_panel = QtWidgets.QVBoxLayout()
        self.right_panel = QtWidgets.QVBoxLayout()

        self.main_layout.addLayout(self.left_panel, 1)
        self.main_layout.addLayout(self.center_panel, 2)
        self.main_layout.addLayout(self.right_panel, 2)

        self.initUI()
        self.initTimer()

    def initUI(self):
        # Header with typing animation
        self.header_label = QtWidgets.QLabel("SLAI")
        self.header_label.setStyleSheet("""
            font-family: Times New Roman;
            font-size: 22px;
            font-weight: bold;
            color: gold;
        """)

        # Header with Control Button
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addWidget(self.header_label)

        # Save Button
        self.save_button = QtWidgets.QPushButton("Save Logs")
        self.save_button.setFixedSize(125, 30)
        self.save_button.setStyleSheet("background-color: gold; color: black; font-weight: bold; border-radius: 6px;")
        self.save_button.clicked.connect(self.save_logs)

        # Load Button
        self.load_button = QtWidgets.QPushButton("Load Chat")
        self.load_button.setFixedSize(125, 30)
        self.load_button.setStyleSheet("background-color: gold; color: black; font-weight: bold; border-radius: 6px;")
        self.load_button.clicked.connect(self.load_chat)

        # Clear Button
        self.clear_button = QtWidgets.QPushButton("Clear Chat")
        self.clear_button.setFixedSize(125, 30)
        self.clear_button.setStyleSheet("background-color: #fefefe; color: 0e0e0e; font-weight: bold; border-radius: 6px;")
        self.clear_button.clicked.connect(self.clear_chat)

        # Add Buttons to Header
        header_layout.addStretch()
        header_layout.addWidget(self.save_button)
        header_layout.addWidget(self.clear_button)
        header_layout.addWidget(self.load_button)

        header_container = QtWidgets.QWidget()
        header_container.setLayout(header_layout)
        self.center_panel.addWidget(header_container)

        self.typing_timer = QtCore.QTimer()
        self.typing_timer.timeout.connect(self.updateTyping)
        self.typing_timer.start(100)

        # Sidebar buttons
        for i in range(4):
            btn = QtWidgets.QPushButton()
            btn.setFixedSize(40, 40)
            btn.setStyleSheet("border-radius: 20px; background-color: gold;")
            self.left_panel.addWidget(btn)
            btn.clicked.connect(lambda _, x=i: self.switchTab(x))

        self.left_panel.addStretch()

        # Tab Stack with padding matching system logs
        self.tab_stack = QtWidgets.QStackedWidget()
        self.tab_stack.setStyleSheet("padding-top: 6px; padding-bottom: 6px;")
        self.center_panel.addWidget(self.tab_stack, 1)

        self.tabs = []
        for i in range(4):
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            output_area = QtWidgets.QTextEdit()
            output_area.setReadOnly(True)
            output_area.setStyleSheet("font-family: Times New Roman; font-size: 14px; color: white; background-color: transparent; border: 1px solid gray; border-radius: 6px;")
            output_area.setText(f"Prompt output for module {i + 1}...")
            layout.addWidget(output_area)
            self.tabs.append(output_area)
            self.tab_stack.addWidget(tab)

        # System Logs
        self.system_logs = QtWidgets.QTextEdit()
        self.system_logs.setReadOnly(True)
        self.system_logs.setStyleSheet("font-family: Times New Roman; font-size: 12px; color: white; background-color: transparent; margin-top: 10px; border: 1px solid gray; border-radius: 6px; padding-top: 6px; padding-bottom: 6px;")
        self.system_logs.setText("System process logs...")
        self.right_panel.addWidget(self.system_logs, 1)

        # Dynamic input field
        self.input_field = DynamicTextEdit()
        self.input_field.setPlaceholderText("Type something here...")
        self.input_field.setStyleSheet("""
            QTextEdit {
                padding: 15px; 
                border: 1px solid gray; 
                background-color: #0e0e0e; 
                border-radius: 6px;
                font: 14px;
            }
        """)
        self.input_field.setMaximumHeight(100)
        self.input_field.setMinimumHeight(45)

        def handleKeyPress(event):
            if event.key() == QtCore.Qt.Key_Return:
                if event.modifiers() & QtCore.Qt.ShiftModifier:
                    self.input_field.textCursor().insertText("\n")
                else:
                    self.submitPrompt()
                    return
            super(DynamicTextEdit, self.input_field).keyPressEvent(event)
            
        self.input_field.keyPressEvent = handleKeyPress
        input_container = QtWidgets.QWidget()
        input_container.setStyleSheet("background: transparent;")
        input_layout = QtWidgets.QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 10)
        self.center_panel.setSpacing(10)
        input_layout.addStretch(1)
        input_layout.addWidget(self.input_field)
        
        self.center_panel.addWidget(input_container)

        # Create a container for the entire center content
        center_container = QtWidgets.QWidget()
        center_container.setLayout(QtWidgets.QVBoxLayout())
        center_container.layout().setContentsMargins(0, 0, 0, 0)
        center_container.layout().setSpacing(0)

        # Footer
        self.footer = QtWidgets.QLabel()
        self.footer.setStyleSheet("font-size: 11px; color: lightgray; padding-top: 6px;")
        self.right_panel.addWidget(self.footer)

        # Status message label
        self.status_message.setStyleSheet("color: gold; font-family: Times New Roman; font-size: 14px;")
        self.status_message.hide()
        self.center_panel.addWidget(self.status_message)

        # Bouncing ball
        self.bouncing_ball = QtWidgets.QLabel(self)
        self.bouncing_ball.setFixedSize(30, 30)
        self.bouncing_ball.hide()

    def save_logs(self):
        current_tab = self.tabs[self.tab_stack.currentIndex()]
        chat_text = current_tab.toPlainText()
        system_text = self.system_logs.toPlainText()

        logs_dir = os.path.join("logs", "chat_logs")
        os.makedirs(logs_dir, exist_ok=True)

        default_filename = os.path.join(logs_dir, "chatlog.txt")
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Logs", default_filename, "Text Files (*.txt);;All Files (*)")

        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as file:
                    file.write(""
                    "=== Chat Log ===\n"
                    "________________\n")
                    file.write(chat_text + "\n\n")
                    file.write("=== System Logs ===\n")
                    file.write(system_text)
                self.status_message.setText("Logs saved successfully.")
                self.status_message.show()
                QTimer.singleShot(3000, self.status_message.hide)
            except Exception as e:
                self.status_message.setText(f"Failed to save logs: {e}")
                self.status_message.show()
                QTimer.singleShot(5000, self.status_message.hide)

    def load_chat(self):
        logs_dir = os.path.join("logs", "chat_logs")
        os.makedirs(logs_dir, exist_ok=True)

        file_path, _ = QFileDialog.getOpenFileName(self, "Load Chat Log", logs_dir, "Text Files (*.txt);;All Files (*)")

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                current_tab = self.tabs[self.tab_stack.currentIndex()]
                current_tab.setPlainText(content)
                self.status_message.setText(f"Loaded: {os.path.basename(file_path)}")
                self.status_message.show()
                QTimer.singleShot(3000, self.status_message.hide)
            except Exception as e:
                self.status_message.setText(f"Failed to load file: {e}")
                self.status_message.show()
                QTimer.singleShot(5000, self.status_message.hide)

    def clear_chat(self):
        current_tab = self.tabs[self.tab_stack.currentIndex()]
        current_tab.clear()
        self.status_message.setText("Chat cleared.")
        self.status_message.show()
        QTimer.singleShot(2000, self.status_message.hide)

    def start_loading_animation(self):
        # Position the ball above input field
        ball_x = self.input_field.x() + self.input_field.width()//2 - 15
        ball_y = self.input_field.y() - 50
        self.bouncing_ball.move(ball_x, ball_y)
        self.bouncing_ball.show()
        
        # Bouncing animation
        self.loading_animation = QPropertyAnimation(self.bouncing_ball, b"pos")
        self.loading_animation.setDuration(1000)
        self.loading_animation.setLoopCount(-1)
        self.loading_animation.setEasingCurve(QEasingCurve.OutBounce)
        self.loading_animation.setStartValue(QRect(ball_x, ball_y, 30, 30).topLeft())
        self.loading_animation.setEndValue(QRect(ball_x, ball_y + 40, 30, 30).topLeft())
        
        # Color animation
        self.color_animation = QVariantAnimation()
        self.color_animation.setDuration(2000)
        self.color_animation.setLoopCount(-1)
        self.color_animation.valueChanged.connect(lambda color: 
            self.bouncing_ball.setStyleSheet(f"""
                background-color: {color.name()};
                border-radius: 15px;
                border: 2px solid #FFFFFF;
            """)
        )
        self.color_animation.setStartValue(QColor("#FFD700"))
        self.color_animation.setKeyValueAt(0.5, QColor("#fefefe"))
        self.color_animation.setEndValue(QColor("#FFD700"))
        
        self.loading_animation.start()
        self.color_animation.start()

    def stop_loading_animation(self):
        if self.loading_animation:
            self.loading_animation.stop()
            self.color_animation.stop()
            self.bouncing_ball.hide()
            self.status_message.hide()

    def submitPrompt(self):
        text = self.input_field.toPlainText().strip()
        if text:
            self.start_loading_animation()
            
            # Simulate async processing
            QtCore.QTimer.singleShot(2000, lambda: self.handle_response(text))

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
            self.header_label.setText(self.display_text)
            self.char_index += 1
        else:
            self.is_pausing = True
            self.pause_counter = 0

    def initTimer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateSystemStats)
        self.timer.start(1000)

    def updateSystemStats(self):
        cpu = psutil.cpu_freq().current
        usage = psutil.cpu_percent()
        threads = psutil.cpu_count(logical=True)
        processes = len(psutil.pids())
        ram = psutil.virtual_memory()
        ram_used = ram.percent
        ram_available = ram.available // (1024 * 1024)

        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_usage = gpu.load * 100
            gpu_temp = gpu.temperature
            gpu_name = gpu.name
        else:
            gpu_name = "N/A"
            gpu_usage = 0
            gpu_temp = 0

        text = f"CPU: {cpu:.1f}MHz | Usage: {usage:.1f}% | Threads: {threads} | Processes: {processes}  "
        text += f"RAM: {ram_used:.1f}% Used, {ram_available}MB Available  |  GPU: {gpu_name} | Usage: {gpu_usage:.1f}% | Temp: {gpu_temp:.1f}C"

        self.footer.setText(text)

    def switchTab(self, index):
        current_index = self.tab_stack.currentIndex()
        if index != current_index:
            self.fadeOut(self.tab_stack.currentWidget())
            self.tab_stack.setCurrentIndex(index)
            self.fadeIn(self.tab_stack.currentWidget())

    def fadeOut(self, widget):
        effect = QtWidgets.QGraphicsOpacityEffect()
        widget.setGraphicsEffect(effect)
        anim = QtCore.QPropertyAnimation(effect, b"opacity")
        anim.setDuration(300)
        anim.setStartValue(1)
        anim.setEndValue(0)
        anim.start()

    def fadeIn(self, widget):
        effect = QtWidgets.QGraphicsOpacityEffect()
        widget.setGraphicsEffect(effect)
        anim = QtCore.QPropertyAnimation(effect, b"opacity")
        anim.setDuration(300)
        anim.setStartValue(0)
        anim.setEndValue(1)
        anim.start()

    def submitPrompt(self):
        text = self.input_field.toPlainText().strip()
        if text:
            self.start_loading_animation()
            self.status_message.setText("Sending prompt to SLAI...")
            self.status_message.show()

            # Direct agent call instead of subprocess
            QtCore.QTimer.singleShot(100, lambda: self.call_slai_pipeline(text))

    def call_slai_pipeline(self, prompt: str):
        current_output = self.tabs[self.tab_stack.currentIndex()]

        try:
            slai_response = self.agent.generate(prompt)
        except Exception as e:
            slai_response = f"[ERROR]: {e}"

        current_output.append(f"\nUser: {prompt}\nSLAI: {slai_response}")
        self.input_field.clear()
        self.stop_loading_animation()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    launcher = MainWindow()
    launcher.showMaximized()
    sys.exit(app.exec_())
