import os, sys
import psutil
import GPUtil
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QFrame, QSizePolicy
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon, QTextCursor, QColor, QPainter
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect, QVariantAnimation

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SLAI Launcher")
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "..", "frontend", "assets", "logo.ico")))
        self.setGeometry(100, 100, 1366, 768)
        self.setStyleSheet("font-family: Times New Roman; background-color: black; color: white;")
        self.status_message = QtWidgets.QLabel()
        self.loading_animation = None
        self.color_animation = None
        self.current_ball_color = QColor("#FFD700")
        
        self.typing_texts = ["SLAI", "Scalable Learning Autonomous Intelligence"]
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
        self.header_label = QtWidgets.QLabel()
        self.header_label.setStyleSheet("font-family: Times New Roman; font-size: 18px; font-weight: bold; color: gold;")
        self.center_panel.addWidget(self.header_label)

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

    def paintEvent(self, event):
        if self.bouncing_ball.isVisible():
            painter = QPainter(self.bouncing_ball)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QtGui.QBrush(self.current_ball_color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(0, 0, 30, 30)

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
        self.color_animation.valueChanged.connect(lambda color: setattr(self, 'current_ball_color', color))
        self.color_animation.setStartValue(QColor("#FFD700"))
        self.color_animation.setKeyValueAt(0.5, QColor("#fefefe"))
        self.color_animation.setEndValue(QColor("#FFD700"))
        
        self.current_ball_color = QColor("#FFD700")
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
        
        # Replace this with actual response checking
        response_found = len(text) % 2 == 0  # Example condition
        
        current_output = self.tabs[self.tab_stack.currentIndex()]
        if response_found:
            current_output.append(f"\nUser: {text}\nSLAI: Response for '{text}'...")
        else:
            self.status_message.setText("SLAI was unable to provide a response.")
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
            current_output = self.tabs[self.tab_stack.currentIndex()]
            current_output.append(f"\nUser: {text}\nSLAI: [Response generated here...]")
            self.input_field.clear()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    launcher = MainWindow()
    launcher.showMaximized()
    sys.exit(app.exec_())
