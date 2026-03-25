import json
import math
import sys
import subprocess

from pathlib import Path
from datetime import datetime

from PyQt5.QtCore import QPointF, QRectF, QRect, QPoint, Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QEvent
from PyQt5.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QBrush,
    QRadialGradient,
    QRegion,
    QTransform
)
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QWidget,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QFrame,
    QVBoxLayout,
    QDialog,
    QLineEdit,
    QMessageBox,
    QHBoxLayout,
    QTextEdit,
)

from src.functions.dropdown import DropdownMenu, DropdownOption, AnimationConfig
from src.functions.auth import AuthService
from src.functions.search import SearchEngine
from monitoring.metrics_collector import MetricsCollector
from monitoring.health_check import HealthChecker
from monitoring.drift_detection import DriftDetector


SL_YELLOW = QColor("#eacb00")
BG_DARK = QColor("#0e1012")
TEXT_WHITE = QColor("#ffffff")
RADIUS = 480
APP_SIZE = 200
APP_DISTANCE = 25
ITEM_SPACING = 30
MAX_ROTATION = 180


class HamburgerIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(36, 24)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(TEXT_WHITE)
        painter.drawRoundedRect(0, 0, 36, 4, 2, 2)
        painter.drawRoundedRect(0, 10, 36, 4, 2, 2)
        painter.drawRoundedRect(0, 20, 36, 4, 2, 2)


class SearchIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(26, 26)
        self.setCursor(Qt.PointingHandCursor)
        self.is_hovered = False

    def enterEvent(self, e):
        self.is_hovered = True
        self.update()

    def leaveEvent(self, e):
        self.is_hovered = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = SL_YELLOW if self.is_hovered else TEXT_WHITE
        pen = QPen(color, 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(11, 11), 8, 8)
        painter.drawLine(QPointF(21, 21), QPointF(16.65, 16.65))

class LoginDialog(QDialog):
    def __init__(self, auth_service: AuthService, parent=None):
        super().__init__(parent)
        self.auth_service = auth_service
        self.setWindowTitle("Login")
        self.setModal(True)
        self.setFixedSize(380, 230)
        self.setStyleSheet(
            """
            QDialog { background-color: #12161a; color: white; }
            QLabel { color: white; font-family: Georgia; }
            QLineEdit {
                background: #0e1012;
                border: 1px solid #5c5c5c;
                border-radius: 6px;
                color: white;
                padding: 6px;
            }
            QPushButton {
                border-radius: 8px;
                padding: 6px 10px;
                font-family: Georgia;
            }
            """
        )

        layout = QVBoxLayout(self)
        title = QLabel("Log in to SLAI Hub")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #eacb00;")
        layout.addWidget(title)

        self.username_input = QLineEdit(self)
        self.username_input.setPlaceholderText("Username")
        self.password_input = QLineEdit(self)
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_input)

        button_row = QHBoxLayout()
        login_btn = QPushButton("Log in")
        signup_btn = QPushButton("Sign up")
        forgot_btn = QPushButton("Forgot password?")
        login_btn.clicked.connect(self._login)
        signup_btn.clicked.connect(self._signup)
        forgot_btn.clicked.connect(self._forgot_password)
        button_row.addWidget(login_btn)
        button_row.addWidget(signup_btn)
        button_row.addWidget(forgot_btn)
        layout.addLayout(button_row)

    def _login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        if not username or not password:
            QMessageBox.warning(self, "Missing fields", "Please enter username and password.")
            return
        try:
            token = self.auth_service.log_in(username, password)
            QMessageBox.information(self, "Success", f"Logged in.\nToken ends at: {token.expires_at.isoformat()}")
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, "Login failed", str(exc))

    def _signup(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        if not username or not password:
            QMessageBox.warning(self, "Missing fields", "Please enter username and password.")
            return
        try:
            self.auth_service.sign_up(username, password)
            QMessageBox.information(self, "Account created", "Sign up successful. You can now log in.")
        except Exception as exc:
            QMessageBox.warning(self, "Sign up failed", str(exc))

    def _forgot_password(self):
        QMessageBox.information(
            self,
            "Forgot password",
            "Password reset flow can be connected here.\nFor now, please sign up with a new account.",
        )

class LoginButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__("login", parent)
        self.setFixedSize(110, 38)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                border: 2px solid white;
                border-radius: 10px;
                background: transparent;
                color: white;
                font-family: Georgia;
                font-size: 18px;
                font-weight: bold;
                padding-left: 28px;
                padding-right: 12px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 25);
            }
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(12, 11)
        painter.scale(16 / 24, 16 / 24)
    
        path = QPainterPath()
        path.moveTo(18, 8); path.lineTo(17, 8); path.lineTo(17, 6)
        path.cubicTo(17, 3.24, 14.76, 1, 12, 1)
        path.cubicTo(9.24, 1, 7, 3.24, 7, 6)
        path.lineTo(7, 8); path.lineTo(6, 8)
        path.cubicTo(4.9, 8, 4, 8.9, 4, 10)
        path.lineTo(4, 20)
        path.cubicTo(4, 21.1, 4.9, 22, 6, 22)
        path.lineTo(18, 22)
        path.cubicTo(19.1, 22, 20, 21.1, 20, 20)
        path.lineTo(20, 10)
        path.cubicTo(20, 8.9, 19.1, 8, 18, 8)
        path.closeSubpath()
    
        path.addEllipse(QPointF(12, 15), 2, 2)
    
        path2 = QPainterPath()
        path2.moveTo(15.1, 8); path2.lineTo(8.9, 8); path2.lineTo(8.9, 6)
        path2.cubicTo(8.9, 4.29, 10.29, 2.9, 12, 2.9)
        path2.cubicTo(13.71, 2.9, 15.1, 4.29, 15.1, 6)
        path2.lineTo(15.1, 8)
        path2.closeSubpath()
        path.addPath(path2)
    
        # Set fill rule on the path, not the painter
        path.setFillRule(Qt.OddEvenFill)
    
        painter.setBrush(TEXT_WHITE)
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)


class SystemMonitoringDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("System Monitoring")
        self.setMinimumSize(780, 520)
        self.setStyleSheet(
            """
            QDialog { background-color: #12161a; color: white; }
            QLabel { color: white; font-family: Georgia; }
            QTextEdit {
                background: #0e1012;
                border: 1px solid #3f454c;
                border-radius: 10px;
                color: #ffffff;
                font-family: Consolas, monospace;
                font-size: 13px;
                padding: 10px;
            }
            QPushButton {
                border-radius: 8px;
                padding: 8px 14px;
                font-family: Georgia;
            }
            """
        )
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.drift_detector = DriftDetector()
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("SLAI System Monitoring")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #eacb00;")
        subtitle = QLabel("Realtime infrastructure snapshot and status summary.")
        subtitle.setStyleSheet("font-size: 14px; color: #c9c9c9;")
        self.body = QTextEdit(self)
        self.body.setReadOnly(True)

        button_row = QHBoxLayout()
        refresh_button = QPushButton("Refresh")
        close_button = QPushButton("Close")
        refresh_button.clicked.connect(self._refresh)
        close_button.clicked.connect(self.accept)
        button_row.addWidget(refresh_button)
        button_row.addStretch()
        button_row.addWidget(close_button)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.body, stretch=1)
        layout.addLayout(button_row)

    def _refresh(self):
        snapshot = self.metrics_collector.collect_snapshot()
        health_result = self.health_checker.run_checks(
            [
                {"name": "SLAI Dashboard", "type": "tcp", "host": "127.0.0.1", "port": 5000, "timeout": 1.0},
                {"name": "SLAI Hub HTTP", "type": "http", "url": "http://127.0.0.1:5000", "timeout": 1.5},
            ]
        )
        drift_result = self.drift_detector.detect(
            [snapshot.memory.percent_used, snapshot.cpu.percent_total],
            [snapshot.memory.percent_used, snapshot.cpu.percent_total],
            threshold=0.05,
            metric_name="runtime_stability",
        )
        self.body.setText(
            "\n".join(
                [
                    f"Generated: {datetime.now().isoformat()}",
                    "",
                    "[Metrics]",
                    snapshot.to_pretty_string(),
                    "",
                    "[Health]",
                    self.health_checker.format_report(health_result),
                    "",
                    "[Drift]",
                    self.drift_detector.format_result(drift_result),
                ]
            )
        )


class AppCard(QWidget):
    # Emitted when mouse enters or leaves the card. Passes (app_name, is_hovered)
    hover_changed = pyqtSignal(str, bool)
    clicked = pyqtSignal(str)

    def __init__(self, parent: QWidget, name: str, image_path: Path, angle: float) -> None:
        super().__init__(parent)
        self.name = name
        self.base_angle = angle
        self.absolute_angle = angle
        
        # Extended widget bounds to cleanly house both rotation & distant tooltips
        self.setFixedSize(500, 500)
        self.hovered = False
        self.setCursor(Qt.PointingHandCursor)

        raw_pixmap = QPixmap(str(image_path))
        if raw_pixmap.isNull():
            self.pixmap = QPixmap(APP_SIZE, APP_SIZE)
            self.pixmap.fill(QColor("#222222"))
        else:
            self.pixmap = raw_pixmap.scaled(
                APP_SIZE, APP_SIZE, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )

    def set_absolute_angle(self, angle: float) -> None:
        self.absolute_angle = angle
        
        # QRegion mask ensures perfect hit testing ONLY on the tilted app box
        transform = QTransform()
        transform.translate(250, 250)
        transform.rotate(self.absolute_angle)
        rect = QRect(-100, -100, 200, 200)
        poly = transform.mapToPolygon(rect)
        region = QRegion(poly)
        
        if self.hovered:
            # We also accept cursor hits floating over the distant text
            text_transform = QTransform()
            text_transform.translate(250, 250)
            text_transform.rotate(self.absolute_angle)
            text_transform.translate(0, -127)  # Pivot offset simulating HTML CSS bottom: 12px
            text_transform.rotate(-self.absolute_angle)
            
            text_rect = QRect(-250, -30, 500, 60)
            text_poly = text_transform.mapToPolygon(text_rect)
            region = region.united(QRegion(text_poly))
            
        self.setMask(region)
        self.update()

    def enterEvent(self, _event) -> None:
        self.hovered = True
        self.set_absolute_angle(self.absolute_angle)
        self.update()
        self.hover_changed.emit(self.name, True)

    def leaveEvent(self, _event) -> None:
        self.hovered = False
        self.set_absolute_angle(self.absolute_angle)
        self.update()
        self.hover_changed.emit(self.name, False)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.name)
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.translate(250, 250)
        painter.rotate(self.absolute_angle)

        # 1. Manual Blur Box Shadow
        painter.save()
        painter.setPen(Qt.NoPen)
        for i in range(1, 18):
            alpha = int((18 - i) * 1.8)
            if alpha < 0: alpha = 0
            painter.setBrush(QColor(0, 0, 0, alpha))
            rect = QRectF(-100 - i, -100 + 15 - i, 200 + i * 2, 200 + i * 2)
            painter.drawRoundedRect(rect, 8 + i, 8 + i)
        painter.restore()

        # 2. Base Dark fill backing transparent images
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#222222"))
        painter.drawRoundedRect(QRectF(-100, -100, 200, 200), 8, 8)

        # 3. Object-Fit Image
        path = QPainterPath()
        path.addRoundedRect(QRectF(-96, -96, 192, 192), 4, 4)
        painter.save()
        painter.setClipPath(path)
        pw, ph = self.pixmap.width(), self.pixmap.height()
        painter.drawPixmap(-pw // 2, -ph // 2, pw, ph, self.pixmap)
        painter.restore()

        # 4. Yellow border-box
        border_color = SL_YELLOW if self.hovered else QColor(0, 0, 0, 0)
        painter.setPen(QPen(border_color, 4))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(QRectF(-98, -98, 196, 196), 6, 6)

        # 5. Elevated Floating Tooltip 
        if self.hovered:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 100))
            painter.drawEllipse(QPointF(0, -98), 7, 7)

            painter.setPen(QPen(BG_DARK, 2))
            painter.setBrush(SL_YELLOW)
            painter.drawEllipse(QPointF(0, -100), 6, 6)

            painter.translate(0, -127)
            painter.rotate(-self.absolute_angle)

            font = QFont("Georgia")
            font.setPixelSize(24)
            font.setBold(True)
            painter.setFont(font)

            text_rect = QRectF(-250, -20, 500, 40)

            painter.setPen(QColor(0, 0, 0, 200))
            for dx, dy in[(0, 2), (1, 2), (-1, 2), (0, 3)]:
                painter.drawText(text_rect.translated(dx, dy), Qt.AlignCenter, self.name)
            painter.setPen(QColor(0, 0, 0, 255))
            painter.drawText(text_rect.translated(0, 1), Qt.AlignCenter, self.name)

            painter.setPen(SL_YELLOW)
            painter.drawText(text_rect, Qt.AlignCenter, self.name)


class HubWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SLAI Hub")
        self.resize(1600, 900)
        self.setMinimumSize(1200, 700)

        self.assets = Path(__file__).resolve().parent / "component" / "assets"
        self._load_descriptions()

        self.target_rotation = 0.0
        self.current_rotation = 0.0
        self.app_cards: list[AppCard] =[]
        self.child_window = None
        self.auth_service = AuthService()
        self.search_engine = SearchEngine(fields=["title", "description"])
        self.search_engine.index_documents(
            [
                {"title": "MusAI", "description": "AI-powered music assistant."},
                {"title": "BuildUp", "description": "Project execution and planning workspace."},
                {"title": "Documaster", "description": "Smart document analysis and extraction."},
                {"title": "SignalSentry", "description": "Monitoring and alert intelligence app."},
            ]
        )
        self.dropdown_menu_model = DropdownMenu(
            [
                DropdownOption("Home", "home"),
                DropdownOption("Products", "products"),
                DropdownOption("Pricing", "pricing"),
                DropdownOption("Contact", "contact"),
                DropdownOption("System Monitoring", "system_monitoring"),
            ],
            animation=AnimationConfig(duration_ms=220, preset="smooth"),
        )
        self.monitoring_dialog = None

        self._create_starfield()
        self._build_ui()

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate)
        self.anim_timer.start(16)

    def _load_descriptions(self) -> None:
        self.app_descriptions = {}
        templates_dir = self.assets.parent / "templates"
        json_path = templates_dir / "descriptions.json"
        
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.app_descriptions = json.load(f)
            except Exception as e:
                print(f"Failed to load descriptions.json: {e}")

    def _create_starfield(self) -> None:
        self.star_pattern = QPixmap(400, 400)
        self.star_pattern.fill(Qt.transparent)
        painter = QPainter(self.star_pattern)
        painter.setRenderHint(QPainter.Antialiasing)
        
        stars =[
            (40, 80, 2, 0.8), (120, 160, 1.5, 0.4), (200, 40, 2.5, 0.7),
            (280, 200, 1, 0.3), (360, 120, 2, 0.6), (80, 320, 1.5, 0.5),
            (160, 280, 2, 0.9), (240, 360, 1.5, 0.4), (320, 320, 2, 0.7)
        ]
        
        painter.setPen(Qt.NoPen)
        for x, y, r, a in stars:
            grad = QRadialGradient(x, y, r * 2)
            grad.setColorAt(0, QColor(255, 255, 255, int(a * 255)))
            grad.setColorAt(1, QColor(0, 0, 0, 0))
            painter.setBrush(grad)
            painter.drawEllipse(QPointF(x, y), r * 2, r * 2)
        painter.end()

    def _build_ui(self) -> None:
        self.bg_logo = QPixmap(str(self.assets / "home_active.png"))

        self.hamburger = HamburgerIcon(self)
        self.logo_label = QLabel("<span style='color:#eacb00'>SLAI</span> <span style='color:#ffffff'>Hub</span>", self)
        self.logo_label.setStyleSheet("font-family: Georgia; font-size: 32px; font-weight: 700; letter-spacing: 0.5px;")
        
        self.search = SearchIcon(self)
        self.login_btn = LoginButton(self)
        self.search_bar = QLineEdit(self)
        self.search_bar.hide()
        self.search_bar.setPlaceholderText("Search products...")
        self.search_bar.returnPressed.connect(self._run_search)
        self.search_bar.setStyleSheet(
            """
            QLineEdit {
                background: #0e1012;
                border: 2px solid #ffffff;
                border-radius: 10px;
                color: #ffffff;
                font-family: Georgia;
                font-size: 16px;
                padding: 6px 10px;
            }
            """
        )
        self.search_expand_anim = QPropertyAnimation(self.search_bar, b"geometry", self)
        self.search_expand_anim.setDuration(240)
        self.search_expand_anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.search_expand_anim.finished.connect(self._on_search_anim_finished)
        self.search_animating_close = False
        self.search_open = False

        self.dropdown_panel = QFrame(self)
        self.dropdown_panel.setFrameShape(QFrame.StyledPanel)
        self.dropdown_panel.hide()
        self.dropdown_panel.setStyleSheet(
            "QFrame { background: #13171b; border: 1px solid #3f454c; border-radius: 10px; }"
        )
        self.dropdown_layout = QVBoxLayout(self.dropdown_panel)
        self.dropdown_layout.setContentsMargins(12, 10, 12, 10)
        self.dropdown_layout.setSpacing(8)
        self.dropdown_effect = QGraphicsOpacityEffect(self.dropdown_panel)
        self.dropdown_panel.setGraphicsEffect(self.dropdown_effect)
        self.dropdown_effect.setOpacity(0.0)
        self.dropdown_anim = QPropertyAnimation(self.dropdown_panel, b"geometry", self)
        self.dropdown_anim.setDuration(300)
        self.dropdown_opacity_anim = QPropertyAnimation(self.dropdown_effect, b"opacity", self)
        self.dropdown_opacity_anim.setDuration(300)
        self.dropdown_anim.finished.connect(self._on_dropdown_anim_finished)
        self.dropdown_open = False
        self.dropdown_animating_close = False

        for option in self.dropdown_menu_model.options:
            btn = QPushButton(option.label, self.dropdown_panel)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                """
                QPushButton {
                    color: white;
                    background: transparent;
                    text-align: left;
                    font-family: Georgia;
                    padding: 6px 4px;
                    border: none;
                }
                QPushButton:hover { color: #eacb00; }
                """
            )
            btn.clicked.connect(lambda _checked=False, v=option.value: self._select_dropdown(v))
            self.dropdown_layout.addWidget(btn)

        self.hero = QLabel(self)
        self.hero.setText(
            "<table cellpadding='0' cellspacing='0'><tr>"
            "<td valign='top' style='color:#eacb00; font-family:Georgia; font-size:42px; font-weight:700; padding-right:18px;'>SLAI:</td>"
            "<td valign='top' style='color:#ffffff; font-family:Georgia; font-size:42px; font-weight:700; line-height:1.4;'>"
            "The future of<br>intelligent automation</td>"
            "</tr></table>"
        )
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 127))
        self.hero.setGraphicsEffect(shadow)

        # The Hover Description Label
        self.desc_label = QLabel(self)
        self.desc_label.setFixedWidth(550)  # Controls standard text wrapping
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("background: transparent;")
        self.desc_label.hide()

        # Timer to track the 1 second hover logic
        self.current_hovered_app = None
        self.hover_timer = QTimer(self)
        self.hover_timer.setSingleShot(True)
        self.hover_timer.timeout.connect(self._show_description)

        # Items
        items_data =[
            ("MusAI", self.assets / "musician_active.png"),
            ("BuildUp", self.assets / "BuildUp.png"),
            ("Documaster", self.assets / "Docmaster.png"),
            ("SignalSentry", self.assets / "SignalS.png"),
            ("ContentOps Autopublisher", self.assets / "Autopublisher.png"),
            ("Colony Clerk", self.assets / "colony.png"),
            ("Coming soon", self.assets / "coming-soon.png"),
            ("Coming soon", self.assets / "coming-soon.png"),
            ("Coming soon", self.assets / "coming-soon.png"),
            ("Coming soon", self.assets / "coming-soon.png"),
            ("Coming soon", self.assets / "coming-soon.png"),
            ("Coming soon", self.assets / "coming-soon.png"),
        ]

        for idx, (name, img) in enumerate(items_data):
            angle = (idx - 2) * ITEM_SPACING
            card = AppCard(self, name, img, angle)
            card.hover_changed.connect(self._on_app_hover_changed)
            card.clicked.connect(self._on_app_clicked)
            self.app_cards.append(card)

        # Force UI overlays to front
        self.hamburger.raise_()
        self.dropdown_panel.raise_()
        self.logo_label.raise_()
        self.search.raise_()
        self.search_bar.raise_()
        self.login_btn.raise_()
        self.hero.raise_()
        self.desc_label.raise_()

        self.hamburger.mousePressEvent = self._toggle_dropdown_event
        self.search.mousePressEvent = self._toggle_search_event
        self.login_btn.clicked.connect(self._open_login_dialog)
        QApplication.instance().installEventFilter(self)

    def _toggle_dropdown_event(self, event):
        if event.button() != Qt.LeftButton:
            return
        is_open = self.dropdown_menu_model.toggle()
        if is_open:
            self._open_dropdown_panel()
        else:
            self._close_dropdown_panel()

    def _select_dropdown(self, value: str):
        selected = self.dropdown_menu_model.select(value)
        self._close_dropdown_panel()
        if value == "system_monitoring":
            self._open_monitoring_dialog()
            return
        QMessageBox.information(self, "Menu", f"Selected: {selected.title()}")

    def _open_monitoring_dialog(self):
        self.monitoring_dialog = SystemMonitoringDialog(self)
        self.monitoring_dialog.exec_()

    def _open_login_dialog(self):
        dialog = LoginDialog(self.auth_service, self)
        dialog.exec_()

    def _toggle_search_event(self, event):
        if event.button() != Qt.LeftButton:
            return
        icon_x = self.search.x()
        icon_y = self.search.y() - 2
        if not self.search_open:
            self.search_bar.show()
            self.search.setVisible(False)
            self.search_expand_anim.stop()
            self.search_expand_anim.setStartValue(QRect(icon_x, icon_y, 26, 30))
            self.search_expand_anim.setEndValue(QRect(icon_x - 260, icon_y, 286, 36))
            self.search_expand_anim.start()
            self.search_open = True
            self.search_bar.setFocus()
        else:
            self._close_search_bar()

    def _close_search_bar(self):
        if not self.search_open:
            return
        icon_x = self.search.x()
        icon_y = self.search.y() - 2
        self.search_animating_close = True
        self.search_expand_anim.stop()
        self.search_expand_anim.setStartValue(self.search_bar.geometry())
        self.search_expand_anim.setEndValue(QRect(icon_x, icon_y, 26, 30))
        self.search_expand_anim.start()

    def _on_search_anim_finished(self):
        if not self.search_animating_close:
            return
        self.search_animating_close = False
        self.search_bar.hide()
        self.search.setVisible(True)
        self.search_open = False

    def _run_search(self):
        query = self.search_bar.text().strip()
        if not query:
            return
        results = self.search_engine.search(query, limit=3)
        if not results:
            QMessageBox.information(self, "Search", "No results found.")
            return
        msg = "\n".join(
            [f"- {result.item.get('title', 'Untitled')} ({result.score:.2f})" for result in results]
        )
        QMessageBox.information(self, "Search results", msg)

    def _on_app_hover_changed(self, app_name: str, is_hovered: bool) -> None:
        if is_hovered:
            self.current_hovered_app = app_name
            self.hover_timer.start(1000)  # Start the 1 second timer
        else:
            if self.current_hovered_app == app_name:
                self.hover_timer.stop()
                self.current_hovered_app = None
                self.desc_label.hide()

    def _show_description(self) -> None:
        if not self.current_hovered_app:
            return
            
        desc = self.app_descriptions.get(
            self.current_hovered_app, 
            "Description not available."
        )

        html_content = f"""
        <div style='font-family: Georgia;'>
            <div style='color: #eacb00; font-size: 20px; font-weight: bold; margin-bottom: 12px;'>{self.current_hovered_app}:</div>
            <div style='color: #ffffff; font-size: 17px; line-height: 1.45;'>{desc}</div>
        </div>
        """
        self.desc_label.setText(html_content)
        self.desc_label.adjustSize()
        self.desc_label.show()

    def _on_app_clicked(self, app_name: str) -> None:
        if app_name not in {"SignalSentry", "ContentOps Autopublisher"}:
            return

        try:
            if app_name == "SignalSentry":
                from component.signal_sentry import SignalSentryWindow
                self.child_window = SignalSentryWindow()
            else:
                from component.autopublisher import AutopublisherWindow
                self.child_window = AutopublisherWindow()
            self.child_window.show()
            self.close()
        except Exception as exc:
            print(f"Failed to launch {app_name}: {exc}")

    def resizeEvent(self, _event) -> None:
        self._position_top_bar()
        self._position_hero()
        self._position_cards()
        if self.dropdown_open and not self.dropdown_anim.state():
            self.dropdown_panel.setGeometry(self._dropdown_open_rect())

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            click_pos = event.globalPos()
            if self.dropdown_open:
                dropdown_rect = QRect(self.dropdown_panel.mapToGlobal(QPoint(0, 0)), self.dropdown_panel.size())
                hamburger_rect = QRect(self.hamburger.mapToGlobal(QPoint(0, 0)), self.hamburger.size())
                if not dropdown_rect.contains(click_pos) and not hamburger_rect.contains(click_pos):
                    self.dropdown_menu_model.close()
                    self._close_dropdown_panel()

            if self.search_open:
                search_rect = QRect(self.search_bar.mapToGlobal(QPoint(0, 0)), self.search_bar.size())
                icon_rect = QRect(self.search.mapToGlobal(QPoint(0, 0)), self.search.size())
                if not search_rect.contains(click_pos) and not icon_rect.contains(click_pos):
                    self._close_search_bar()
        return super().eventFilter(obj, event)

    def wheelEvent(self, event) -> None:
        if event.angleDelta().y() > 0:
            self.target_rotation += 10
        else:
            self.target_rotation -= 10
        self.target_rotation = max(-MAX_ROTATION, min(MAX_ROTATION, self.target_rotation))

    def _position_top_bar(self) -> None:
        center_y = 49
        self.hamburger.move(40, center_y - 12)
        if not self.dropdown_open:
            closed_rect = self._dropdown_closed_rect()
            self.dropdown_panel.setGeometry(closed_rect)
        self.logo_label.adjustSize()
        self.logo_label.move(101, center_y - self.logo_label.height() // 2)
        self.login_btn.move(self.width() - 40 - 110, center_y - 19)
        self.search.move(self.login_btn.x() - 25 - 26, center_y - 13)
        if self.search_open:
            self.search_bar.setGeometry(self.search.x() - 260, self.search.y() - 2, 286, 36)

    def _dropdown_open_rect(self):
        x = 20
        y = 10
        width = 260
        height = min(self.height() - 20, int(self.height() * 0.95))
        return QRect(x, y, width, height)

    def _dropdown_closed_rect(self):
        open_rect = self._dropdown_open_rect()
        return QRect(open_rect.x(), open_rect.y(), open_rect.width(), 0)

    def _open_dropdown_panel(self):
        self.dropdown_animating_close = False
        self.dropdown_open = True
        self.dropdown_panel.show()
        self.dropdown_panel.raise_()
        self.dropdown_anim.stop()
        self.dropdown_opacity_anim.stop()
        self.dropdown_anim.setStartValue(self._dropdown_closed_rect())
        self.dropdown_anim.setEndValue(self._dropdown_open_rect())
        self.dropdown_anim.setEasingCurve(QEasingCurve.OutCubic)
        self.dropdown_opacity_anim.setStartValue(0.0)
        self.dropdown_opacity_anim.setEndValue(1.0)
        self.dropdown_opacity_anim.setEasingCurve(QEasingCurve.OutCubic)
        self.dropdown_anim.start()
        self.dropdown_opacity_anim.start()

    def _close_dropdown_panel(self):
        if not self.dropdown_open:
            return
        self.dropdown_animating_close = True
        self.dropdown_open = False
        self.dropdown_anim.stop()
        self.dropdown_opacity_anim.stop()
        self.dropdown_anim.setStartValue(self.dropdown_panel.geometry())
        self.dropdown_anim.setEndValue(self._dropdown_closed_rect())
        self.dropdown_anim.setEasingCurve(QEasingCurve.InCubic)
        self.dropdown_opacity_anim.setStartValue(self.dropdown_effect.opacity())
        self.dropdown_opacity_anim.setEndValue(0.0)
        self.dropdown_opacity_anim.setEasingCurve(QEasingCurve.InCubic)
        self.dropdown_anim.start()
        self.dropdown_opacity_anim.start()

    def _on_dropdown_anim_finished(self):
        if self.dropdown_animating_close:
            self.dropdown_animating_close = False
            self.dropdown_panel.hide()

    def _position_hero(self) -> None:
        self.hero.adjustSize()
        self.hero.move(int(self.width() * 0.15), int(self.height() * 0.25))
        
        # Position the description block right underneath the hero title
        self.desc_label.move(self.hero.x(), self.hero.y() + self.hero.height() + 25)

    def _position_cards(self) -> None:
        pivot_x = self.width() * 0.50
        pivot_y = self.height() * 1.10
        app_translation_radius = RADIUS + APP_DISTANCE + 100

        for card in self.app_cards:
            ang = math.radians(card.base_angle + self.current_rotation)
            x = pivot_x + app_translation_radius * math.sin(ang) - 250
            y = pivot_y - app_translation_radius * math.cos(ang) - 250
            card.move(int(x), int(y))
            card.set_absolute_angle(card.base_angle + self.current_rotation)

    def _animate(self) -> None:
        self.current_rotation += (self.target_rotation - self.current_rotation) * 0.08
        self._position_cards()
        self.update()

    def _draw_mountains(self, painter: QPainter) -> None:
        y_base = self.height()
        h = self.height() * 0.45
        w = self.width()

        def make_poly(points):
            poly = QPolygonF()
            for vx, vy in points:
                poly.append(QPointF(w * (vx / 100.0), y_base - h + h * (vy / 50.0)))
            return poly

        back_pts =[(0, 50), (8, 35), (18, 45), (32, 15), (45, 35), (65, 5), (85, 30), (100, 10), (100, 50)]
        front_pts =[(0, 50), (12, 40), (25, 48), (40, 25), (55, 42), (75, 15), (90, 38), (100, 20), (100, 50)]

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#141619"))
        painter.setOpacity(0.9)
        painter.drawPolygon(make_poly(back_pts))

        painter.setBrush(QColor("#1a1d21"))
        painter.setOpacity(0.6)
        painter.drawPolygon(make_poly(front_pts))
        painter.setOpacity(1.0)

    def _draw_background_logo(self, painter: QPainter) -> None:
        if self.bg_logo.isNull():
            return
        logo_w = int(min(self.width() * 0.45, 600))
        logo_h = int(min(self.width() * 0.45, 600))
        x = self.width() - logo_w - 80
        y = 40
        painter.setOpacity(0.95)
        painter.drawPixmap(x, y, logo_w, logo_h, self.bg_logo)
        painter.setOpacity(1.0)

    def _draw_wheel(self, painter: QPainter) -> None:
        cx = self.width() / 2
        cy = self.height() * 1.10

        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.current_rotation)

        painter.setPen(QPen(TEXT_WHITE, 4))
        painter.setBrush(BG_DARK)
        painter.drawEllipse(QPointF(0, 0), RADIUS, RADIUS)

        painter.setPen(QPen(TEXT_WHITE, 2))
        for angle in range(0, 360, 10):
            rad = math.radians(angle)
            x2 = RADIUS * math.sin(rad)
            y2 = -RADIUS * math.cos(rad)
            painter.drawLine(QPointF(0, 0), QPointF(x2, y2))

        painter.restore()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.fillRect(self.rect(), BG_DARK)
        painter.setOpacity(0.6)
        painter.fillRect(self.rect(), QBrush(self.star_pattern))
        painter.setOpacity(1.0)
        
        self._draw_background_logo(painter)
        self._draw_mountains(painter)
        self._draw_wheel(painter)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = HubWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
