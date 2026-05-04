from __future__ import annotations

import io
import json

from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QEasingCurve, QPoint, QParallelAnimationGroup, QPropertyAnimation, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.functions.auth import AuthService, AuthToken
from src.functions.email import EmailService
from src.functions.ratelimiter import RateLimiter
from src.functions.storage import Storage
from src.functions.utils.functions_error import AccountLockedError, InvalidCredentialsError, UserAlreadyExistsError
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Authentication Dialog")
printer = PrettyPrinter


class AuthDialog(QDialog):
    """Modular authentication flow with login, sign-up, and verification panels."""

    authentication_succeeded = pyqtSignal(str, object)
    authentication_cancelled = pyqtSignal()
    verification_required = pyqtSignal(str)
    status_changed = pyqtSignal(str, str)

    PURPOSE_SIGNUP = "signup_verification"
    PURPOSE_LOGIN = "login_verification"
    PURPOSE_REAUTH = "reauth_verification"

    def __init__(
        self,
        auth_service: AuthService,
        rate_limiter: RateLimiter,
        email_service: Optional[EmailService],
        storage: Optional[Storage],
        parent: Optional[QWidget] = None,
        username_hint: str = "",
        mode: str = "default",
    ) -> None:
        super().__init__(parent)
        self.auth_service = auth_service
        self.rate_limiter = rate_limiter
        self.email_service = email_service
        self.storage = storage

        self.pending_username = ""
        self.pending_purpose = ""
        self.pending_token: Optional[AuthToken] = None
        self.mode = mode

        self.setWindowTitle("Authentication")
        self.setModal(True)
        self.setFixedSize(860, 760)
        self.setStyleSheet("QDialog { background-color: #0b1018; border-radius: 28px; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.card = QFrame(self)
        self.card.setStyleSheet("QFrame { border-radius: 28px; }")
        self.card.setObjectName("auth_card")
        root.addWidget(self.card)

        self.viewport = QWidget(self.card)
        self.viewport.setGeometry(0, 0, self.width(), self.height())

        self.login_panel = self._build_login_panel()
        self.signup_panel = self._build_signup_panel()
        self.verify_panel = self._build_verify_panel()

        self._panel_map = {
            "login": self.login_panel,
            "signup": self.signup_panel,
            "verify": self.verify_panel,
        }
        self.current_panel = "login"
        self._reflow_panels()

        self.login_username.setText(username_hint)
        if mode == "reauth":
            self.login_panel.hide()
            self.signup_panel.hide()
            self.current_panel = "verify"
            self.verify_panel.move(0, 0)
            self._prepare_reauth(username_hint)

        logger.info(f"Authentication Dialog successfully initialized")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.viewport.setGeometry(0, 0, self.card.width(), self.card.height())
        self._reflow_panels()

    def _reflow_panels(self) -> None:
        w, h = self.card.width(), self.card.height()
        for name, panel in self._panel_map.items():
            panel.setFixedSize(w, h)
            if name == self.current_panel:
                panel.move(0, 0)
            else:
                panel.move(w + 10, 0)

    def _build_logo(self, dark: bool) -> QLabel:
        label = QLabel(self)
        label.setAlignment(Qt.AlignCenter)
        logo_path = Path(__file__).resolve().parents[1] / "assets" / "logo.png"
        pixmap = QPixmap(str(logo_path))
        if not pixmap.isNull():
            label.setPixmap(
                pixmap.scaled(
                    210,
                    210,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            # Fallback for environments where the asset cannot be loaded.
            label.setText("SLAI")
            color = "#d8bd1d" if dark else "#0b1018"
            label.setStyleSheet(f"font-family: Georgia; font-size: 48px; font-weight: 700; color: {color};")
        return label

    def _message_label(self, dark: bool) -> QLabel:
        lbl = QLabel("", self)
        lbl.setWordWrap(True)
        lbl.setFixedHeight(46)
        base = "#ff7f7f" if dark else "#5e1d1d"
        lbl.setStyleSheet(
            f"font-family: Georgia; font-size: 14px; color: {base}; padding: 4px 2px;"
        )
        return lbl

    def _line_edit_style(self, dark: bool) -> str:
        if dark:
            return (
                "QLineEdit { background:#090d13; border:3px solid #43474f; border-radius:16px;"
                " color:#f4f4f4; padding:10px 14px; font-family:Georgia; font-size:18px; }"
                "QLineEdit:focus { border-color:#d9be33; }"
            )
        return (
            "QLineEdit { background:#bfa503; border:3px solid #0b1018; border-radius:16px;"
            " color:#0b1018; padding:10px 14px; font-family:Georgia; font-size:18px; }"
            "QLineEdit:focus { border-color:#20252c; }"
        )

    def _button_style(self, primary_dark: bool) -> str:
        if primary_dark:
            return (
                "QPushButton { background:#0b1018; color:#d9be33; border:2px solid #0b1018;"
                " border-radius:16px; padding:12px; font-family:Georgia; font-size:22px; font-weight:700; }"
                "QPushButton:hover { background:#121a26; }"
                "QPushButton:pressed { background:#070b11; border:2px inset #05080d; padding-top:14px; padding-left:14px; }"
                "QPushButton:disabled { background:#37404d; color:#9da8b8; }"
            )
        return (
            "QPushButton { background:#dcc83d; color:#0b1018; border:2px solid #dcc83d;"
            " border-radius:16px; padding:12px; font-family:Georgia; font-size:22px; font-weight:700; }"
            "QPushButton:hover { background:#ead85a; }"
            "QPushButton:pressed { background:#c7af2b; border:2px inset #967f08; padding-top:14px; padding-left:14px; }"
            "QPushButton:disabled { background:#736726; color:#c6c6c6; }"
        )

    def _build_login_panel(self) -> QWidget:
        panel = QFrame(self.viewport)
        panel.setStyleSheet("QFrame { background:#090f1a; border-radius:28px; }")
        wrap = QVBoxLayout(panel)
        wrap.setContentsMargins(48, 24, 48, 28)
        wrap.setSpacing(10)

        close = QPushButton("×", panel)
        close.setFixedSize(42, 42)
        close.setStyleSheet("QPushButton { background:transparent; color:#e5ca35; font-size:42px; border:none; } QPushButton:pressed{padding-top:3px;}")
        close.clicked.connect(self._cancel)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close)
        wrap.addLayout(row)

        wrap.addWidget(self._build_logo(dark=True))

        title = QLabel("Log in to SLAI Hub", panel)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-family: Georgia; font-size: 48px; font-weight: 700; color:#e5ca35;")
        wrap.addWidget(title)

        self.login_message = self._message_label(dark=True)
        wrap.addWidget(self.login_message)

        self.login_username = QLineEdit(panel)
        self.login_username.setPlaceholderText("Username")
        self.login_username.setStyleSheet(self._line_edit_style(dark=True))
        self.login_password = QLineEdit(panel)
        self.login_password.setPlaceholderText("Password")
        self.login_password.setEchoMode(QLineEdit.Password)
        self.login_password.setStyleSheet(self._line_edit_style(dark=True))
        wrap.addWidget(self.login_username)
        wrap.addWidget(self.login_password)

        login_button = QPushButton("Log in", panel)
        login_button.setStyleSheet(self._button_style(primary_dark=False))
        login_button.clicked.connect(self._handle_login)
        wrap.addWidget(login_button)

        forgot = QPushButton("Forgot Password?", panel)
        forgot.setFlat(True)
        forgot.setStyleSheet("QPushButton{color:#f0f0f0; background:transparent; border:none; font-family:Georgia; font-size:16px;} QPushButton:pressed{padding-top:2px;}")
        forgot.clicked.connect(self._handle_forgot_password)
        wrap.addWidget(forgot, alignment=Qt.AlignHCenter)

        signup = QPushButton("Don’t have an account? Sign up", panel)
        signup.setFlat(True)
        signup.setStyleSheet("QPushButton{color:#f0f0f0; background:transparent; border:none; font-family:Georgia; font-size:18px;} QPushButton:hover{color:#e5ca35;} QPushButton:pressed{padding-top:2px;}")
        signup.clicked.connect(lambda: self._animate_to("signup"))
        wrap.addWidget(signup, alignment=Qt.AlignHCenter)
        wrap.addStretch(1)
        return panel

    def _build_signup_panel(self) -> QWidget:
        panel = QFrame(self.viewport)
        panel.setStyleSheet("QFrame { background:#dcc83d; border-radius:28px; }")
        wrap = QVBoxLayout(panel)
        wrap.setContentsMargins(48, 24, 48, 28)
        wrap.setSpacing(10)

        close = QPushButton("×", panel)
        close.setFixedSize(42, 42)
        close.setStyleSheet("QPushButton { background:transparent; color:#0b1018; font-size:42px; border:none; } QPushButton:pressed{padding-top:3px;}")
        close.clicked.connect(self._cancel)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close)
        wrap.addLayout(row)

        title = QLabel("Create An Account", panel)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-family: Georgia; font-size: 54px; font-weight: 700; color:#0b1018;")
        subtitle = QLabel("Welcome! Register for an account", panel)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-family: Georgia; font-size: 24px; color:#0b1018;")
        wrap.addWidget(title)
        wrap.addWidget(subtitle)

        self.signup_message = self._message_label(dark=False)
        wrap.addWidget(self.signup_message)

        self.signup_username = QLineEdit(panel)
        self.signup_username.setPlaceholderText("Username")
        self.signup_username.setStyleSheet(self._line_edit_style(dark=False))
        self.signup_email = QLineEdit(panel)
        self.signup_email.setPlaceholderText("Email")
        self.signup_email.setStyleSheet(self._line_edit_style(dark=False))
        self.signup_password = QLineEdit(panel)
        self.signup_password.setPlaceholderText("Password")
        self.signup_password.setEchoMode(QLineEdit.Password)
        self.signup_password.setStyleSheet(self._line_edit_style(dark=False))
        self.signup_confirm = QLineEdit(panel)
        self.signup_confirm.setPlaceholderText("Confirm Password")
        self.signup_confirm.setEchoMode(QLineEdit.Password)
        self.signup_confirm.setStyleSheet(self._line_edit_style(dark=False))
        wrap.addWidget(self.signup_username)
        wrap.addWidget(self.signup_email)
        wrap.addWidget(self.signup_password)
        wrap.addWidget(self.signup_confirm)

        create_btn = QPushButton("Sign Up", panel)
        create_btn.setStyleSheet(self._button_style(primary_dark=True))
        create_btn.clicked.connect(self._handle_signup)
        wrap.addWidget(create_btn)

        login_link = QPushButton("Already have an account? Log in", panel)
        login_link.setFlat(True)
        login_link.setStyleSheet("QPushButton{color:#344050; background:transparent; border:none; font-family:Georgia; font-size:18px;} QPushButton:hover{color:#0b1018;}")
        login_link.clicked.connect(lambda: self._animate_to("login", reverse=True))
        wrap.addWidget(login_link, alignment=Qt.AlignHCenter)
        wrap.addStretch(1)
        return panel

    def _build_verify_panel(self) -> QWidget:
        panel = QFrame(self.viewport)
        panel.setStyleSheet("QFrame { background:#dcc83d; border-radius:28px; }")
        wrap = QVBoxLayout(panel)
        wrap.setContentsMargins(48, 24, 48, 28)
        wrap.setSpacing(10)

        close = QPushButton("×", panel)
        close.setFixedSize(42, 42)
        close.setStyleSheet("QPushButton { background:transparent; color:#0b1018; font-size:42px; border:none; } QPushButton:pressed{padding-top:3px;}")
        close.clicked.connect(self._cancel)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close)
        wrap.addLayout(row)

        title = QLabel("Verify Account", panel)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-family: Georgia; font-size: 54px; font-weight: 700; color:#0b1018;")
        subtitle = QLabel("Welcome! Please verify this is your account", panel)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-family: Georgia; font-size: 24px; color:#0b1018;")
        wrap.addWidget(title)
        wrap.addWidget(subtitle)

        self.verify_message = self._message_label(dark=False)
        wrap.addWidget(self.verify_message)

        self.verify_code = QLineEdit(panel)
        self.verify_code.setPlaceholderText("Your 24-character passcode")
        self.verify_code.setStyleSheet(self._line_edit_style(dark=False))
        wrap.addWidget(self.verify_code)

        verify_btn = QPushButton("Submit", panel)
        verify_btn.setStyleSheet(self._button_style(primary_dark=True))
        verify_btn.clicked.connect(self._handle_verify_submit)
        wrap.addWidget(verify_btn)

        resend_btn = QPushButton("Resend verification code", panel)
        resend_btn.setFlat(True)
        resend_btn.setStyleSheet("QPushButton{color:#344050; background:transparent; border:none; font-family:Georgia; font-size:18px;} QPushButton:hover{color:#0b1018;}")
        resend_btn.clicked.connect(self._handle_resend)
        wrap.addWidget(resend_btn, alignment=Qt.AlignHCenter)

        back_btn = QPushButton("Back to log in", panel)
        back_btn.setFlat(True)
        back_btn.setStyleSheet("QPushButton{color:#344050; background:transparent; border:none; font-family:Georgia; font-size:18px;} QPushButton:hover{color:#0b1018;}")
        back_btn.clicked.connect(lambda: self._animate_to("login", reverse=True))
        wrap.addWidget(back_btn, alignment=Qt.AlignHCenter)
        wrap.addStretch(1)
        return panel

    def _animate_to(self, target: str, reverse: bool = False) -> None:
        if target == self.current_panel:
            return

        current_widget = self._panel_map[self.current_panel]
        target_widget = self._panel_map[target]
        w = self.card.width()
        in_start = QPoint(-w, 0) if reverse else QPoint(w, 0)
        out_end = QPoint(w, 0) if reverse else QPoint(-w // 2, 0)

        target_widget.show()
        target_widget.raise_()
        target_widget.move(in_start)

        current_opacity = QGraphicsOpacityEffect(current_widget)
        current_widget.setGraphicsEffect(current_opacity)
        current_opacity.setOpacity(1.0)

        anim_group = QParallelAnimationGroup(self)
        move_out = QPropertyAnimation(current_widget, b"pos")
        move_out.setDuration(360)
        move_out.setStartValue(current_widget.pos())
        move_out.setEndValue(out_end)
        move_out.setEasingCurve(QEasingCurve.InOutCubic)

        fade_out = QPropertyAnimation(current_opacity, b"opacity")
        fade_out.setDuration(300)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.2)

        move_in = QPropertyAnimation(target_widget, b"pos")
        move_in.setDuration(420)
        move_in.setStartValue(in_start)
        move_in.setEndValue(QPoint(0, 0))
        move_in.setEasingCurve(QEasingCurve.OutBack)

        anim_group.addAnimation(move_out)
        anim_group.addAnimation(fade_out)
        anim_group.addAnimation(move_in)

        def _finalize() -> None:
            current_widget.hide()
            current_widget.move(w + 10, 0)
            current_widget.setGraphicsEffect(None)
            self.current_panel = target

        anim_group.finished.connect(_finalize)
        anim_group.start()

    def _set_message(self, panel: str, message: str, level: str = "error") -> None:
        mapping = {
            "login": self.login_message,
            "signup": self.signup_message,
            "verify": self.verify_message,
        }
        label = mapping[panel]
        color = {
            "error": "#a22020" if panel != "login" else "#ff8f8f",
            "info": "#1f3f7a" if panel != "login" else "#8ab6ff",
            "success": "#156e2b" if panel != "login" else "#95f0a8",
        }.get(level, "#ff8f8f")
        label.setStyleSheet(f"font-family: Georgia; font-size: 14px; color: {color}; padding:4px 2px;")
        label.setText(message)
        self.status_changed.emit(message, level)

    def _clear_messages(self) -> None:
        self.login_message.clear()
        self.signup_message.clear()
        self.verify_message.clear()

    def _prepare_reauth(self, username: str) -> None:
        self.pending_username = username.strip()
        self.pending_purpose = self.PURPOSE_REAUTH
        self.verify_code.clear()
        self._set_message("verify", "Sending a verification code for account re-activation...", "info")
        if not self.pending_username:
            self._set_message("verify", "Unable to start re-verification: missing username.")
            return
        try:
            passcode, delivered, failure_reason = self._issue_and_send_code(
                self.pending_username,
                self.pending_purpose,
            )
            self.verification_required.emit(self.pending_username)
            if delivered:
                self._set_message("verify", "Check your email and enter the 24-character passcode.", "info")
            else:
                self._set_message(
                    "verify",
                    (
                        "Email delivery is unavailable. Use this one-time passcode now: "
                        f"{passcode} (valid for 30 minutes, single-use)."
                        + (f" Reason: {failure_reason}" if failure_reason else "")
                    ),
                    "info",
                )
        except Exception as exc:
            self._set_message("verify", str(exc))

    def _cancel(self) -> None:
        self.authentication_cancelled.emit()
        self.reject()

    def _validate_email(self, email: str) -> bool:
        return "@" in email and "." in email.split("@")[-1]

    def _handle_signup(self) -> None:
        self._clear_messages()
        username = self.signup_username.text().strip()
        email = self.signup_email.text().strip()
        password = self.signup_password.text()
        confirm = self.signup_confirm.text()

        if not username or not email or not password or not confirm:
            self._set_message("signup", "Please complete all sign-up fields.")
            return
        if not self._validate_email(email):
            self._set_message("signup", "Please provide a valid email address.")
            return
        if password != confirm:
            self._set_message("signup", "Password and confirmation do not match.")
            return

        if not self.rate_limiter.allow(f"signup:{username.lower()}"):
            self._set_message("signup", "Too many sign-up attempts. Please wait and try again.")
            return

        try:
            self.auth_service.sign_up(username=username, password=password, email=email)
            self.pending_username = username
            self.pending_purpose = self.PURPOSE_SIGNUP
            passcode, delivered, failure_reason = self._issue_and_send_code(
                username,
                self.pending_purpose,
            )
            self.verification_required.emit(username)
            if delivered:
                self._set_message("verify", "Account created. Enter the code sent to your email.", "info")
            else:
                self._set_message(
                    "verify",
                    (
                        "Account created, but email is unavailable. Use this one-time passcode now: "
                        f"{passcode} (valid for 30 minutes, single-use)."
                        + (f" Reason: {failure_reason}" if failure_reason else "")
                    ),
                    "info",
                )
            self._animate_to("verify")
        except UserAlreadyExistsError:
            self._set_message("signup", "Username already exists. Please choose another.")
        except Exception as exc:
            self._set_message("signup", str(exc))

    def _handle_login(self) -> None:
        self._clear_messages()
        username = self.login_username.text().strip()
        password = self.login_password.text()
        if not username or not password:
            self._set_message("login", "Please enter username and password.")
            return

        if not self.rate_limiter.allow(f"login:{username.lower()}"):
            self._set_message("login", "Too many login attempts. Please wait and try again.")
            return

        try:
            self.auth_service.verify_credentials(username, password)
            self.pending_username = username
            self.pending_purpose = self.PURPOSE_LOGIN
            passcode, delivered, failure_reason = self._issue_and_send_code(
                username,
                self.pending_purpose,
            )
            self.verification_required.emit(username)
            if delivered:
                self._set_message("verify", "Password accepted. Enter the emailed verification passcode.", "info")
            else:
                self._set_message(
                    "verify",
                    (
                        "Password accepted, but email is unavailable. Use this one-time passcode now: "
                        f"{passcode} (valid for 30 minutes, single-use)."
                        + (f" Reason: {failure_reason}" if failure_reason else "")
                    ),
                    "info",
                )
            self._animate_to("verify")
        except AccountLockedError as exc:
            self._set_message("login", str(exc))
        except InvalidCredentialsError:
            self._set_message("login", "Invalid username or password.")
        except Exception as exc:
            self._set_message("login", str(exc))

    def _handle_verify_submit(self) -> None:
        username = self.pending_username.strip()
        code = self.verify_code.text().strip()
        if not username:
            self._set_message("verify", "No pending account verification context was found.")
            return
        if len(code) != 24:
            self._set_message("verify", "Verification passcode must be exactly 24 characters.")
            return
        if not self.rate_limiter.allow(f"verify:{username.lower()}"):
            self._set_message("verify", "Too many verification attempts. Please wait and retry.")
            return

        try:
            result = self.auth_service.verify_passcode(username=username, passcode=code, purpose=self.pending_purpose)
            if not result:
                self._set_message("verify", "Invalid or expired verification passcode.")
                return

            if self.pending_purpose in {self.PURPOSE_LOGIN, self.PURPOSE_REAUTH}:
                token = self.auth_service.complete_login(username)
                self.pending_token = token
                self.auth_service.record_activity(username)
                self._persist_auth_snapshot(username, event="auth_success")
                self.authentication_succeeded.emit(username, token)
                self.accept()
                return

            # signup verification success
            self._set_message("verify", "Account verified. You can now log in.", "success")
            self.login_username.setText(username)
            self.login_password.clear()
            self.pending_purpose = ""
            self.verify_code.clear()
            self._animate_to("login", reverse=True)

        except Exception as exc:
            self._set_message("verify", str(exc))

    def _handle_resend(self) -> None:
        username = self.pending_username.strip()
        if not username:
            self._set_message("verify", "No pending verification challenge to resend.")
            return
        if not self.rate_limiter.allow(f"verify:{username.lower()}"):
            self._set_message("verify", "Rate limit reached. Please wait before resending.")
            return
        try:
            purpose = self.pending_purpose or self.PURPOSE_LOGIN
            passcode, delivered, failure_reason = self._issue_and_send_code(username, purpose)
            if delivered:
                self._set_message("verify", "A new verification code was emailed.", "info")
            else:
                self._set_message(
                    "verify",
                    (
                        "Email delivery is unavailable. Use this new one-time passcode: "
                        f"{passcode} (valid for 30 minutes, single-use)."
                        + (f" Reason: {failure_reason}" if failure_reason else "")
                    ),
                    "info",
                )
        except Exception as exc:
            self._set_message("verify", str(exc))

    def _handle_forgot_password(self) -> None:
        username = self.login_username.text().strip()
        if not username:
            self._set_message("login", "Enter your username first so we can send recovery guidance.")
            return

        email = self.auth_service.get_user_email(username)
        if not email:
            self._set_message("login", "No email is configured for this user account.")
            return
        if self.email_service is None:
            self._set_message("login", "Email service is unavailable. Contact support.")
            return

        body_html = (
            "<h3>SLAI password recovery request</h3>"
            f"<p>We received a password recovery request for <b>{username}</b>.</p>"
            "<p>If this was not you, you can safely ignore this email.</p>"
        )
        body_text = f"Password recovery request for {username}. If this was not you, ignore this email."

        try:
            self.email_service.send(
                to=email,
                subject="SLAI password recovery request",
                body_html=body_html,
                body_text=body_text,
            )
            self._set_message("login", f"Recovery guidance was emailed to {email}.", "info")
        except Exception as exc:
            self._set_message("login", f"Email delivery failed: {exc}")

    def _issue_and_send_code(self, username: str, purpose: str) -> tuple[str, bool, str]:
        email = self.auth_service.get_user_email(username)
        if not email:
            raise ValueError("A valid account email is required for verification.")

        passcode = self.auth_service.create_verification_challenge(
            username=username,
            purpose=purpose,
            expires_in_minutes=30,
            invalidate_existing=True,
        )
        self._persist_auth_snapshot(username, event=f"challenge:{purpose}")
        if self.email_service is None:
            return passcode, False, "Email service is not configured."

        body_html = (
            "<h3>SLAI Hub verification code</h3>"
            f"<p>A verification request was received for <b>{username}</b>.</p>"
            f"<p><b>Code:</b> {passcode}</p>"
            "<p>This one-time code is valid for 30 minutes and can only be used once.</p>"
            "<p>If you did not request this, please secure your account immediately.</p>"
        )
        body_text = (
            f"SLAI verification code for {username}: {passcode}. "
            "Valid for 30 minutes. Single-use only."
        )
        try:
            self.email_service.send(
                to=email,
                subject="SLAI Hub verification code",
                body_html=body_html,
                body_text=body_text,
            )
            return passcode, True, ""
        except Exception as exc:
            return passcode, False, str(exc)

    def _persist_auth_snapshot(self, username: str, event: str) -> None:
        if self.storage is None:
            return
        payload = {
            "username": username,
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
        }
        safe_username = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in username)
        try:
            self.storage.upload(
                file_obj=io.BytesIO(json.dumps(payload, indent=2).encode("utf-8")),
                filename="auth_state.json",
                subpath=f"auth_state/{safe_username}",
                metadata={"username": username, "event": event},
            )
        except Exception:
            # Snapshot persistence is best-effort and should not block auth.
            return