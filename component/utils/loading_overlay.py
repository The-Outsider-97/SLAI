"""Reusable full-screen loading overlay for SLAI desktop windows."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from src.functions.loader import LoaderState


class LoadingOverlay(QWidget):
    """Simple translucent overlay bound to Loader state updates."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("LoadingOverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            QWidget#LoadingOverlay {
                background-color: rgba(6, 8, 11, 210);
            }
            QLabel {
                color: #ffffff;
                font-family: Georgia;
            }
            QLabel#LoadingTitle {
                font-size: 24px;
                font-weight: 700;
                color: #eacb00;
            }
            QProgressBar {
                border: 1px solid #3f454c;
                border-radius: 8px;
                background: #0e1012;
                color: #ffffff;
                text-align: center;
                min-height: 26px;
            }
            QProgressBar::chunk {
                background-color: #eacb00;
                border-radius: 7px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(180, 180, 180, 180)
        layout.addStretch()
        self.title = QLabel("Loading…")
        self.title.setObjectName("LoadingTitle")
        self.detail = QLabel("")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.title, alignment=Qt.AlignCenter)
        layout.addWidget(self.detail, alignment=Qt.AlignCenter)
        layout.addWidget(self.progress)
        layout.addStretch()
        self.hide()

    def sync_geometry(self) -> None:
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())

    def on_loader_update(self, state: LoaderState) -> None:
        self.sync_geometry()
        self.title.setText(state.message or "Loading…")
        percent = int((state.progress or 0.0) * 100)
        self.progress.setValue(max(0, min(100, percent)))
        eta_text = f"ETA: {state.eta:.1f}s" if state.eta is not None else "ETA: --"
        self.detail.setText(eta_text)
        self.setVisible(state.is_running)
