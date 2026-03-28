from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QObject, QPoint, Qt, pyqtSignal
from PyQt5.QtWidgets import QAction, QMenu, QPushButton, QWidget

from src.functions.dropdown import DropdownMenu, DropdownOption


class UserMenuController(QObject):
    """Encapsulates the authenticated user button and its account menu."""

    logout_requested = pyqtSignal()
    profile_requested = pyqtSignal()
    settings_requested = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._parent = parent
        self._username: Optional[str] = None
        self._menu_model = DropdownMenu(
            [
                DropdownOption("Profile", "profile"),
                DropdownOption("Settings", "settings"),
                DropdownOption("Log out", "logout"),
            ]
        )

        self.button = QPushButton("Account", parent)
        self.button.setFixedSize(160, 38)
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.setStyleSheet(
            """
            QPushButton {
                border: 2px solid white;
                border-radius: 10px;
                background: transparent;
                color: white;
                font-family: Georgia;
                font-size: 16px;
                font-weight: bold;
                padding: 0 12px;
                text-align: left;
            }
            QPushButton:hover { background: rgba(255, 255, 255, 25); }
            """
        )

        self._menu = QMenu(parent)
        self._menu.setStyleSheet(
            """
            QMenu {
                background-color: #13171b;
                color: #ffffff;
                border: 1px solid #3f454c;
                border-radius: 8px;
                padding: 6px;
            }
            QMenu::item {
                padding: 8px 16px;
                border-radius: 6px;
            }
            QMenu::item:selected {
                background-color: #1d2329;
                color: #eacb00;
            }
            QMenu::separator {
                height: 1px;
                margin: 6px 4px;
                background: #3f454c;
            }
            """
        )

        self._username_action = QAction("Signed in", self._menu)
        self._username_action.setEnabled(False)
        self._menu.addAction(self._username_action)
        self._menu.addSeparator()

        self._actions = {}
        for option in self._menu_model.options:
            action = QAction(option.label, self._menu)
            action.triggered.connect(lambda _checked=False, value=option.value: self._on_action(value))
            self._menu.addAction(action)
            self._actions[option.value] = action

        self.button.clicked.connect(self.toggle_menu)
        self._menu.aboutToHide.connect(self._menu_model.close)

    def set_user(self, username: str) -> None:
        self._username = username
        self.button.setText(username)
        self._username_action.setText(f"Signed in as {username}")

    def clear_user(self) -> None:
        self._username = None
        self.button.setText("Account")
        self._username_action.setText("Signed in")
        self.hide_menu()

    def hide_menu(self) -> None:
        self._menu.hide()
        self._menu_model.close()

    def toggle_menu(self) -> None:
        if self._menu_model.toggle():
            anchor = self.button.mapToGlobal(QPoint(0, self.button.height() + 4))
            self._menu.popup(anchor)
            self._menu_model.is_open = True
        else:
            self.hide_menu()

    def _on_action(self, value: str) -> None:
        self._menu_model.select(value)
        self.hide_menu()
        if value == "logout":
            self.logout_requested.emit()
        elif value == "profile":
            self.profile_requested.emit()
        elif value == "settings":
            self.settings_requested.emit()
