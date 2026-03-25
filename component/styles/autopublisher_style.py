"""Styling constants for ContentOps Autopublisher."""

from __future__ import annotations

from typing import Iterable

UNSUPPORTED_QSS_PROPERTIES: tuple[str, ...] = (
    "transition",
    "box-shadow",
    "filter",
)


AUTOPUBLISHER_STYLE = """
QMainWindow {
    background-color: #0f1218;
    color: #e8edf5;
}
QWidget {
    background-color: transparent;
    color: #e8edf5;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 12px;
}
QFrame#Sidebar {
    background-color: #0b0f14;
    border-right: 1px solid #202938;
}
QFrame#Topbar, QFrame#StatsRow {
    background-color: #141a24;
    border: 1px solid #232d3e;
    border-radius: 10px;
}
QFrame#StageColumn, QFrame#DetailPanel, QFrame#StatCard, QFrame#AgentFleetCard {
    background-color: #151d29;
    border: 1px solid #2a3548;
    border-radius: 12px;
}
QFrame#TopicCard {
    background-color: #182131;
    border: 1px solid #304059;
    border-radius: 10px;
}
QLabel#AppTitle {
    font-size: 18px;
    font-weight: 600;
}
QLabel#PanelTitle, QLabel#ColumnTitle {
    font-size: 14px;
    font-weight: 600;
    color: #f4f7fb;
}
QLabel#Muted {
    color: #9aa9bf;
    font-size: 11px;
}
QPushButton {
    background-color: #243146;
    border: 1px solid #3a4a66;
    color: #e9eff7;
    border-radius: 8px;
    padding: 6px 10px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #2f4060;
}
QPushButton:pressed {
    background-color: #1d2a41;
}
QPushButton#Primary {
    background-color: #3b5ea8;
    border: 1px solid #5478c0;
}
QListWidget {
    background-color: transparent;
    border: none;
    outline: 0;
}
QListWidget::item {
    margin: 4px;
}
QPlainTextEdit, QTextEdit {
    background-color: #101722;
    border: 1px solid #2c3a51;
    border-radius: 8px;
    color: #dce6f5;
    padding: 6px;
}
QProgressBar {
    border: 1px solid #2f3d56;
    border-radius: 6px;
    background: #0f1520;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #4d7bd6;
    border-radius: 6px;
}
"""


def sanitize_qss(qss: str, unsupported: Iterable[str] = UNSUPPORTED_QSS_PROPERTIES) -> str:
    """Remove unsupported CSS properties that Qt stylesheets cannot parse."""
    unsupported_set = {p.strip().lower() for p in unsupported}
    cleaned: list[str] = []
    for line in qss.splitlines():
        stripped = line.strip().lower()
        if ":" in stripped:
            key = stripped.split(":", 1)[0].strip()
            if key in unsupported_set:
                continue
        cleaned.append(line)
    return "\n".join(cleaned)