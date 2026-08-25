"""DocuMaster PyQt style following SLAIHub desktop apps."""

from __future__ import annotations


def sanitize_qss(style: str) -> str:
    return style.strip()


DOCMASTER_STYLE = """
QMainWindow, QWidget {
    background-color: #0e1012;
    color: #f5f7ff;
    font-family: Inter, Segoe UI, Arial, sans-serif;
    font-size: 13px;
}
QFrame#Sidebar { background-color: #0a0d10; border: 1px solid #222831; border-radius: 16px; }
QFrame#Topbar, QFrame#StatsRow, QFrame#DetailPanel, QFrame#StageColumn, QFrame#DropPanel {
    background-color: #121820; border: 1px solid #26313d; border-radius: 14px;
}
QFrame#StatCard {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #111923, stop:1 #17212d);
    border: 1px solid #2b3745; border-radius: 12px;
}
QLabel#AppTitle { color: #eacb00; font-size: 23px; font-weight: 800; }
QLabel#PanelTitle { color: #f5f7ff; font-size: 17px; font-weight: 700; }
QLabel#ColumnTitle { color: #eacb00; font-size: 15px; font-weight: 700; }
QLabel#Muted { color: #9aa4b2; }
QPushButton {
    background-color: #17212d; color: #f5f7ff; border: 1px solid #2b3745;
    border-radius: 10px; padding: 9px 13px; font-weight: 600;
}
QPushButton:hover { border-color: #eacb00; background-color: #1d2a39; }
QPushButton:pressed { background-color: #101820; }
QPushButton:disabled { color: #647080; background-color: #101820; border-color: #202833; }
QPushButton#Primary { background-color: #eacb00; color: #111111; border: 1px solid #ffe156; }
QPushButton#Primary:hover { background-color: #f5d71c; }
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QListWidget, QTableWidget {
    background-color: #0f151d; color: #f5f7ff; border: 1px solid #2b3745;
    border-radius: 10px; padding: 7px; selection-background-color: #eacb00; selection-color: #111111;
}
QHeaderView::section {
    background-color: #17212d; color: #f5f7ff; border: 1px solid #2b3745; padding: 6px; font-weight: 700;
}
QTableWidget::item { padding: 5px; }
QTableWidget::item:selected { background-color: #273548; color: #ffffff; }
QPlainTextEdit[readOnly='true'] { color: #d6deea; }
QComboBox::drop-down { border: none; width: 24px; }
QTabWidget::pane { border: 1px solid #26313d; border-radius: 12px; background-color: #121820; }
QTabBar::tab {
    background-color: #0f151d; color: #9aa4b2; border: 1px solid #26313d; border-bottom: none;
    border-top-left-radius: 10px; border-top-right-radius: 10px; min-width: 120px; padding: 9px 14px;
}
QTabBar::tab:selected { background-color: #17212d; color: #f5f7ff; }
QProgressBar {
    background-color: #0f151d; color: #f5f7ff; border: 1px solid #2b3745;
    border-radius: 9px; text-align: center; min-height: 18px;
}
QProgressBar::chunk { background-color: #eacb00; border-radius: 9px; }
QScrollBar:vertical { background: transparent; width: 10px; }
QScrollBar::handle:vertical { background: #334153; border-radius: 5px; min-height: 24px; }
"""
