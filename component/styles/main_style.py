"""Shared SLAI desktop styling primitives.

This module is intentionally reusable across multiple PyQt5 apps.
"""

MAIN_STYLE = """
QWidget {
    background-color: #0b1020;
    color: #e8ecff;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QFrame#panel {
    background-color: #111933;
    border: 1px solid #263055;
    border-radius: 14px;
}
QFrame#card {
    background-color: #0f1730;
    border: 1px solid #263055;
    border-radius: 12px;
}
QLabel[class='title'] {
    font-size: 22px;
    font-weight: 700;
    color: #f5f7ff;
}
QLabel[class='subtitle'] {
    color: #9aa7d1;
    font-size: 12px;
}
QLabel[class='sectionTitle'] {
    font-size: 15px;
    font-weight: 600;
    color: #f1f4ff;
}
QPushButton {
    border: 1px solid #2a3763;
    background-color: #172142;
    color: #e8ecff;
    border-radius: 10px;
    padding: 8px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #1d2b56;
}
QPushButton:pressed {
    background-color: #101a35;
}
QPushButton[class='primary'] {
    border: none;
    background-color: #5b8cff;
}
QPushButton[class='primary']:hover {
    background-color: #4b79e6;
}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox {
    border: 1px solid #2a3763;
    border-radius: 10px;
    padding: 7px;
    background-color: #0f1730;
    color: #f1f4ff;
}
QTabBar::tab {
    background: #0f1730;
    border: 1px solid #263055;
    border-bottom: none;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    min-width: 120px;
    padding: 8px 12px;
    color: #9aa7d1;
}
QTabBar::tab:selected {
    background: #172142;
    color: #f1f4ff;
}
QTableWidget {
    background-color: #0f1730;
    border: 1px solid #263055;
    border-radius: 10px;
    gridline-color: #263055;
}
QHeaderView::section {
    background-color: #111933;
    color: #dfe6ff;
    border: none;
    border-bottom: 1px solid #263055;
    padding: 6px;
    font-weight: 600;
}
QScrollBar:vertical {
    width: 10px;
    border: none;
    background: transparent;
}
QScrollBar::handle:vertical {
    background: #2a3763;
    border-radius: 5px;
    min-height: 24px;
}
"""


def metric_card_style() -> str:
    """Shared style for metric cards across SLAI desktop apps."""
    return (
        "QFrame {"
        "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, "
        "stop:0 #111933, stop:1 #172142);"
        "border: 1px solid #263055;"
        "border-radius: 12px;"
        "}"
    )
