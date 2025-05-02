from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class MusicianSetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SLAI: Musician Model")
        self.setFixedSize(400, 400)
        self.setStyleSheet("""
            QDialog {
                background-color: black;
                border: 4px solid gold;
                border-radius: 20px;
            }
            QLabel {
                color: white;
                font-size: 18px;
                font-family: 'Times New Roman';
            }
            QComboBox {
                background-color: white;
                color: black;
                font-size: 16px;
                padding: 6px;
                border-radius: 20px;
            }
            QPushButton {
                background-color: gold;
                color: black;
                font-weight: bold;
                border-radius: 10px;
                padding: 8px;
            }
        """)

        layout = QVBoxLayout(self)

        title = QLabel("SLAI: Musician Model")
        title.setFont(QFont('Times New Roman', 24, QFont.Bold))
        title.setStyleSheet("color: gold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.expert_label = QLabel("Choose your expert level")
        layout.addWidget(self.expert_label)

        self.expert_combo = QComboBox()
        self.expert_combo.addItems(["Beginner", "Intermediate", "Advanced"])
        layout.addWidget(self.expert_combo)

        self.instrument_label = QLabel("Choose an instrument")
        layout.addWidget(self.instrument_label)

        self.instrument_combo = QComboBox()
        self.instrument_combo.addItems(["Piano", "Guitar", "Violin", "Cello", "Bass", "Drum", "Vocalist"])
        layout.addWidget(self.instrument_combo)

        self.submit_button = QPushButton("Confirm")
        self.submit_button.clicked.connect(self.accept)
        layout.addWidget(self.submit_button)

    def get_choices(self):
        return (self.expert_combo.currentText(), self.instrument_combo.currentText())
