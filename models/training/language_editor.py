import json

from functools import partial
from collections import deque
from PyQt5.QtCore import pyqtSignal, Qt, QSize, QObject, QTimer
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QTextEdit, QVBoxLayout, QProgressBar, QLineEdit,
    QFrame, QScrollArea, QDialog, QHBoxLayout, QListWidgetItem, QLabel, QSizePolicy, QGridLayout # Keep QGridLayout if needed for future complex layouts
)
from models.training.shared_signals import training_signals, safe_connect
from logs.logger import get_logger
logger = get_logger(__name__)

#class TrainingSignals(QObject):
#    log_message = pyqtSignal(str)
#training_signals = TrainingSignals() # Replace with actual import

STRUCTURED_WORDLIST_PATH = "src/agents/language/structured_wordlist_en.json"

class LanguageEditorSignals(QObject):
    term_decision = pyqtSignal(str, str, str, str) # word, term_type ('synonym'/'related'), term, decision ('keep'/'reject'/'<new_term>')
    log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gui_decisions = {}

        self.request_approval.connect(self.show_approval_prompt)
        self.approval_result.connect(self.receive_approval)

    def show_approval_prompt(self, word, item, term, term_type):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Approval Required: {term_type}")
    
        layout = QVBoxLayout(dialog)
        label = QLabel(f"Approve {term_type} '{term}' for word '{word}'?")
        layout.addWidget(label)
    
        input_field = QLineEdit()
        input_field.setPlaceholderText("y / n / or corrected term")
        layout.addWidget(input_field)
    
        button = QPushButton("Submit")
        layout.addWidget(button)

        def submit_decision():
            decision = input_field.text().strip()
            if not decision:
                decision = 'y'
            self.editor_signals.term_decision.emit(word, term_type, term, decision)
            dialog.accept()
    
        button.clicked.connect(submit_decision)
        dialog.exec_()

    def receive_approval(self, item, decision):
        self.gui_decisions[item] = decision

class LanguageEditorSignals(QObject):
    term_decision = pyqtSignal(str, str, str, str) # word, term_type ('synonym'/'related'), term, decision ('keep'/'reject'/'<new_term>')
    log_message = pyqtSignal(str)

class LanguageEditor(QWidget):
    saveClicked = pyqtSignal(dict)
    loadClicked = pyqtSignal()
    rejectBatchClicked = pyqtSignal()
    nextBatchClicked = pyqtSignal()
    prevBatchClicked = pyqtSignal()
    shuffleBatchClicked = pyqtSignal()
    approveBatchClicked = pyqtSignal()

    editor_signals = LanguageEditorSignals()

    def __init__(self, parent=None):
        super().__init__(parent)
        # from models.training.shared_signals import training_signals # Ensure this import is correct
        self.init_ui()
        self.current_batch_widgets = {}
        self.current_batch_data = {}
        self.last_batch_data = None # Keep for undo functionality
        # Connect log signals
        self.editor_signals.log_message.connect(self.append_log)
        safe_connect(training_signals.log_message, self.append_log)
        safe_connect(training_signals.batch_status, self.update_batch_status)

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 1. Top bar with batch counter and progress bar
        top_bar = QHBoxLayout()
        self.batch_counter = QLabel("Batch 0/0")
        self.batch_counter.setStyleSheet("color: #38b4fe; font-size: 15px; font-weight: bold;")
        top_bar.addWidget(self.batch_counter)
        top_bar.addStretch()

        self.training_progress = QProgressBar()
        self.training_progress.setMaximum(100)
        self.training_progress.setValue(0)
        self.training_progress.setTextVisible(False) # Text often looks better outside or not at all
        self.training_progress.setFixedHeight(15)
        self.training_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #38b4fe;
                border-radius: 2px;
                background-color: #2a2a2a;
                font-size: 12px;
                text-align: center; /* Not visible if text not shown */
            }
            QProgressBar::chunk {
                background-color: #f9f9f9;
                border-radius: 3px;
            }
        """)
        top_bar.addWidget(self.training_progress)
        main_layout.addLayout(top_bar)

        # 2. Scrollable card area
        self.word_area = QScrollArea()
        self.word_area.setWidgetResizable(True)
        self.word_area.setStyleSheet("""
            QScrollArea {
                background-color: #0e0e0e; /* Dark background for the area */
                border: none; /* No border for the scroll area itself */
            }
        """)
        self.word_container_widget = QWidget() # This widget holds the layout for cards
        self.word_container_widget.setStyleSheet("background-color: #0e0e0e;") # Match scroll area background
        self.word_container_layout = QVBoxLayout(self.word_container_widget) # Vertical layout for cards
        self.word_container_layout.setSpacing(8) # Spacing between cards
        self.word_container_layout.setContentsMargins(5, 5, 5, 5) # Margins inside the container
        self.word_container_layout.setAlignment(Qt.AlignTop) # Align cards to the top
        self.word_area.setWidget(self.word_container_widget)
        main_layout.addWidget(self.word_area, 1) # Add stretch factor 1

        # 3. Log Output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #1c1c1c; border: 1px solid #444; color: #aaa; font-size: 11px; border-radius: 4px;") # Slightly different background
        self.log_output.setFixedHeight(100) # Fixed height for log area
        main_layout.addWidget(self.log_output)

        # 4. Search Bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Filter by word...")
        self.search_bar.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px;
                font-size: 16px;
                color: #eee;
            }
        """)
        self.search_bar.textChanged.connect(self.filter_cards)
        main_layout.addWidget(self.search_bar)

        # 5. Navigation button bar
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(0) # Compact buttons
        nav_layout.setContentsMargins(0, 5, 0, 5) # Add some vertical margin

        def nav_button(icon_path, tooltip, slot):
            btn = QPushButton()
            # Make sure icon paths are correct relative to where you run script
            try:
                icon = QIcon(icon_path)
                if icon.isNull():
                    logger.warning(f"Icon not found or invalid: {icon_path}. Using text instead.")
                    btn.setText(tooltip[0]) # Use first letter as placeholder
                else:
                    btn.setIcon(icon)
            except Exception as e:
                 logger.error(f"Error loading icon {icon_path}: {e}. Using text instead.")
                 btn.setText(tooltip[0]) # Use first letter

            btn.setToolTip(tooltip)
            btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    padding: 5px 10px; /* Add padding */
                    background-color: transparent;
                    color: #aaa; /* Icon color depends on SVG/PNG */
                }
                QPushButton:hover {
                    background-color: #2a2a2a; /* Slightly lighter hover */
                }
                QPushButton:pressed {
                    background-color: #1a1a1a;
                }
            """)
            btn.setIconSize(QSize(25, 25))
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(slot)
            return btn

        self.btn_reject = nav_button("frontend/assets/icons/downvote_icon.svg", "Reject Batch", self.reject_batch)
        self.btn_prev = nav_button("frontend/assets/icons/previous0.svg", "Previous Batch", self.previous_batch)
        self.btn_undo = nav_button("frontend/assets/icons/repeat0.svg", "Undo Batch", self.undo_last_batch)
        self.btn_shuffle = nav_button("frontend/assets/icons/shuffle0.svg", "Shuffle Batch", self.shuffle_batch)
        self.btn_next = nav_button("frontend/assets/icons/next0.svg", "Next Batch", self.next_batch)
        self.btn_approve = nav_button("frontend/assets/icons/upvote_icon.svg", "Approve Batch", self.approve_batch)

        nav_layout.addStretch() # Push buttons to the center/sides if needed
        for btn in [self.btn_reject, self.btn_prev, self.btn_undo, self.btn_shuffle, self.btn_next, self.btn_approve]:
            nav_layout.addWidget(btn)
        nav_layout.addStretch()
        main_layout.addLayout(nav_layout)

    def reject_batch(self):
        self.rejectBatchClicked.emit()
        self.append_log("[Reject] Batch rejection signal sent.")

    def previous_batch(self):
        self.prevBatchClicked.emit()
        self.append_log("[Previous] Move to previous batch signal sent.")

    def undo_last_batch(self):
        if self.last_batch_data:
            self.display_batch(self.last_batch_data)
            self.append_log("[Undo] Last batch redisplayed.")
        else:
            self.append_log("[Undo] No previous batch data available.")

    def shuffle_batch(self):
        self.shuffleBatchClicked.emit()
        self.append_log("[Shuffle] Shuffle batch signal sent.")

    def next_batch(self):
        self.nextBatchClicked.emit()
        self.append_log("[Next] Move to next batch signal sent.")

    def approve_batch(self):
        self.approveBatchClicked.emit()
        self.append_log("[Approve] Batch approval signal sent.")

    def show_approval_dialog(self, word, entry):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Approve: {word}")
        layout = QVBoxLayout(dialog)

        label = QLabel(f"<b>{word}</b><br>Synonyms: {entry.get('synonyms', [])}<br>Related: {entry.get('related_terms', [])}")
        label.setWordWrap(True)
        layout.addWidget(label)

        btn_layout = QHBoxLayout()
        btn_approve = QPushButton("Approve")
        btn_reject = QPushButton("Reject")
        btn_approve.clicked.connect(lambda: (self._save_approved_entry(word, entry), dialog.accept()))
        btn_reject.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_approve)
        btn_layout.addWidget(btn_reject)

        layout.addLayout(btn_layout)
        dialog.exec_()

    # --- Modified display_batch ---
    def display_batch(self, word_entries):
        # Store current data for potential undo BEFORE clearing
        self.last_batch_data = list(self.current_batch_data.items()) # Store previous data
        # Clear previous cards and data
        self.current_batch_data.clear()
        self.current_batch_widgets.clear()
        # Safer way to clear layout
        while self.word_container_layout.count():
            item = self.word_container_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not word_entries:
            no_data_label = QLabel("No entries to display in this batch.")
            no_data_label.setAlignment(Qt.AlignCenter)
            no_data_label.setStyleSheet("color: #777; margin: 20px;")
            self.word_container_layout.addWidget(no_data_label)
            return

        # Store data for potential modification
        for word, entry in word_entries:
             # Store a copy to prevent modification issues if entry is mutable
             self.current_batch_data[word] = entry.copy()

        # Create new cards
        for word, entry in word_entries:
            card = QFrame()
            card.setObjectName(f"card_{word}") # For potential lookup
            card.setStyleSheet("""
                QFrame {
                    background-color: #1a1a1a;
                    border-radius: 6px;
                    padding: 10px; 
                    margin-bottom: 12px;
                    min-height: 30px;
                }
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(4, 4, 4, 4) # Margins inside the card
            card_layout.setSpacing(3)         # Spacing between elements in card

            # Display Word (Header style) - Assuming 'word' is the key
            header = QLabel(f"\"{word}\"") # Add quotes like in screenshot
            header.setStyleSheet("color: #38b4fe; font-weight: bold; font-size: 14px; margin-bottom: 4px;")
            card_layout.addWidget(header)

            # Display POS - Assuming 'pos' exists in the entry dict
            pos_text = f"pos: [{', '.join(entry.get('pos', ['N/A']))}]" # Handle list or single string POS
            pos_label = QLabel(pos_text)
            pos_label.setStyleSheet("color: #aaa; font-size: 16px; margin-bottom: 3px;")
            pos_label.setWordWrap(True)
            card_layout.addWidget(pos_label)

            # Display Synonyms - Use the simplified display function
            self.add_term_display(card_layout, word, "synonyms", entry.get("synonyms", []))

            # Display Related Terms - Use the simplified display function
            self.add_term_display(card_layout, word, "related_terms", entry.get("related_terms", []))

            self.word_container_layout.addWidget(card)
            self.current_batch_widgets[word] = card # Keep reference if needed

        self.word_container_layout.addStretch(1)
        QTimer.singleShot(0, lambda: self.word_area.verticalScrollBar().setValue(0))

    def update_batch_status(self, status_message):
        # Simple: append to log
        self.append_log(f"[STATUS] {status_message}")

        # Optional: update a dedicated label on the UI
        if not hasattr(self, 'batch_status_label'):
            self.batch_status_label = QLabel()
            self.batch_status_label.setStyleSheet("color: #f9f9f9; font-weight: bold;")
            self.layout().insertWidget(1, self.batch_status_label)  # Insert below top bar

        self.batch_status_label.setText(status_message)

    def add_term_display(self, layout, word, term_type, terms):
        type_label = QLabel(f"{term_type.replace('_', ' ').capitalize()}s:")
        type_label.setStyleSheet("font-weight: bold; color: #ccc; margin-top: 5px;")
        layout.addWidget(type_label)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(3)
        layout.addLayout(grid_layout)

        row = 0
        for term in terms:
            term_label = QLineEdit(term) # Use QLineEdit for editing
            term_label.setStyleSheet("QLineEdit { background-color: #2a2a2a; border: 1px solid #444; border-radius: 3px; padding: 1px 3px; color: #f0f0f0; }")
            term_label.setReadOnly(True) # Initially read-only

            btn_edit = QPushButton("E")
            btn_edit.setFixedSize(20, 20)
            btn_edit.setToolTip("Edit")
            # Use partial to pass arguments to the lambda
            btn_edit.clicked.connect(partial(self.handle_edit_term, term_label))

            btn_keep = QPushButton("✔")
            btn_keep.setFixedSize(20, 20)
            btn_keep.setStyleSheet("QPushButton { color: lightgreen; }")
            btn_keep.setToolTip("Keep")
            # Use partial to pass arguments
            btn_keep.clicked.connect(partial(self.handle_term_decision, word, term_type, term, 'keep', term_label))

            btn_reject = QPushButton("✖")
            btn_reject.setFixedSize(20, 20)
            btn_reject.setStyleSheet("QPushButton { color: red; }")
            btn_reject.setToolTip("Reject")
            # Use partial to pass arguments
            btn_reject.clicked.connect(partial(self.handle_term_decision, word, term_type, term, 'reject', term_label))

            grid_layout.addWidget(term_label, row, 0)
            grid_layout.addWidget(btn_edit, row, 1)
            grid_layout.addWidget(btn_keep, row, 2)
            grid_layout.addWidget(btn_reject, row, 3)
            row += 1

        # Add 'Add New' button
        btn_add = QPushButton("+ Add")
        btn_add.setStyleSheet("QPushButton { color: #38b4fe; background-color: transparent; border: none; text-align: left; }")
        btn_add.clicked.connect(partial(self.handle_add_term, word, term_type, grid_layout))
        layout.addWidget(btn_add)

    def handle_add_term(self, word, term_type, grid_layout):
        new_term_input = QLineEdit()
        new_term_input.setPlaceholderText(f"Enter new {term_type}")
        grid_layout.addWidget(new_term_input)

        save_button = QPushButton("Save")
        def save_new_term():
            new_term = new_term_input.text().strip()
            if new_term:
                self.editor_signals.term_decision.emit(word, term_type, "", new_term)
                new_term_input.setDisabled(True)
                save_button.setDisabled(True)
        save_button.clicked.connect(save_new_term)
        grid_layout.addWidget(save_button)

#    def handle_add_term(self, word, term_type, grid_layout):
#        # Add a new empty row with input and buttons
#        row = grid_layout.rowCount()
#        new_term_input = QLineEdit()
#        new_term_input.setPlaceholderText("Enter new term...")
#        new_term_input.setStyleSheet("QLineEdit { background-color: #444; border: 1px solid #777; border-radius: 3px; padding: 1px 3px; color: white; }")

#        btn_save_new = QPushButton("✔")
#        btn_save_new.setFixedSize(20, 20)
#        btn_save_new.setStyleSheet("QPushButton { color: lightgreen; }")
#        btn_save_new.setToolTip("Save New")
#        btn_save_new.clicked.connect(partial(self.save_new_term, word, term_type, new_term_input, grid_layout, row))

#        btn_cancel_new = QPushButton("✖")
#        btn_cancel_new.setFixedSize(20, 20)
#        btn_cancel_new.setStyleSheet("QPushButton { color: red; }")
#        btn_cancel_new.setToolTip("Cancel Add")
#        btn_cancel_new.clicked.connect(partial(self.cancel_add_term, grid_layout, row))

#        grid_layout.addWidget(new_term_input, row, 0)
#        grid_layout.addWidget(btn_save_new, row, 2) # Place save where keep was
#        grid_layout.addWidget(btn_cancel_new, row, 3) # Place cancel where reject was

    def handle_edit_term(self, term_label):
        if term_label.isReadOnly():
            term_label.setReadOnly(False)
            term_label.setStyleSheet("QLineEdit { background-color: #444; border: 1px solid #777; border-radius: 3px; padding: 1px 3px; font-size: 16px; color: white; }") # Indicate editing
            term_label.setFocus()
        else:
            # Logic when editing is finished (e.g., Enter press or focus lost)
            term_label.setReadOnly(True)
            term_label.setStyleSheet("QLineEdit { background-color: #2a2a2a; border: 1px solid #444; border-radius: 3px; padding: 1px 3px; font-size: 16px; color: #f0f0f0; }")
            # Here you might want to emit the 'term_decision' signal with the new term
            # This requires knowing the original term. Storing it in the widget or using a closure might be needed.
            # For simplicity, we'll assume the decision is made via Keep/Reject after editing.

    def handle_term_decision(self, word, term_type, original_term, decision, term_widget):
        new_term = term_widget.text() if not term_widget.isReadOnly() else original_term
        final_decision = new_term if decision == 'keep' and new_term != original_term else decision

        # Optionally update widget appearance (e.g., strike-through for reject)
        if decision == 'reject':
             term_widget.setStyleSheet("QLineEdit { background-color: #2a2a2a; border: 1px solid #444; border-radius: 3px; padding: 1px 3px; color: #777; text-decoration: line-through; }")
        elif final_decision != 'keep': # Modified and kept
             term_widget.setStyleSheet("QLineEdit { background-color: #2a2a2a; border: 1px solid lightgreen; border-radius: 3px; padding: 1px 3px; color: lightgreen; }") # Indicate saved change
        else: # Kept original
             term_widget.setStyleSheet("QLineEdit { background-color: #2a2a2a; border: 1px solid #444; border-radius: 3px; padding: 1px 3px; color: lightgreen; }") # Indicate kept

        term_widget.setReadOnly(True) # Ensure it's read-only after decision

        # Emit the signal for the backend
        self.editor_signals.term_decision.emit(word, term_type, original_term, final_decision)
        self.append_log(f"Decision for {word} [{term_type}]: '{original_term}' -> '{final_decision}'")

    # --- REMOVED/COMMENTED OUT term interaction methods ---
    # def add_term_widgets(self, layout, word, term_type, terms): ...
    # def save_new_term(self, word, term_type, input_widget, grid_layout, row): ...
    # def cancel_add_term(self, grid_layout, row): ...
    # If you need interaction again, you'll need to redesign how it works with this layout.
    # Maybe double-clicking a card opens an edit dialog?

    def search_entire_wordlist(self, search_text, all_word_entries):
        search_text = search_text.strip().lower()
        matched = [(word, entry) for word, entry in all_word_entries if search_text in word.lower()]
        if matched:
            self.display_batch(matched)
        else:
            self.append_log(f"No matches found for '{search_text}'.")

    def filter_cards(self, text):
        """Filter cards based on the word (header text)."""
        text = text.strip().lower()
        # Iterate through the widgets managed by the layout
        for i in range(self.word_container_layout.count()):
            item = self.word_container_layout.itemAt(i)
            widget = item.widget()
            # Check if it's one of our cards (QFrame)
            if isinstance(widget, QFrame):
                # Find the header label within the card
                header_label = widget.findChild(QLabel)
                if header_label:
                    # Extract word, remove quotes, compare
                    word_in_card = header_label.text().strip('\"').lower()
                    widget.setVisible(text in word_in_card)

    def append_log(self, message: str):
        """Append a message to the log panel."""
        # Ensure log_output exists
        if hasattr(self, 'log_output'):
            self.log_output.append(message)
            self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum()) # Auto-scroll
        else:
            logger.warning("Attempted to log before log_output was initialized.")


    def update_batch_counter(self, current, total):
        self.batch_counter.setText(f"Batch {current}/{total}")

    def update_progress(self, percent):
        self.training_progress.setValue(percent)

    # --- Backend/Data Handling Methods (Keep or adapt as needed) ---
    # These methods (_save_approved_entry, load_entry, emit_*) interact
    # with your data logic and signals. Review them to ensure they still
    # work correctly with the potentially changed data flow (e.g., batch approval
    # might need to iterate through self.current_batch_data).

    # Example: _save_approved_entry might need adjustment if it used output_area
    def _save_approved_entry(self, word, entry):
         """Save approved entry to structured wordlist"""
         try:
             # Make sure file exists or handle FileNotFoundError
             try:
                with open(STRUCTURED_WORDLIST_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
             except FileNotFoundError:
                 data = {"words": {}} # Create structure if file missing

             data['words'][word] = entry # Add or update the word entry

             with open(STRUCTURED_WORDLIST_PATH, 'w', encoding='utf-8') as f:
                 json.dump(data, f, indent=2, ensure_ascii=False) # Use 'w' to overwrite, ensure_ascii=False for wider char support

             # Log success to the log panel
             log_entry = (
                 f"✓ Entry saved for \"{word}\":\n"
                 f"  pos: {entry.get('pos', 'N/A')}\n"
                 f"  synonyms: {entry.get('synonyms', [])}\n"
                 f"  related_terms: {entry.get('related_terms', [])}\n"
                 + "-"*40
             )
             self.append_log(log_entry)

         except Exception as e:
             self.append_log(f"Error saving \"{word}\": {str(e)}")
             logger.error(f"Error saving {word} to {STRUCTURED_WORDLIST_PATH}: {e}", exc_info=True)

    # These emit signals based on batch actions, ensure they work as intended
    # def emit_approve(self): self.approveBatchClicked.emit() # Example: Signal might not need data now
    # def emit_reject(self): self.rejectBatchClicked.emit()
    # def emit_save(self): self.saveClicked.emit(self.current_batch_data) # Example: Save might operate on current batch
