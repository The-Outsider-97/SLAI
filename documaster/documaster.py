"""DocuMaster desktop module for SLAIHub.

SLAIHub's main.py opens DocuMaster by importing DocumasterWindow from this file.
This module opens only DocuMaster, registers optional DocuMaster AI routes, and
sets explicit upload/file-processing limits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from flask import Flask, jsonify, send_from_directory  # type: ignore
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QComboBox,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QMainWindow,
)

from .routes.ai_routes import create_ai_blueprint
from .services.document_ai_service import DocumentAIService
from .services.document_extractor import DEFAULT_MAX_FILE_SIZE_BYTES
from .styles.documaster_style import DOCMASTER_STYLE, sanitize_qss
from .utils.documaster_utils import DocMasterFileService, MergePlan, MergePageItem, format_payload

try:
    from component.utils.loading_overlay import LoadingOverlay  # type: ignore
    from src.functions.loading import create_loading_controller, start_loading, update_loading, complete_loading  # type: ignore
except Exception:  # noqa: BLE001
    LoadingOverlay = None  # type: ignore

    class _NoopLoader:
        on_update = None

    def create_loading_controller():  # type: ignore
        return _NoopLoader()

    def start_loading(*_args, **_kwargs) -> None:  # type: ignore
        return None

    def update_loading(*_args, **_kwargs) -> None:  # type: ignore
        return None

    def complete_loading(*_args, **_kwargs) -> None:  # type: ignore
        return None


MAX_UPLOAD_MB = int(DEFAULT_MAX_FILE_SIZE_BYTES / (1024 * 1024))
SUPPORTED_AI_TASKS = {
    "Analyze": "analyze",
    "Summarize": "summarize",
    "Explain": "explain",
    "Ask questions": "ask",
    "Quality check": "quality_check",
    "Rewrite suggestions": "rewrite_suggestions",
    "Extract key points": "key_points",
}


def register_documaster_ai_routes(app: Flask, service: Optional[DocumentAIService] = None) -> None:
    """Register DocuMaster AI routes and explicit upload limits on a Flask app."""
    app.config["MAX_CONTENT_LENGTH"] = DEFAULT_MAX_FILE_SIZE_BYTES
    app.register_blueprint(create_ai_blueprint(service or DocumentAIService()))


def create_documaster_flask_app(*, static_folder: str | Path | None = None) -> Flask:
    """Optional Flask preview app. SLAIHub desktop GUI remains primary."""
    root = Path(__file__).resolve().parent
    public = Path(static_folder) if static_folder else root / "public"
    app = Flask(__name__, static_folder=str(public), static_url_path="")
    register_documaster_ai_routes(app)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "app": "DocuMaster", "maxUploadMb": MAX_UPLOAD_MB, "aiRoute": "/api/ai/health"})

    @app.errorhandler(413)
    def payload_too_large(_error):
        return jsonify({"status": "error", "error": f"File is too large. Maximum upload size is {MAX_UPLOAD_MB} MB."}), 413

    @app.get("/")
    def index():
        if (public / "index.html").exists():
            return send_from_directory(str(public), "index.html")
        return jsonify({"status": "ok", "message": "DocuMaster desktop GUI is the primary interface in SLAIHub."})

    return app


class Worker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.fn = fn

    def run(self) -> None:
        try:
            self.finished.emit(self.fn())
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DocumasterWindow(QMainWindow):
    """SLAIHub-native DocuMaster desktop GUI."""

    home_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DocuMaster · SLAI document workspace")
        self.resize(1540, 940)
        self.setMinimumSize(1180, 740)

        self.file_service = DocMasterFileService()
        self.ai_service = DocumentAIService()
        self.selected_file: Optional[Path] = None
        self.merge_files: List[Path] = []
        self.merge_plan: List[Dict[str, Any]] = []
        self.last_ai_payload: Optional[Dict[str, Any]] = None
        self._threads: List[QThread] = []

        self._build_ui()
        self._refresh_runtime_status()
        self._set_stat_defaults()

        if LoadingOverlay is not None:
            self.loading_overlay = LoadingOverlay(self.centralWidget())
            self.loading_overlay.sync_geometry()
            self.loading_controller = create_loading_controller()
            self.loading_controller.on_update = self.loading_overlay.on_loader_update
        else:
            self.loading_overlay = None
            self.loading_controller = create_loading_controller()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        self.setStyleSheet(sanitize_qss(DOCMASTER_STYLE))
        outer = QHBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)
        outer.addWidget(self._build_sidebar())

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)
        outer.addWidget(main, 1)
        main_layout.addWidget(self._build_topbar())
        main_layout.addWidget(self._build_stats_row())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_ai_tab(), "AI Assistant")
        self.tabs.addTab(self._build_reader_tab(), "Reader / Analysis")
        self.tabs.addTab(self._build_counter_tab(), "Word Counter")
        self.tabs.addTab(self._build_converter_tab(), "Converter")
        self.tabs.addTab(self._build_merger_tab(), "PDF Merge / Page Organizer")
        self.tabs.addTab(self._build_editor_tab(), "PDF Editor / Future Tools")
        self.tabs.addTab(self._build_runtime_tab(), "Runtime / SLAI Health")
        main_layout.addWidget(self.tabs, 1)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame(objectName="Sidebar")
        sidebar.setFixedWidth(240)
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(9)
        title = QLabel("DocuMaster", objectName="AppTitle")
        subtitle = QLabel("SLAI document workspace", objectName="Muted")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        for label, handler, primary in [
            ("Home", self._return_home, False),
            ("Initialize SLAI Runtime", self._initialize_runtime, True),
        ]:
            btn = QPushButton(label, objectName="Primary" if primary else "")
            btn.clicked.connect(handler)
            layout.addWidget(btn)
            if label.startswith("Initialize"):
                self.runtime_btn = btn

        layout.addSpacing(10)
        for label, index in [
            ("AI Assistant", 0),
            ("Reader / Analysis", 1),
            ("Word Counter", 2),
            ("Converter", 3),
            ("PDF Merge", 4),
            ("PDF Editor", 5),
            ("Runtime Health", 6),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked=False, tab=index: self.tabs.setCurrentIndex(tab))
            layout.addWidget(btn)

        layout.addStretch(1)
        self.runtime_status = QLabel("Runtime: unknown", objectName="Muted")
        self.runtime_status.setWordWrap(True)
        layout.addWidget(self.runtime_status)
        privacy = QLabel("Privacy: documents are processed in memory and are not stored by default.", objectName="Muted")
        privacy.setWordWrap(True)
        layout.addWidget(privacy)
        return sidebar

    def _build_topbar(self) -> QFrame:
        topbar = QFrame(objectName="Topbar")
        layout = QHBoxLayout(topbar)
        brand = QLabel("DocuMaster AI Document Assistant", objectName="AppTitle")
        self.status_label = QLabel("Ready. Select a document to start.", objectName="Muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(brand)
        layout.addStretch(1)
        layout.addWidget(self.status_label)
        return topbar

    def _build_stats_row(self) -> QFrame:
        row = QFrame(objectName="StatsRow")
        layout = QHBoxLayout(row)
        self.stat_labels: Dict[str, QLabel] = {}
        for label in ["Words", "Characters", "Pages", "Reader", "Confidence"]:
            card = QFrame(objectName="StatCard")
            box = QVBoxLayout(card)
            box.addWidget(QLabel(label, objectName="Muted"))
            value = QLabel("--", objectName="PanelTitle")
            box.addWidget(value)
            self.stat_labels[label] = value
            layout.addWidget(card)
        return row

    def _build_ai_tab(self) -> QWidget:
        body = QWidget()
        layout = QHBoxLayout(body)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        left = QFrame(objectName="StageColumn")
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Document Input", objectName="ColumnTitle"))
        self.file_label = QLabel("No document selected", objectName="Muted")
        self.file_label.setWordWrap(True)
        select_btn = QPushButton("Select document", objectName="Primary")
        select_btn.clicked.connect(self._select_document)
        clear_btn = QPushButton("Clear / reset")
        clear_btn.clicked.connect(self._clear_document)
        left_layout.addWidget(self.file_label)
        left_layout.addWidget(select_btn)
        left_layout.addWidget(clear_btn)

        left_layout.addWidget(QLabel("Action", objectName="ColumnTitle"))
        self.action_combo = QComboBox()
        self.action_combo.addItems(list(SUPPORTED_AI_TASKS.keys()))
        self.action_combo.currentTextChanged.connect(self._toggle_question_input)
        left_layout.addWidget(self.action_combo)

        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Question for document Q&A")
        self.question_input.setVisible(False)
        left_layout.addWidget(self.question_input)

        run_btn = QPushButton("Run SLAI Assistant", objectName="Primary")
        run_btn.clicked.connect(self._run_ai_task)
        left_layout.addWidget(run_btn)
        left_layout.addStretch(1)
        self.privacy_text = QLabel(
            f"Max file size: {MAX_UPLOAD_MB} MB. Supported: PDF, DOCX, TXT, HTML, XML, ODT. No permanent storage by default.",
            objectName="Muted",
        )
        self.privacy_text.setWordWrap(True)
        left_layout.addWidget(self.privacy_text)
        splitter.addWidget(left)

        right = QScrollArea()
        right.setWidgetResizable(True)
        right_body = QWidget()
        right_layout = QVBoxLayout(right_body)
        response_panel = QFrame(objectName="DetailPanel")
        response_layout = QVBoxLayout(response_panel)
        response_layout.addWidget(QLabel("AI Response", objectName="PanelTitle"))
        self.ai_response = QPlainTextEdit()
        self.ai_response.setReadOnly(True)
        response_layout.addWidget(self.ai_response)
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        response_layout.addWidget(self.confidence_bar)
        right_layout.addWidget(response_panel)

        raw_panel = QFrame(objectName="DetailPanel")
        raw_layout = QVBoxLayout(raw_panel)
        raw_layout.addWidget(QLabel("Structured JSON", objectName="PanelTitle"))
        self.ai_json = QPlainTextEdit()
        self.ai_json.setReadOnly(True)
        raw_layout.addWidget(self.ai_json)
        right_layout.addWidget(raw_panel)
        right.setWidget(right_body)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        return body

    def _build_reader_tab(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        actions = QHBoxLayout()
        select_btn = QPushButton("Select document", objectName="Primary")
        select_btn.clicked.connect(self._select_document)
        analyze_btn = QPushButton("Analyze selected document")
        analyze_btn.clicked.connect(lambda: self._run_named_task("analyze"))
        for btn in (select_btn, analyze_btn):
            actions.addWidget(btn)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.reader_output = QPlainTextEdit()
        self.reader_output.setReadOnly(True)
        layout.addWidget(self.reader_output, 1)
        return body

    def _build_counter_tab(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        btn = QPushButton("Select file", objectName="Primary")
        btn.clicked.connect(self._run_word_count)
        layout.addWidget(btn)
        self.counter_output = QPlainTextEdit()
        self.counter_output.setReadOnly(True)
        layout.addWidget(self.counter_output, 1)
        return body

    def _build_converter_tab(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        actions = QHBoxLayout()
        self.convert_format = QComboBox()
        self.convert_format.addItems(["txt", "docx", "html", "xml"])
        convert_btn = QPushButton("Convert selected document", objectName="Primary")
        convert_btn.clicked.connect(self._run_converter)
        actions.addWidget(QLabel("Convert to:"))
        actions.addWidget(self.convert_format)
        actions.addWidget(convert_btn)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.converter_output = QPlainTextEdit()
        self.converter_output.setReadOnly(True)
        layout.addWidget(self.converter_output, 1)
        return body

    def _build_merger_tab(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        actions = QHBoxLayout()
        buttons = [
            ("Add PDFs", self._add_merge_files, True),
            ("Build / refresh page plan", self._build_merge_plan, False),
            ("Move page up", self._move_merge_page_up, False),
            ("Move page down", self._move_merge_page_down, False),
            ("Include / exclude selected", self._toggle_selected_merge_pages, False),
            ("Merge selected plan", self._merge_selected_pdfs, True),
            ("Clear", self._clear_merge_files, False),
        ]
        for label, handler, primary in buttons:
            btn = QPushButton(label, objectName="Primary" if primary else "")
            btn.clicked.connect(handler)
            actions.addWidget(btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.merge_table = QTableWidget(0, 6)
        self.merge_table.setHorizontalHeaderLabels(["Include", "Order", "File", "Page", "Label", "Preview"])
        self.merge_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.merge_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.merge_table, 1)
        self.merge_output = QLabel("Select at least two PDFs, build a page plan, then include/exclude and reorder pages.", objectName="Muted")
        self.merge_output.setWordWrap(True)
        layout.addWidget(self.merge_output)
        return body

    def _build_editor_tab(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        title = QLabel("PDF Editor / Future Tools", objectName="PanelTitle")
        note = QLabel(
            "Prepared for future SLAI-powered PDF editing, rewriting, and comparison. Current implementation keeps originals unchanged and focuses on analysis, conversion, and merge-plan control.",
            objectName="Muted",
        )
        note.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(note)
        layout.addStretch(1)
        return body

    def _build_runtime_tab(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        actions = QHBoxLayout()
        init_btn = QPushButton("Initialize / refresh SLAI runtime", objectName="Primary")
        init_btn.clicked.connect(self._initialize_runtime)
        refresh_btn = QPushButton("Refresh health")
        refresh_btn.clicked.connect(self._refresh_runtime_status)
        actions.addWidget(init_btn)
        actions.addWidget(refresh_btn)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.runtime_output = QPlainTextEdit()
        self.runtime_output.setReadOnly(True)
        layout.addWidget(self.runtime_output, 1)
        return body

    # ------------------------------------------------------------------
    # Background worker helper
    # ------------------------------------------------------------------
    def _run_background(self, label: str, fn: Callable[[], Any], on_success: Callable[[Any], None]) -> None:
        start_loading(self.loading_controller, label)
        self.status_label.setText(label)
        thread = QThread(self)
        worker = Worker(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda result: self._finish_background(thread, worker, on_success, result))
        worker.failed.connect(lambda message: self._fail_background(thread, worker, message))
        self._threads.append(thread)
        thread.start()

    def _finish_background(self, thread: QThread, worker: Worker, on_success: Callable[[Any], None], result: Any) -> None:
        complete_loading(self.loading_controller, "Completed")
        on_success(result)
        worker.deleteLater()
        thread.quit()
        thread.wait()
        if thread in self._threads:
            self._threads.remove(thread)
        thread.deleteLater()

    def _fail_background(self, thread: QThread, worker: Worker, message: str) -> None:
        complete_loading(self.loading_controller, "Failed")
        self.status_label.setText("Task failed")
        QMessageBox.critical(self, "DocuMaster error", self._user_error(message))
        worker.deleteLater()
        thread.quit()
        thread.wait()
        if thread in self._threads:
            self._threads.remove(thread)
        thread.deleteLater()

    @staticmethod
    def _user_error(message: str) -> str:
        return str(message).split("Traceback", 1)[0].strip()

    # ------------------------------------------------------------------
    # Document AI actions
    # ------------------------------------------------------------------
    def _select_document(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select document",
            str(Path.home()),
            "Documents (*.pdf *.docx *.txt *.html *.htm *.xml *.odt)",
        )
        if not path:
            return
        self.selected_file = Path(path)
        self.file_label.setText(str(self.selected_file))
        self.status_label.setText(f"Selected: {self.selected_file.name}")

    def _clear_document(self) -> None:
        self.selected_file = None
        self.last_ai_payload = None
        self.file_label.setText("No document selected")
        self.ai_response.clear()
        self.ai_json.clear()
        self.confidence_bar.setValue(0)
        if hasattr(self, "reader_output"):
            self.reader_output.clear()
        self._set_stat_defaults()
        self.status_label.setText("Ready. Select a document to start.")

    def _toggle_question_input(self, text: str) -> None:
        self.question_input.setVisible(SUPPORTED_AI_TASKS.get(text) == "ask")

    def _run_named_task(self, task: str) -> None:
        if not self.selected_file:
            QMessageBox.information(self, "Document required", "Select a document first.")
            return
        self._run_background(
            "Running DocuMaster document analysis…",
            lambda: self.ai_service.run_path(task=task, path=self.selected_file),
            self._display_ai_payload,
        )

    def _run_ai_task(self) -> None:
        if not self.selected_file:
            QMessageBox.information(self, "Document required", "Select a document first.")
            return
        task = SUPPORTED_AI_TASKS[self.action_combo.currentText()]
        question = self.question_input.text().strip()
        if task == "ask" and not question:
            QMessageBox.information(self, "Question required", "Add a question before running document Q&A.")
            return
        self._run_background(
            "Running DocuMaster SLAI task…",
            lambda: self.ai_service.run_path(task=task, path=self.selected_file, question=question),
            self._display_ai_payload,
        )

    def _display_ai_payload(self, payload: Dict[str, Any]) -> None:
        self.last_ai_payload = payload
        self.ai_response.setPlainText(format_payload(payload))
        self.ai_json.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        if hasattr(self, "reader_output"):
            self.reader_output.setPlainText(format_payload(payload))
        self._apply_payload_stats(payload)
        self.status_label.setText(f"Completed: {payload.get('task', 'document task')}" if payload.get("status") in {"success", "partial"} else "Task failed")

    def _run_word_count(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Select document", str(Path.home()), "Documents (*.pdf *.docx *.txt *.html *.htm *.xml *.odt)")
        if not path:
            return
        self._run_background("Counting document words…", lambda: self.file_service.count_file(path), self._display_word_count)

    def _display_word_count(self, stats: Any) -> None:
        self.counter_output.setPlainText(
            f"File: {stats.filename}\n"
            f"Words: {stats.word_count}\nCharacters: {stats.char_count}\nSentences: {stats.sentence_count}\n"
            f"Average word length: {stats.avg_word_length}\nAverage sentence length: {stats.avg_sentence_length}\n"
            f"Unique words: {stats.unique_word_count}\nReadability: {stats.readability_score} ({stats.reading_level})\n\n"
            f"Top words:\n" + "\n".join(f"- {item['text']}: {item['value']}" for item in stats.top_words[:20]) +
            f"\n\nPreview:\n{stats.preview}"
        )
        self.stat_labels["Words"].setText(str(stats.word_count))
        self.stat_labels["Characters"].setText(str(stats.char_count))

    def _run_converter(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Select document", str(Path.home()), "Documents (*.pdf *.docx *.txt *.html *.htm *.xml *.odt)")
        if not path:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder", str(Path.home()))
        if not out_dir:
            return
        self._run_background(
            "Converting document…",
            lambda: self.file_service.convert_file(path, self.convert_format.currentText(), out_dir),
            lambda output: self.converter_output.setPlainText(f"Conversion successful.\n\nOutput:\n{output}"),
        )

    # ------------------------------------------------------------------
    # Merge/page organizer actions
    # ------------------------------------------------------------------
    def _add_merge_files(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(self, "Select PDFs", str(Path.home()), "PDF files (*.pdf)")
        for path in paths:
            p = Path(path)
            if p not in self.merge_files:
                self.merge_files.append(p)
        self.merge_output.setText(f"{len(self.merge_files)} PDF file(s) selected. Build the page plan to inspect and reorder pages.")

    def _clear_merge_files(self) -> None:
        self.merge_files.clear()
        self.merge_plan.clear()
        self.merge_table.setRowCount(0)
        self.merge_output.setText("Merge plan cleared.")

    def _build_merge_plan(self) -> None:
        if len(self.merge_files) < 2:
            QMessageBox.information(self, "More PDFs required", "Select at least two PDF files.")
            return
        self._run_background("Building PDF page plan…", lambda: self.file_service.build_pdf_merge_plan(self.merge_files).to_dict(), self._display_merge_plan)

    def _display_merge_plan(self, plan: Dict[str, Any]) -> None:
        self.merge_plan = list(plan.get("pages", []))
        self._refresh_merge_table()
        metadata = plan.get("metadata", {})
        self.merge_output.setText(
            f"Merge plan ready: {metadata.get('file_count', len(self.merge_files))} file(s), {metadata.get('page_count', len(self.merge_plan))} page(s). Uncheck or reorder pages before merging."
        )

    def _refresh_merge_table(self) -> None:
        self.merge_table.setRowCount(len(self.merge_plan))
        for row, item in enumerate(self.merge_plan):
            item["user_order_index"] = row
            include_item = QTableWidgetItem("Yes" if item.get("included", True) else "No")
            include_item.setCheckState(Qt.Checked if item.get("included", True) else Qt.Unchecked)
            include_item.setFlags(include_item.flags() | Qt.ItemIsUserCheckable)
            self.merge_table.setItem(row, 0, include_item)
            self.merge_table.setItem(row, 1, QTableWidgetItem(str(row + 1)))
            self.merge_table.setItem(row, 2, QTableWidgetItem(Path(str(item.get("source_file", ""))).name))
            self.merge_table.setItem(row, 3, QTableWidgetItem(str(item.get("page_number", ""))))
            self.merge_table.setItem(row, 4, QTableWidgetItem(str(item.get("page_label", ""))))
            preview = item.get("preview_metadata", {}).get("preview", "") if isinstance(item.get("preview_metadata", {}), dict) else ""
            self.merge_table.setItem(row, 5, QTableWidgetItem(str(preview)))
        self.merge_table.resizeColumnsToContents()

    def _sync_merge_plan_from_table(self) -> None:
        ordered: List[Dict[str, Any]] = []
        for row in range(self.merge_table.rowCount()):
            if row >= len(self.merge_plan):
                continue
            item = dict(self.merge_plan[row])
            include_cell = self.merge_table.item(row, 0)
            item["included"] = include_cell.checkState() == Qt.Checked if include_cell else bool(item.get("included", True))
            item["user_order_index"] = row
            ordered.append(item)
        self.merge_plan = ordered

    def _selected_merge_rows(self) -> List[int]:
        return sorted({index.row() for index in self.merge_table.selectedIndexes()})

    def _move_merge_page_up(self) -> None:
        self._sync_merge_plan_from_table()
        rows = self._selected_merge_rows()
        if not rows or rows[0] == 0:
            return
        for row in rows:
            self.merge_plan[row - 1], self.merge_plan[row] = self.merge_plan[row], self.merge_plan[row - 1]
        self._refresh_merge_table()
        for row in [r - 1 for r in rows]:
            self.merge_table.selectRow(row)

    def _move_merge_page_down(self) -> None:
        self._sync_merge_plan_from_table()
        rows = self._selected_merge_rows()
        if not rows or rows[-1] >= len(self.merge_plan) - 1:
            return
        for row in reversed(rows):
            self.merge_plan[row + 1], self.merge_plan[row] = self.merge_plan[row], self.merge_plan[row + 1]
        self._refresh_merge_table()
        for row in [r + 1 for r in rows]:
            self.merge_table.selectRow(row)

    def _toggle_selected_merge_pages(self) -> None:
        rows = self._selected_merge_rows()
        for row in rows:
            cell = self.merge_table.item(row, 0)
            if cell:
                cell.setCheckState(Qt.Unchecked if cell.checkState() == Qt.Checked else Qt.Checked)
                cell.setText("Yes" if cell.checkState() == Qt.Checked else "No")
        self._sync_merge_plan_from_table()

    def _merge_selected_pdfs(self) -> None:
        if not self.merge_plan:
            QMessageBox.information(self, "Page plan required", "Build the page plan before merging.")
            return
        self._sync_merge_plan_from_table()
        output, _filter = QFileDialog.getSaveFileName(self, "Save merged PDF", str(Path.home() / "merged.pdf"), "PDF files (*.pdf)")
        if not output:
            return
        plan = MergePlan(status="success", pages=[MergePageItem(**item) for item in self.merge_plan]).to_dict()
        self._run_background(
            "Merging selected PDF pages…",
            lambda: self.file_service.merge_pdfs_with_plan(plan, output),
            lambda merged: self.merge_output.setText(f"Merge successful. Output: {merged}"),
        )

    # ------------------------------------------------------------------
    # Runtime/status helpers
    # ------------------------------------------------------------------
    def _initialize_runtime(self) -> None:
        self._run_background("Initializing SLAI runtime…", self.ai_service.initialize_runtime, self._display_runtime_health)

    def _refresh_runtime_status(self) -> None:
        self._display_runtime_health(self.ai_service.health().get("slai", {}))

    def _display_runtime_health(self, health: Dict[str, Any]) -> None:
        slai_health = health if "available" in health else self.ai_service.health().get("slai", {})
        mode = slai_health.get("mode", "unknown")
        status = slai_health.get("status", "unknown")
        reader = "available" if slai_health.get("reader_agent_available") else "unavailable"
        self.runtime_status.setText(f"Runtime: {mode} · {status} · Reader: {reader}")
        if hasattr(self, "runtime_output"):
            self.runtime_output.setPlainText(json.dumps(slai_health, indent=2, ensure_ascii=False, default=str))
        self.status_label.setText(f"Runtime: {mode} / {status}")

    def _apply_payload_stats(self, payload: Dict[str, Any]) -> None:
        metadata = payload.get("metadata", {})
        result = payload.get("result", {})
        self.stat_labels["Words"].setText(str(metadata.get("word_count", 0)))
        self.stat_labels["Characters"].setText(str(metadata.get("char_count", 0)))
        self.stat_labels["Pages"].setText(str(metadata.get("page_count", 0)))
        self.stat_labels["Reader"].setText("SLAI" if metadata.get("reader_agent_used") else "Local")
        confidence = int(float(result.get("confidence", 0.0) or 0.0) * 100)
        self.stat_labels["Confidence"].setText(f"{confidence}%")
        self.confidence_bar.setValue(confidence)

    def _set_stat_defaults(self) -> None:
        for label in self.stat_labels.values():
            label.setText("--")
        if hasattr(self, "confidence_bar"):
            self.confidence_bar.setValue(0)

    def _return_home(self) -> None:
        if self.receivers(self.home_requested) > 0:
            self.home_requested.emit()
            return
        self.close()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.loading_overlay is not None:
            self.loading_overlay.sync_geometry()


def launch_documaster() -> None:
    existing_app = QApplication.instance()
    app = existing_app or QApplication(sys.argv)
    win = DocumasterWindow()
    win.show()
    if existing_app is None:
        app.exec_()


if __name__ == "__main__":
    launch_documaster()
