"""DocMaster desktop module for SLAIHub.

This is the primary DocMaster entrypoint. SLAIHub's `main.py` should import
`DocumasterWindow` from this file when the Documaster app-card is clicked.

The module also exposes `create_documaster_flask_app()` for optional local/cloud
preview routes. The PyQt GUI does not depend on the browser `public/` frontend.
"""

from __future__ import annotations

import json
import sys

from pathlib import Path
from typing import Any, Dict, List, Optional
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (  # type: ignore
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QComboBox,
    QLineEdit,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from flask import Flask, jsonify, send_from_directory  # type: ignore

from .routes.ai_routes import create_ai_blueprint
from .services.document_ai_service import DocumentAIService
from .services.document_extractor import DEFAULT_MAX_FILE_SIZE_BYTES, DocumentExtractionError
from .styles.documaster_style import DOCMASTER_STYLE, sanitize_qss
from .utils.documaster_utils import DocMasterFileService, format_payload

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
    """Register DocMaster AI routes and explicit upload limits on a Flask app."""
    app.config["MAX_CONTENT_LENGTH"] = DEFAULT_MAX_FILE_SIZE_BYTES
    app.register_blueprint(create_ai_blueprint(service or DocumentAIService()))



def create_documaster_flask_app(*, static_folder: str | Path | None = None) -> Flask:
    """Optional Flask app factory for local/cloud preview.

    This is kept for compatibility, but the SLAIHub UI should use the PyQt
    `DocumasterWindow` below.
    """
    root = Path(__file__).resolve().parent
    public = Path(static_folder) if static_folder else root / "public"
    app = Flask(__name__, static_folder=str(public), static_url_path="")
    register_documaster_ai_routes(app)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "app": "DocMaster", "maxUploadMb": MAX_UPLOAD_MB, "aiRoute": "/api/ai/health"})

    @app.errorhandler(413)
    def payload_too_large(_error):
        return jsonify({"status": "error", "error": f"File is too large. Maximum upload size is {MAX_UPLOAD_MB} MB."}), 413

    @app.get("/")
    def index():
        if (public / "index.html").exists():
            return send_from_directory(str(public), "index.html")
        return jsonify({"status": "ok", "message": "DocMaster desktop GUI is the primary interface in SLAIHub."})

    return app


class DocumasterWindow(QMainWindow):
    """SLAIHub-native DocMaster desktop GUI."""

    home_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DocMaster · SLAI-driven")
        self.resize(1500, 900)
        self.setMinimumSize(1180, 720)

        self.file_service = DocMasterFileService()
        self.ai_service = DocumentAIService()
        self.selected_file: Optional[Path] = None
        self.merge_files: List[Path] = []
        self.last_ai_payload: Optional[Dict[str, Any]] = None

        self._build_ui()
        self._refresh_runtime_status()
        self._refresh_merge_list()

        if LoadingOverlay is not None:
            self.loading_overlay = LoadingOverlay(self.centralWidget())
            self.loading_overlay.sync_geometry()
            self.loading_controller = create_loading_controller()
            self.loading_controller.on_update = self.loading_overlay.on_loader_update
        else:
            self.loading_overlay = None
            self.loading_controller = create_loading_controller()

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
        self.tabs.addTab(self._build_ai_tab(), "SLAI Assistant")
        self.tabs.addTab(self._build_counter_tab(), "Word Counter")
        self.tabs.addTab(self._build_converter_tab(), "Converter")
        self.tabs.addTab(self._build_merger_tab(), "PDF Merger")
        self.tabs.addTab(self._build_editor_tab(), "PDF Editor")
        main_layout.addWidget(self.tabs, 1)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame(objectName="Sidebar")
        sidebar.setFixedWidth(230)
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(9)
        title = QLabel("DocMaster", objectName="AppTitle")
        subtitle = QLabel("SLAI document workspace", objectName="Muted")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        home_btn = QPushButton("Home")
        home_btn.clicked.connect(self._return_home)
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(self._return_home)
        runtime_btn = QPushButton("Initialize SLAI Runtime", objectName="Primary")
        runtime_btn.clicked.connect(self._initialize_runtime)
        self.runtime_btn = runtime_btn

        for widget in (home_btn, back_btn, runtime_btn):
            layout.addWidget(widget)

        layout.addSpacing(10)
        for label, index in [("SLAI Assistant", 0), ("Word Counter", 1), ("Converter", 2), ("PDF Merger", 3), ("PDF Editor", 4)]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked=False, tab=index: self.tabs.setCurrentIndex(tab))
            layout.addWidget(btn)

        layout.addStretch(1)
        self.runtime_status = QLabel("Runtime: fallback", objectName="Muted")
        self.runtime_status.setWordWrap(True)
        layout.addWidget(self.runtime_status)
        privacy = QLabel("Privacy: documents are processed in memory and not stored by default.", objectName="Muted")
        privacy.setWordWrap(True)
        layout.addWidget(privacy)
        return sidebar

    def _build_topbar(self) -> QFrame:
        topbar = QFrame(objectName="Topbar")
        layout = QHBoxLayout(topbar)
        brand = QLabel("DocMaster AI Document Assistant", objectName="AppTitle")
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
        for label in ["Words", "Characters", "Sentences", "Readability", "Confidence"]:
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
        self.confidence_bar.setValue(0)
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

    def _build_counter_tab(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        actions = QHBoxLayout()
        select_btn = QPushButton("Select file", objectName="Primary")
        select_btn.clicked.connect(self._run_word_count)
        actions.addWidget(select_btn)
        actions.addStretch(1)
        layout.addLayout(actions)
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
        add_btn = QPushButton("Add PDFs", objectName="Primary")
        add_btn.clicked.connect(self._add_merge_files)
        merge_btn = QPushButton("Merge PDFs")
        merge_btn.clicked.connect(self._merge_selected_pdfs)
        clear_btn = QPushButton("Clear list")
        clear_btn.clicked.connect(self._clear_merge_files)
        for btn in (add_btn, merge_btn, clear_btn):
            actions.addWidget(btn)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.merge_list = QListWidget()
        layout.addWidget(self.merge_list, 1)
        self.merge_output = QLabel("Select at least two PDFs.", objectName="Muted")
        self.merge_output.setWordWrap(True)
        layout.addWidget(self.merge_output)
        return body

    def _build_editor_tab(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        title = QLabel("PDF Editor", objectName="PanelTitle")
        note = QLabel(
            "Prepared for future SLAI-powered PDF editing. Current implementation keeps original documents unchanged and focuses on analysis, suggestions, conversion, and merging.",
            objectName="Muted",
        )
        note.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(note)
        layout.addStretch(1)
        return body

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
        self._set_stat_defaults()
        self.status_label.setText("Ready. Select a document to start.")

    def _toggle_question_input(self, text: str) -> None:
        self.question_input.setVisible(SUPPORTED_AI_TASKS.get(text) == "ask")

    def _run_ai_task(self) -> None:
        if not self.selected_file:
            QMessageBox.information(self, "Document required", "Select a document first.")
            return
        task = SUPPORTED_AI_TASKS[self.action_combo.currentText()]
        question = self.question_input.text().strip()
        if task == "ask" and not question:
            QMessageBox.information(self, "Question required", "Add a question before running document Q&A.")
            return

        start_loading(self.loading_controller, "Running DocMaster AI task…")
        update_loading(self.loading_controller, progress=0.35, message="Extracting document text…")
        self.status_label.setText("Processing document...")
        try:
            update_loading(self.loading_controller, progress=0.65, message="Calling SLAI adapter / fallback engine…")
            payload = self.ai_service.run_path(task=task, path=self.selected_file, question=question)
            self.last_ai_payload = payload
            self.ai_response.setPlainText(format_payload(payload))
            self.ai_json.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
            self._apply_payload_stats(payload)
            if payload.get("status") == "success":
                self.status_label.setText(f"Completed: {task}")
                complete_loading(self.loading_controller, "DocMaster task completed")
            else:
                self.status_label.setText("Task completed with errors")
                complete_loading(self.loading_controller, "DocMaster task failed")
        except Exception as exc:  # noqa: BLE001
            complete_loading(self.loading_controller, "DocMaster task failed")
            QMessageBox.critical(self, "DocMaster error", f"Unable to process document.\n\n{type(exc).__name__}: {exc}")

    def _run_word_count(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Select document", str(Path.home()), "Documents (*.pdf *.docx *.txt *.html *.htm *.xml *.odt)")
        if not path:
            return
        try:
            stats = self.file_service.count_file(path)
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
            self.stat_labels["Sentences"].setText(str(stats.sentence_count))
            self.stat_labels["Readability"].setText(str(stats.readability_score))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Count failed", str(exc))

    def _run_converter(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Select document", str(Path.home()), "Documents (*.pdf *.docx *.txt *.html *.htm *.xml *.odt)")
        if not path:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder", str(Path.home()))
        if not out_dir:
            return
        try:
            output = self.file_service.convert_file(path, self.convert_format.currentText(), out_dir)
            self.converter_output.setPlainText(f"Conversion successful.\n\nOutput:\n{output}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Conversion failed", str(exc))

    def _add_merge_files(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(self, "Select PDFs", str(Path.home()), "PDF files (*.pdf)")
        for path in paths:
            p = Path(path)
            if p not in self.merge_files:
                self.merge_files.append(p)
        self._refresh_merge_list()

    def _clear_merge_files(self) -> None:
        self.merge_files.clear()
        self._refresh_merge_list()

    def _refresh_merge_list(self) -> None:
        if not hasattr(self, "merge_list"):
            return
        self.merge_list.clear()
        for path in self.merge_files:
            item = QListWidgetItem(str(path))
            self.merge_list.addItem(item)
        if hasattr(self, "merge_output"):
            self.merge_output.setText(f"{len(self.merge_files)} PDF file(s) selected.")

    def _merge_selected_pdfs(self) -> None:
        if len(self.merge_files) < 2:
            QMessageBox.information(self, "More PDFs required", "Select at least two PDF files.")
            return
        output, _filter = QFileDialog.getSaveFileName(self, "Save merged PDF", str(Path.home() / "merged.pdf"), "PDF files (*.pdf)")
        if not output:
            return
        try:
            merged = self.file_service.merge_pdfs(self.merge_files, output)
            self.merge_output.setText(f"Merge successful. Output: {merged}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Merge failed", str(exc))

    def _initialize_runtime(self) -> None:
        start_loading(self.loading_controller, "Initializing SLAI runtime…")
        update_loading(self.loading_controller, progress=0.4, message="Loading safe agent components…")
        result = self.ai_service.initialize_runtime()
        self._refresh_runtime_status()
        complete_loading(self.loading_controller, "Runtime initialization checked")
        if result.get("available"):
            QMessageBox.information(self, "SLAI runtime", "SLAI runtime initialized for DocMaster.")
        else:
            QMessageBox.information(self, "SLAI fallback", result.get("error") or "SLAI runtime unavailable; fallback mode remains active.")

    def _refresh_runtime_status(self) -> None:
        health = self.ai_service.health().get("slai", {})
        mode = health.get("mode", "safe_fallback")
        status = health.get("status", "unknown")
        if hasattr(self, "runtime_status"):
            self.runtime_status.setText(f"Runtime: {mode} · {status}")

    def _apply_payload_stats(self, payload: Dict[str, Any]) -> None:
        metadata = payload.get("metadata", {})
        result = payload.get("result", {})
        self.stat_labels["Words"].setText(str(metadata.get("word_count", 0)))
        self.stat_labels["Characters"].setText(str(metadata.get("char_count", 0)))
        self.stat_labels["Sentences"].setText(str(metadata.get("sentence_count", "--")))
        self.stat_labels["Readability"].setText(str(metadata.get("readability_score", "--")))
        confidence = int(float(result.get("confidence", 0.0) or 0.0) * 100)
        self.stat_labels["Confidence"].setText(f"{confidence}%")
        self.confidence_bar.setValue(confidence)

    def _set_stat_defaults(self) -> None:
        for label in self.stat_labels.values():
            label.setText("--")

    def _return_home(self) -> None:
        if self.receivers(self.home_requested) > 0:
            self.home_requested.emit()
            return
        try:
            from main import HubWindow  # pyright: ignore[reportMissingImports]

            self._home_window = HubWindow()
            self._home_window.showMaximized()
            self.close()
        except Exception:
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
