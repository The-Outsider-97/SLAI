"""SignalSentry PyQt5 app.

Composition-only module: styles and business logic are sourced from dedicated layers.
"""

from __future__ import annotations

import subprocess
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from component.styles.main_style import metric_card_style
from component.styles.signal_sentry_style import SIGNAL_SENTRY_STYLE
from component.utils.loading_overlay import LoadingOverlay
from component.utils.main_utils import fmt_pct, utc_timestamp_label
from src.functions.loading import create_loading_controller, start_loading, update_loading, complete_loading
from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.collaborative_agent import CollaborativeAgent
from src.agents.planning.planning_types import Task, TaskType

_BACKEND_IMPORT_ERROR = None

try:
    from component.utils.signal_sentry_utils import (
        AgentBinding,
        FeedbackEvent,
        MonitoredEntity,
        OnboardingResult,
        PipelineResult,
        Signal,
        SourceEntry,
        agent_status_for_tab,
        feedback_usefulness_ratio,
        get_signal_sentry_runtime,
        run_agentic_monitoring,
        run_onboarding_workflow,
        seed_signals,
        seed_sources,
        seed_watchlist,
        summarize_digest,
        watchlist_summary,
    )
except Exception as exc:
    _BACKEND_IMPORT_ERROR = exc

    @dataclass
    class Signal:
        id: int = 0
        title: str = "Backend unavailable"
        severity: str = "low"
        confidence: float = 0.0
        priority_score: float = 0.0
        signal_type: str = "system"
        source_url: str = "-"
        evidence_note: str = "SignalSentry backend failed to import."
        suggested_action: str = "Fix backend dependencies and rerun."

    @dataclass
    class SourceEntry:
        name: str
        url: str
        source_type: str
        schedule: str
        last_status: str

    @dataclass
    class MonitoredEntity:
        name: str
        entity_type: str = "unknown"
        focus: str = ""

    @dataclass
    class FeedbackEvent:
        signal_id: int
        verdict: str

    @dataclass
    class OnboardingResult:
        taxonomy: dict = field(default_factory=lambda: {"entities": []})

    @dataclass
    class AgentBinding:
        available: bool = False
        implementation: str = ""
        error: str = "backend import failed"

    @dataclass
    class PipelineResult:
        digest_signals: list = field(default_factory=list)
        archived_signals: list = field(default_factory=list)
        escalations: list = field(default_factory=list)
        workflow_trace: list = field(default_factory=list)
        agent_bindings: dict = field(default_factory=dict)

    def agent_status_for_tab(_tab_name: str) -> dict:
        return {"Backend": f"offline · {_BACKEND_IMPORT_ERROR}"}

    def feedback_usefulness_ratio(_events) -> float:
        return 0.0

    def run_agentic_monitoring(*_args, **_kwargs):
        raise RuntimeError(f"SignalSentry backend import failed: {_BACKEND_IMPORT_ERROR}")

    def run_onboarding_workflow(*_args, **_kwargs):
        return OnboardingResult()

    def seed_signals():
        return []

    def seed_sources():
        return []

    def seed_watchlist():
        return {"companies": [], "products": [], "themes": []}

    def summarize_digest(_signals) -> str:
        return f"Backend unavailable: {_BACKEND_IMPORT_ERROR}"

    def watchlist_summary() -> str:
        return "0 entities"

class SignalSentryWindow(QMainWindow):
    """Desktop implementation of the SignalSentry intelligence workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SignalSentry · SLAI-driven")
        self.resize(1400, 900)
        self._bootstrap_agents()

        self.signals: List[Signal] = seed_signals()
        self.sources: List[SourceEntry] = seed_sources()
        self.watchlist: Dict[str, List[MonitoredEntity]] = seed_watchlist()
        self.feedback_events: List[FeedbackEvent] = []
        self.last_pipeline_result: PipelineResult | None = None
        self.onboarding_result: OnboardingResult = run_onboarding_workflow(
            entities=[entity for group in self.watchlist.values() for entity in group],
            sources=self.sources,
        )

        self.setStyleSheet(SIGNAL_SENTRY_STYLE)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(12)

        outer.addWidget(self._build_header())
        outer.addLayout(self._build_metrics_row())

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        outer.addWidget(self.tabs, 1)

        self.agent_status = QListWidget()
        self.agent_status.setMinimumWidth(240)
        self.watchlist_widgets: Dict[str, QListWidget] = {}

        self.dashboard_tab = self._build_dashboard_tab()
        self.digest_tab = self._build_digest_tab()
        self.sources_tab = self._build_sources_tab()
        self.feedback_tab = self._build_feedback_tab()

        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.digest_tab, "Daily Digest")
        self.tabs.addTab(self.sources_tab, "Source Manager")
        self.tabs.addTab(self.feedback_tab, "Feedback & Tuning")

        self._refresh_all_views()
        self.loading_overlay = LoadingOverlay(self.centralWidget())
        self.loading_overlay.sync_geometry()
        self.loading_controller = create_loading_controller()
        self.loading_controller.on_update = self.loading_overlay.on_loader_update

        self._refresh_all_views()

        # if _BACKEND_IMPORT_ERROR is not None:
        #     QMessageBox.warning(
        #         self,
        #         "SignalSentry backend unavailable",
        #         f"The UI opened, but the backend could not load:\n\n{_BACKEND_IMPORT_ERROR}"
        #     )

    def _bootstrap_agents(self) -> None:
        """
        Keep explicit agent wiring at the component level while syncing with shared utils runtime.
        """
        self._planning_task_registered = False
        self._planning_enabled = True
        self._runtime = None
        if _BACKEND_IMPORT_ERROR is not None:
            return

        try:
            self._runtime = get_signal_sentry_runtime()
            self.shared_memory = self._runtime.shared_memory
            self.factory = self._runtime.factory
            self.collab = self._runtime.collab
            self.knowledge_agent = self._runtime.knowledge_agent
            self.planning_agent = self._runtime.planning_agent
            self.execution_agent = self._runtime.execution_agent
            self.learning_agent = self._runtime.learning_agent
        except Exception:
            self.shared_memory = SharedMemory()
            self.factory = AgentFactory()
            self.collab = CollaborativeAgent(shared_memory=self.shared_memory, agent_factory=self.factory)
            self.knowledge_agent = self.factory.create("knowledge", self.shared_memory)
            self.planning_agent = self.factory.create("planning", self.shared_memory)
            self.execution_agent = self.factory.create("execution", self.shared_memory)
            self.learning_agent = self.factory.create("learning", self.shared_memory)

    def _ensure_planning_task_registered(self) -> None:
        if self._planning_task_registered or not self._planning_enabled or _BACKEND_IMPORT_ERROR is not None:
            return

        task = Task(
            id="signal_sentry_daily_monitoring",
            name="SignalSentry daily monitoring",
            task_description="Plan crawl/parse/diff/score/summarize workflow for monitored sources.",
            task_type=TaskType.ABSTRACT,
            owner="SignalSentry",
            tags=["signal_sentry", "monitoring"],
        )
        if hasattr(self.planning_agent, "register_task"):
            self.planning_agent.register_task(task)
        self._planning_task_registered = True

    def _current_pipeline(self) -> PipelineResult:
        """Return cached pipeline result or run a fresh monitoring pass."""
        return self.last_pipeline_result or run_agentic_monitoring(self.signals, alignment_enabled=True)

    def _build_header(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        brand = QLabel("SignalSentry")
        brand.setProperty("class", "brand")
        subtitle = QLabel("Autonomous market and competitor intelligence")
        subtitle.setProperty("class", "subtitle")
        self.run_state_label = QLabel("Last run: --")
        self.run_state_label.setProperty("class", "subtitle")
        self.watchlist_state_label = QLabel("Seeded watchlist: --")
        self.watchlist_state_label.setProperty("class", "subtitle")
        left.addWidget(brand)
        left.addWidget(subtitle)
        left.addWidget(self.run_state_label)
        left.addWidget(self.watchlist_state_label)

        button_wrap = QHBoxLayout()
        home_btn = QPushButton("Home")
        home_btn.clicked.connect(self._return_home)
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(self._return_home)
        export_btn = QPushButton("Export weekly brief")
        export_btn.clicked.connect(self._mock_export)
        self.scan_btn = QPushButton("Run scan now")
        self.scan_btn.setProperty("class", "primary")
        self.scan_btn.clicked.connect(self._run_scan)
        button_wrap.addWidget(home_btn)
        button_wrap.addWidget(back_btn)
        button_wrap.addWidget(export_btn)
        button_wrap.addWidget(self.scan_btn)

        layout.addLayout(left, 1)
        layout.addLayout(button_wrap)
        return container

    def _build_metrics_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self.metric_labels: Dict[str, QLabel] = {}
        for key in ["Critical Alerts (24h)", "New Signals", "Precision@Top10", "Source Health"]:
            card = QFrame()
            card.setStyleSheet(metric_card_style())
            box = QVBoxLayout(card)
            cap = QLabel(key)
            cap.setProperty("class", "subtitle")
            val = QLabel("--")
            val.setProperty("class", "title")
            box.addWidget(cap)
            box.addWidget(val)
            row.addWidget(card)
            self.metric_labels[key] = val
        return row

    def _build_dashboard_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        splitter = QSplitter(Qt.Horizontal)
        left = QFrame()
        left.setObjectName("panel")
        left_lay = QVBoxLayout(left)
        left_lay.addWidget(self._section_title("Priority alerts & intelligence feed"))
        self.signal_list = QListWidget()
        left_lay.addWidget(self.signal_list)

        right = QFrame()
        right.setObjectName("panel")
        right_lay = QVBoxLayout(right)
        right_lay.addWidget(self._section_title("Source reliability"))

        self.crawl_progress = QProgressBar()
        self.crawl_progress.setValue(96)
        self.parser_progress = QProgressBar()
        self.parser_progress.setProperty("class", "warn")
        self.parser_progress.setValue(88)
        right_lay.addWidget(QLabel("Crawler success rate"))
        right_lay.addWidget(self.crawl_progress)
        right_lay.addWidget(QLabel("Parser confidence"))
        right_lay.addWidget(self.parser_progress)

        right_lay.addWidget(self._section_title("Unresolved handler escalations"))
        self.unresolved_list = QListWidget()
        right_lay.addWidget(self.unresolved_list)

        right_lay.addWidget(self._section_title("Agent status by workflow"))
        right_lay.addWidget(self.agent_status)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([900, 420])

        layout.addWidget(splitter)
        return tab

    def _build_digest_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_lay = QVBoxLayout(panel)
        panel_lay.addWidget(self._section_title("Daily intelligence brief"))

        self.digest_summary = QTextEdit()
        self.digest_summary.setReadOnly(True)
        panel_lay.addWidget(self.digest_summary)

        self.digest_list = QListWidget()
        panel_lay.addWidget(self.digest_list)

        self.digest_agent_note = QLabel("Language Agent + Safety Agent gate")
        self.digest_agent_note.setProperty("class", "badge")
        panel_lay.addWidget(self.digest_agent_note, alignment=Qt.AlignLeft)

        layout.addWidget(panel)
        return tab

    def _build_sources_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_lay = QVBoxLayout(panel)
        panel_lay.addWidget(self._section_title("Monitored sources & crawl policy"))

        panel_lay.addWidget(self._section_title("Seeded monitored watchlist"))
        watchlist_row = QHBoxLayout()
        watchlist_row.setSpacing(10)
        for title, key in [("Companies", "companies"), ("Products", "products"), ("Themes", "themes")]:
            box = QFrame()
            box.setObjectName("panel")
            box_lay = QVBoxLayout(box)
            box_lay.setContentsMargins(10, 10, 10, 10)
            header = QLabel(title)
            header.setProperty("class", "subtitle")
            box_lay.addWidget(header)
            list_widget = QListWidget()
            list_widget.setMinimumHeight(180)
            box_lay.addWidget(list_widget)
            watchlist_row.addWidget(box)
            self.watchlist_widgets[key] = list_widget
        panel_lay.addLayout(watchlist_row)

        self.watchlist_count_label = QLabel("Seeded monitoring profile: --")
        self.watchlist_count_label.setProperty("class", "badge")
        panel_lay.addWidget(self.watchlist_count_label, alignment=Qt.AlignLeft)

        self.source_table = QTableWidget(0, 4)
        self.source_table.setHorizontalHeaderLabels(["Source", "Type", "Schedule", "Last status"])
        self.source_table.horizontalHeader().setStretchLastSection(True)
        panel_lay.addWidget(self.source_table)

        source_input_row = QHBoxLayout()
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("https://competitor.com/changelog")
        add_source_btn = QPushButton("Add source")
        add_source_btn.clicked.connect(self._add_source)
        source_input_row.addWidget(self.source_input)
        source_input_row.addWidget(add_source_btn)
        panel_lay.addLayout(source_input_row)

        planning_note = QLabel("Planning Agent + Browser Agent use retry/backoff policy for failures")
        planning_note.setProperty("class", "badge")
        panel_lay.addWidget(planning_note, alignment=Qt.AlignLeft)

        layout.addWidget(panel)
        return tab

    def _build_feedback_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_lay = QVBoxLayout(panel)
        panel_lay.addWidget(self._section_title("Signal usefulness & threshold tuning"))

        self.feedback_list = QListWidget()
        panel_lay.addWidget(self.feedback_list)

        controls = QHBoxLayout()
        useful_btn = QPushButton("Mark selected as useful")
        useful_btn.clicked.connect(lambda: self._record_feedback("useful"))
        low_btn = QPushButton("Mark selected as low value")
        low_btn.clicked.connect(lambda: self._record_feedback("low_value"))
        controls.addWidget(useful_btn)
        controls.addWidget(low_btn)
        panel_lay.addLayout(controls)

        self.feedback_ratio_label = QLabel("Feedback ratio: --")
        self.feedback_ratio_label.setProperty("class", "badge")
        panel_lay.addWidget(self.feedback_ratio_label, alignment=Qt.AlignLeft)

        policy_note = QLabel(
            "Evaluation + Reasoning reprioritize signals; Handler routes critical disputed alerts for human review."
        )
        policy_note.setProperty("class", "subtitle")
        panel_lay.addWidget(policy_note)

        layout.addWidget(panel)
        return tab

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("class", "sectionTitle")
        return label

    def _refresh_all_views(self) -> None:
        self._refresh_metrics()
        self._refresh_dashboard()
        self._refresh_digest()
        self._refresh_watchlist()
        self._refresh_sources()
        self._refresh_feedback()
        self._on_tab_changed(self.tabs.currentIndex())

    def _watchlist_total_count(self) -> int:
        return sum(len(items) for items in self.watchlist.values())

    def _refresh_metrics(self) -> None:
        critical = sum(1 for sig in self.signals if sig.severity == "critical")
        self.metric_labels["Critical Alerts (24h)"].setText(str(critical))
        self.metric_labels["New Signals"].setText(str(len(self.signals)))
        precision = sum(sig.confidence for sig in self.signals[:10]) / max(1, min(10, len(self.signals)))
        self.metric_labels["Precision@Top10"].setText(f"{precision:.2f}")
        health = (self.crawl_progress.value() / 100.0 + self.parser_progress.value() / 100.0) / 2
        self.metric_labels["Source Health"].setText(fmt_pct(health))
        self.watchlist_state_label.setText(f"Seeded watchlist: {self._watchlist_total_count()} monitored targets")

    
    def _refresh_dashboard(self) -> None:
        pipeline = self.last_pipeline_result
        if pipeline is None:
            self.signal_list.clear()
            self.unresolved_list.clear()
            return
    
        self.signal_list.clear()
        for signal in pipeline.digest_signals:
            row = QListWidgetItem(
                f"[{signal.severity.upper()} · {signal.priority_score:.2f}] {signal.title}\n"
                f"{signal.signal_type} | conf {signal.confidence:.2f} | source {signal.source_url}\n"
                f"Reasoning: {signal.evidence_note}\nAction: {signal.suggested_action}"
            )
            self.signal_list.addItem(row)
    
        self.unresolved_list.clear()
        for escalation in pipeline.escalations:
            self.unresolved_list.addItem(
                f"{escalation.title} | owner: {escalation.owner} | Handler status: {escalation.status}"
            )

    
    def _refresh_digest(self) -> None:
        pipeline = self.last_pipeline_result
        if pipeline is None:
            self.digest_summary.setText(
                "SignalSentry is ready.\n\nPress 'Run scan now' to initialize the SLAI backend."
            )
            self.digest_list.clear()
            return
    
        summary = summarize_digest(pipeline.digest_signals)
        self.digest_summary.setText(
            f"{summary}\n\n"
            f"Coverage: {watchlist_summary()}\n"
            f"Workflow: {' | '.join(pipeline.workflow_trace)}"
        )
        self.digest_list.clear()
        for signal in pipeline.digest_signals:
            self.digest_list.addItem(
                f"{signal.title} (priority {signal.priority_score:.2f}) - evidence: {signal.source_url}"
            )

    def _refresh_watchlist(self) -> None:
        for key, widget in self.watchlist_widgets.items():
            widget.clear()
            for entity in self.watchlist.get(key, []):
                detail = entity.focus if entity.focus else entity.entity_type
                widget.addItem(f"{entity.name}\n{detail}")
        self.watchlist_count_label.setText(
            f"Seeded monitoring profile: {watchlist_summary()} | "
            f"Taxonomy entities: {len(self.onboarding_result.taxonomy.get('entities', []))}"
        )

    def _refresh_sources(self) -> None:
        self.source_table.setRowCount(len(self.sources))
        for row_idx, src in enumerate(self.sources):
            self.source_table.setItem(row_idx, 0, QTableWidgetItem(f"{src.name}\n{src.url}"))
            self.source_table.setItem(row_idx, 1, QTableWidgetItem(src.source_type))
            self.source_table.setItem(row_idx, 2, QTableWidgetItem(src.schedule))
            self.source_table.setItem(row_idx, 3, QTableWidgetItem(src.last_status))

    def _refresh_feedback(self) -> None:
        self.feedback_list.clear()
        for signal in self.signals:
            self.feedback_list.addItem(f"#{signal.id} {signal.title} | priority {signal.priority_score:.2f}")
        ratio = feedback_usefulness_ratio(self.feedback_events)
        self.feedback_ratio_label.setText(f"Useful signal ratio: {fmt_pct(ratio)}")

    def _on_tab_changed(self, index: int) -> None:
        tab_name = self.tabs.tabText(index)
        status_map = agent_status_for_tab(tab_name)
        pipeline = self.last_pipeline_result
    
        if pipeline is not None:
            for agent_name, binding in pipeline.agent_bindings.items():
                if binding.available:
                    status_map[agent_name] = f"live · {binding.implementation}"
                else:
                    status_map[agent_name] = f"error · {binding.error}"
    
        self.agent_status.clear()
        for agent, state in status_map.items():
            self.agent_status.addItem(f"{agent} Agent: {state}")

    def _run_scan(self) -> None:
        start_loading(self.loading_controller, "Running SignalSentry scan…")
        update_loading(self.loading_controller, progress=0.2, message="Preparing workflow agents…")
        self._ensure_planning_task_registered()
        update_loading(self.loading_controller, progress=0.45, message="Collecting signals…")
        pipeline = self._ensure_pipeline_result(force=True)
        if pipeline is None:
            complete_loading(self.loading_controller, "Scan failed")
            QMessageBox.critical(
                self,
                "Scan failed",
                "SignalSentry could not bind required live SLAI agents. Check runtime dependencies/logs.",
            )
            return
        self.last_pipeline_result = pipeline
        update_loading(self.loading_controller, progress=0.85, message="Refreshing dashboard…")
        if self.last_pipeline_result.digest_signals:
            self.signals = self.last_pipeline_result.digest_signals + self.last_pipeline_result.archived_signals
        self.run_state_label.setText(f"Last run: {utc_timestamp_label()}")
        self._refresh_all_views()
        complete_loading(self.loading_controller, "Scan complete")
        QMessageBox.information(
            self,
            "Scan complete",
            (
                "Collaborative agent published refreshed digest after Safety/Alignment checks.\n"
                f"Signals in digest: {len(self.last_pipeline_result.digest_signals)}\n"
                f"Critical escalations: {len(self.last_pipeline_result.escalations)}"
            ),
        )

    def _mock_export(self) -> None:
        QMessageBox.information(
            self,
            "Export complete",
            "Language agent rendered a weekly brief package for email/API delivery.",
        )

    def _add_source(self) -> None:
        url = self.source_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Missing URL", "Provide a source URL first.")
            return
        self.sources.append(SourceEntry("Custom source", url, "Manual", "Daily", "pending"))
        self.onboarding_result = run_onboarding_workflow(
            entities=[entity for group in self.watchlist.values() for entity in group],
            sources=self.sources,
        )
        self.source_input.clear()
        self._refresh_sources()

    def _record_feedback(self, verdict: str) -> None:
        current_item = self.feedback_list.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "No selection", "Select a signal row to provide feedback.")
            return
        try:
            signal_id = int(current_item.text().split()[0].replace("#", ""))
        except (ValueError, IndexError):
            QMessageBox.warning(self, "Parse error", "Unable to parse selected signal.")
            return

        self.feedback_events.append(FeedbackEvent(signal_id=signal_id, verdict=verdict))
        self._refresh_feedback()

    def _return_home(self) -> None:
        main_path = Path(__file__).resolve().parents[1] / "main.py"
        subprocess.Popen([sys.executable, str(main_path)])
        self.close()

    def _ensure_pipeline_result(self, force: bool = False) -> PipelineResult | None:
        if self.last_pipeline_result is not None and not force:
            return self.last_pipeline_result
        try:
            self.last_pipeline_result = run_agentic_monitoring(self.signals, alignment_enabled=True)
            return self.last_pipeline_result
        except Exception as exc:  # noqa: BLE001
            self.last_pipeline_result = None
            self.run_state_label.setText(f"Last run failed: {utc_timestamp_label()}")
            self.digest_summary.setText(f"Pipeline error: {exc}")
            return None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "loading_overlay"):
            self.loading_overlay.sync_geometry()


def launch_signal_sentry() -> None:
    existing_app = QApplication.instance()
    app = existing_app or QApplication(sys.argv)
    win = SignalSentryWindow()
    win.show()
    if existing_app is None:
        app.exec_()


if __name__ == "__main__":
    launch_signal_sentry()
