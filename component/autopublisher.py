"""ContentOps Autopublisher operational workspace (PyQt5)."""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from component.styles.autopublisher_style import AUTOPUBLISHER_STYLE, sanitize_qss
from logs.logger import get_logger

logger = get_logger("ContentOps Autopublisher")


@dataclass
class TopicCandidate:
    title: str
    rationale: str
    status: str = "Backlog"
    evidence: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentBrief:
    topic: str
    text: str = ""
    research: str = ""


@dataclass
class Draft:
    topic: str
    text: str = ""
    variants: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    coverage: float = 0.0
    coherence: float = 0.0
    evidence: float = 0.0
    readability: float = 0.0
    compliance: float = 0.0
    final_score: float = 0.0
    verdict: str = "needs human review"
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PublishPackage:
    topic: str
    package_text: str
    exported_at: str


@dataclass
class Workspace:
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    topics: List[TopicCandidate] = field(default_factory=list)
    briefs: Dict[str, ContentBrief] = field(default_factory=dict)
    drafts: Dict[str, Draft] = field(default_factory=dict)
    quality_reports: Dict[str, QualityReport] = field(default_factory=dict)
    packages: Dict[str, PublishPackage] = field(default_factory=dict)


class AutopublisherWindow(QMainWindow):
    BOARD_COLUMNS = ["Backlog", "In Progress", "Ready", "Approved"]
    CORE_AGENT_TYPES = ["planning", "browser", "knowledge", "reasoning", "language", "evaluation", "safety", "learning"]
    OPTIONAL_AGENT_TYPES = ["alignment", "adaptive"]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ContentOps Autopublisher")
        self.resize(1500, 900)

        self.workspace = Workspace()
        self.selected_topic: Optional[str] = None

        self.shared_memory = None
        self.factory = None
        self.collab = None
        self.agents: Dict[str, Any] = {}
        self.runtime_initialized = False
        self.runtime_init_error: Optional[str] = None
        self.agent_runtime_status: str = "not_initialized"
        self.agent_runtime_error: Optional[str] = None
        self.agent_statuses: Dict[str, str] = {}
        self.agent_runtime_summary: Dict[str, Any] = {}
        self.torch_probe_details: str = "not_tested"
        self.runtime_components: Dict[str, str] = {
            "ui": "ready",
            "lightweight_runtime": "not_initialized",
            "runtime_core": "not_initialized",
            "registry": "not_initialized",
            "optional_heavy_agents": "not_attempted",
            "shared_memory_module": "not_loaded",
            "collaborative_agent_module": "not_loaded",
            "agent_factory_module": "not_loaded",
            "torch_subsystem": "untested",
            "core_agents": "not_evaluated",
            "optional_agents": "not_evaluated",
        }

        self._build_ui()
        self._refresh_board()
        self._refresh_agent_fleet()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        self.setStyleSheet(sanitize_qss(AUTOPUBLISHER_STYLE))
        outer = QHBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(220)
        side_layout = QVBoxLayout(sidebar)
        side_layout.addWidget(QLabel("ContentOps", objectName="AppTitle"))
        self.btn_runtime = QPushButton("Initialize Agent Runtime", objectName="Primary")
        self.btn_runtime.clicked.connect(self.initialize_runtime)
        side_layout.addWidget(self.btn_runtime)
        for n in ["Dashboard", "Calendar", "Pipeline", "Diagnostics", "Exports"]:
            side_layout.addWidget(QPushButton(n))
        side_layout.addStretch(1)
        outer.addWidget(sidebar)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setSpacing(8)
        outer.addWidget(main, 1)

        topbar = QFrame(objectName="Topbar")
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.addWidget(QLabel("ContentOps Autopublisher", objectName="AppTitle"))
        topbar_layout.addStretch(1)
        self.status_label = QLabel("UI ready. Agent runtime not loaded.")
        self.status_label.setObjectName("Muted")
        topbar_layout.addWidget(self.status_label)
        main_layout.addWidget(topbar)

        stats_row = QFrame(objectName="StatsRow")
        stats_layout = QHBoxLayout(stats_row)
        self.stat_labels: Dict[str, QLabel] = {}
        for key in ["Topics", "In Progress", "Ready", "Approved", "Avg Quality"]:
            card = QFrame(objectName="StatCard")
            c = QVBoxLayout(card)
            c.addWidget(QLabel(key, objectName="Muted"))
            v = QLabel("0")
            v.setObjectName("PanelTitle")
            c.addWidget(v)
            self.stat_labels[key] = v
            stats_layout.addWidget(card)
        main_layout.addWidget(stats_row)

        actions = QHBoxLayout()
        self.btn_plan = QPushButton("Generate Weekly Plan", objectName="Primary")
        self.btn_brief = QPushButton("Build Brief")
        self.btn_draft = QPushButton("Generate Draft")
        self.btn_qa = QPushButton("Run QA")
        self.btn_repurpose = QPushButton("Repurpose")
        self.btn_export = QPushButton("Export Package")
        for b in [self.btn_plan, self.btn_brief, self.btn_draft, self.btn_qa, self.btn_repurpose, self.btn_export]:
            actions.addWidget(b)
        actions.addStretch(1)
        main_layout.addLayout(actions)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)

        board = QWidget()
        board_layout = QHBoxLayout(board)
        self.board_lists: Dict[str, QListWidget] = {}
        for col in self.BOARD_COLUMNS:
            frame = QFrame(objectName="StageColumn")
            fl = QVBoxLayout(frame)
            fl.addWidget(QLabel(col, objectName="ColumnTitle"))
            lw = QListWidget()
            lw.itemSelectionChanged.connect(self._on_board_selection)
            self.board_lists[col] = lw
            fl.addWidget(lw)
            board_layout.addWidget(frame)
        splitter.addWidget(board)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_body = QWidget()
        right_layout = QVBoxLayout(right_body)

        self.brief_panel = self._make_panel("Brief & Research")
        self.draft_panel = self._make_panel("Draft Workspace")
        self.qa_panel = self._make_panel("Quality Diagnostics")
        self.policy_panel = self._make_panel("Brand Policy Center")
        self.agent_fleet_panel = self._make_panel("Agent Fleet / Status")

        self.brief_text = QPlainTextEdit(); self.brief_text.setReadOnly(True)
        self.draft_text = QPlainTextEdit(); self.draft_text.setReadOnly(True)
        self.qa_text = QPlainTextEdit(); self.qa_text.setReadOnly(True)
        self.policy_text = QPlainTextEdit(); self.policy_text.setReadOnly(True)
        self.fleet_text = QPlainTextEdit(); self.fleet_text.setReadOnly(True)
        self.qa_score = QProgressBar(); self.qa_score.setRange(0, 100)

        self.brief_panel.layout().addWidget(self.brief_text)
        self.draft_panel.layout().addWidget(self.draft_text)
        self.qa_panel.layout().addWidget(self.qa_score)
        self.qa_panel.layout().addWidget(self.qa_text)
        self.policy_panel.layout().addWidget(self.policy_text)
        self.agent_fleet_panel.layout().addWidget(self.fleet_text)

        for p in [self.brief_panel, self.draft_panel, self.qa_panel, self.policy_panel, self.agent_fleet_panel]:
            right_layout.addWidget(p)
        right_layout.addStretch(1)

        right_scroll.setWidget(right_body)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        self.btn_plan.clicked.connect(self.generate_weekly_plan)
        self.btn_brief.clicked.connect(self.build_brief)
        self.btn_draft.clicked.connect(self.generate_draft)
        self.btn_qa.clicked.connect(self.run_qa)
        self.btn_repurpose.clicked.connect(self.repurpose)
        self.btn_export.clicked.connect(self.export_package)

    def _make_panel(self, title: str) -> QFrame:
        panel = QFrame(objectName="DetailPanel")
        v = QVBoxLayout(panel)
        v.addWidget(QLabel(title, objectName="PanelTitle"))
        return panel

    def _trace_import(self, module_path: str):
        logger.info("Autopublisher import stage: %s", module_path)
        return importlib.import_module(module_path)

    def _probe_torch_subsystem(self) -> None:
        try:
            torch_module = importlib.import_module("torch")
            version = getattr(torch_module, "__version__", "unknown")
            cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
            cuda_text = cuda_version if cuda_version else "cpu"
            self.runtime_components["torch_subsystem"] = "available"
            self.torch_probe_details = f"import_ok version={version} cuda={cuda_text}"
        except Exception as torch_exc:
            self.runtime_components["torch_subsystem"] = "unavailable"
            self.torch_probe_details = f"{type(torch_exc).__name__}: {torch_exc}"
            logger.warning("Torch subsystem probe failed: %s", self.torch_probe_details)

    def initialize_runtime(self) -> bool:
        if self.runtime_initialized:
            return True

        self.status_label.setText("Initializing lightweight agent runtime...")
        self.runtime_components["lightweight_runtime"] = "initializing"
        try:
            shared_memory_mod = self._trace_import("src.agents.collaborative.shared_memory")
            self.runtime_components["shared_memory_module"] = "loaded"
            collab_mod = self._trace_import("src.agents.collaborative_agent")
            self.runtime_components["collaborative_agent_module"] = "loaded"
            factory_mod = self._trace_import("src.agents.agent_factory")
            self.runtime_components["agent_factory_module"] = "loaded"

            SharedMemory = getattr(shared_memory_mod, "SharedMemory")
            CollaborativeAgent = getattr(collab_mod, "CollaborativeAgent")
            AgentFactory = getattr(factory_mod, "AgentFactory")

            self.shared_memory = SharedMemory()
            self.factory = AgentFactory()
            self.collab = CollaborativeAgent(shared_memory=self.shared_memory, agent_factory=self.factory)
            self.runtime_components["lightweight_runtime"] = "initialized"
            self.runtime_components["runtime_core"] = "initialized"
            registered = self.factory.get_registered_agent_types()
            self.runtime_components["registry"] = f"initialized ({len(registered)} registered)"

            try:
                importlib.import_module("torch")
                self.runtime_components["torch_subsystem"] = "available"
            except Exception as torch_exc:
                self.runtime_components["torch_subsystem"] = f"unavailable ({type(torch_exc).__name__}: {torch_exc})"
                logger.warning("Optional torch subsystem unavailable: %s", torch_exc)

            optional_failures: Dict[str, str] = {}
            unavailable_reasons: Dict[str, str] = {}
            available_count = 0
            requested_agents = [*self.CORE_AGENT_TYPES, *self.OPTIONAL_AGENT_TYPES]
            for name in requested_agents:
                try:
                    self.agents[name] = self.factory.create(name, self.shared_memory)
                    available_count += 1
                except Exception as exc:
                    status = "optional" if name in self.OPTIONAL_AGENT_TYPES else "core"
                    logger.warning("Agent '%s' unavailable during runtime init (%s): %s", name, status, exc)
                    self.agents[name] = None
                    reason = f"{type(exc).__name__}: {exc}"
                    unavailable_reasons[name] = reason
                    if name in self.OPTIONAL_AGENT_TYPES:
                        optional_failures[name] = reason

            core_available = sum(1 for name in self.CORE_AGENT_TYPES if self.agents.get(name) is not None)
            optional_available = sum(1 for name in self.OPTIONAL_AGENT_TYPES if self.agents.get(name) is not None)
            missing_core = [name for name in self.CORE_AGENT_TYPES if self.agents.get(name) is None]
            self.runtime_components["core_agents"] = f"{core_available}/{len(self.CORE_AGENT_TYPES)}"
            self.runtime_components["optional_agents"] = f"{optional_available}/{len(self.OPTIONAL_AGENT_TYPES)}"
            self.runtime_components["optional_heavy_agents"] = "available" if not optional_failures else "degraded"
            self.agent_statuses = {k: ("online" if self.agents.get(k) is not None else f"unavailable: {unavailable_reasons.get(k, 'unknown')}") for k in requested_agents}

            if core_available == 0:
                self.agent_runtime_status = "blocked"
                if str(self.runtime_components.get("torch_subsystem", "")).startswith("unavailable"):
                    self.agent_runtime_error = "No core agents are available; torch-dependent runtime is unavailable."
                else:
                    self.agent_runtime_error = "No core agents are available."
            elif missing_core:
                self.agent_runtime_status = "degraded"
                self.agent_runtime_error = f"Missing core agents: {', '.join(missing_core)}"
            elif optional_available < len(self.OPTIONAL_AGENT_TYPES):
                self.agent_runtime_status = "degraded"
                self.agent_runtime_error = "Optional agents unavailable."
            else:
                self.agent_runtime_status = "fully_available"
                self.agent_runtime_error = None

            self.agent_runtime_summary = {
                "registered_agents": registered,
                "requested_agents": requested_agents,
                "available_agents": available_count,
                "core_available": core_available,
                "core_total": len(self.CORE_AGENT_TYPES),
                "optional_available": optional_available,
                "optional_total": len(self.OPTIONAL_AGENT_TYPES),
                "missing_core": missing_core,
                "unavailable_reasons": unavailable_reasons,
            }
            logger.info(
                "Runtime init summary: status=%s available_agents=%s core=%s/%s optional=%s/%s",
                self.agent_runtime_status,
                available_count,
                core_available,
                len(self.CORE_AGENT_TYPES),
                optional_available,
                len(self.OPTIONAL_AGENT_TYPES),
            )

            self.runtime_initialized = True
            self.runtime_init_error = None
            if self.agent_runtime_status == "fully_available":
                self.status_label.setText("UI ready; agent runtime fully available")
            elif self.agent_runtime_status == "degraded":
                self.status_label.setText("UI ready; agent runtime degraded")
            else:
                self.status_label.setText("UI ready; agent runtime blocked")
            self._refresh_agent_fleet()
            return True
        except Exception as exc:
            self.runtime_initialized = False
            self.runtime_init_error = f"{type(exc).__name__}: {exc}"
            self.runtime_components["lightweight_runtime"] = f"failed ({self.runtime_init_error})"
            logger.error("Runtime init failed: %s", self.runtime_init_error, exc_info=True)
            detail = (
                "Agent runtime failed to initialize.\n\n"
                f"Error: {self.runtime_init_error}\n\n"
                "This commonly indicates a deep binary dependency problem in a heavy ML backend "
                "(for example, torch DLL dependency resolution on Windows)."
            )
            QMessageBox.critical(self, "Autopublisher Runtime Initialization Failed", detail)
            self.status_label.setText("UI ready; lightweight runtime unavailable")
            self._refresh_agent_fleet()
            return False

    def _ensure_runtime(self) -> bool:
        return self.initialize_runtime()

    def _agent_call(self, agent_name: str, payload: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._ensure_runtime():
            return {
                "status": "runtime_unavailable",
                "agent": agent_name,
                "error": self.runtime_init_error,
            }

        agent = self.agents.get(agent_name)
        if agent is None:
            return {"status": "unavailable", "agent": agent_name}

        attempted = []
        for method_name, args in [
            ("perform_task", (payload,) if context is None else (payload, context)),
            ("predict", (payload,)),
            ("execute", (payload,)),
            ("act", (payload, context or {})),
            ("get_action", (payload, context or {})),
        ]:
            if hasattr(agent, method_name):
                attempted.append(method_name)
                try:
                    result = getattr(agent, method_name)(*args)
                    return {"status": "ok", "method": method_name, "result": result}
                except TypeError:
                    continue
                except Exception as exc:
                    return {"status": "error", "method": method_name, "error": str(exc)}

        return {"status": "unsupported", "agent": agent_name, "attempted": attempted}

    def _extract_topics(self, data: Any) -> List[TopicCandidate]:
        found: List[TopicCandidate] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key.lower() in {"title", "topic", "headline", "subject"} and isinstance(value, str) and value.strip():
                        found.append(TopicCandidate(title=value.strip(), rationale="from planning output", data={"source": node}))
                    walk(value)
            elif isinstance(node, list):
                for i in node:
                    walk(i)

        walk(data)
        dedup = {}
        for t in found:
            dedup.setdefault(t.title, t)
        return list(dedup.values())

    def generate_weekly_plan(self) -> None:
        self.status_label.setText("Generating plan...")
        planning = self._agent_call("planning", {"objective": "weekly editorial slate", "workspace": "autopublisher"})

        if planning.get("status") == "ok":
            if self.shared_memory is not None:
                self.shared_memory.set("autopublisher:last_plan", planning["result"])
            topics = self._extract_topics(planning["result"])
            if topics:
                self.workspace.topics = topics
                self.selected_topic = topics[0].title
                self.status_label.setText(f"Plan generated: {len(topics)} topics")
            else:
                self.status_label.setText("Plan returned without structured topics")
        else:
            self.status_label.setText(f"Plan failed: {planning.get('status')}")

        self._refresh_board()
        self._refresh_agent_fleet()
        self._update_detail_panels()

    def _require_selected_topic(self) -> Optional[TopicCandidate]:
        if not self.selected_topic:
            QMessageBox.information(self, "Selection required", "Select a board item first.")
            return None
        for t in self.workspace.topics:
            if t.title == self.selected_topic:
                return t
        return None

    def build_brief(self) -> None:
        topic = self._require_selected_topic()
        if not topic:
            return

        browser = self._agent_call("browser", {"query": topic.title, "mode": "research"}, {"task": "serp_evidence"})
        knowledge = self._agent_call("knowledge", {"topic": topic.title, "task": "brand_context"})
        reasoning = self._agent_call("reasoning", {"topic": topic.title, "task": "angle"})
        language = self._agent_call("language", f"Create a concise content brief for: {topic.title}")

        brief_text = json.dumps({"language": language, "reasoning": reasoning}, indent=2, default=str)
        research_text = json.dumps({"browser": browser, "knowledge": knowledge}, indent=2, default=str)
        self.workspace.briefs[topic.title] = ContentBrief(topic=topic.title, text=brief_text, research=research_text)
        topic.status = "In Progress"
        self.status_label.setText("Brief built")
        self._refresh_board(); self._update_detail_panels(); self._refresh_agent_fleet()

    def generate_draft(self) -> None:
        topic = self._require_selected_topic()
        if not topic:
            return

        brief = self.workspace.briefs.get(topic.title)
        if not brief:
            QMessageBox.information(self, "Brief missing", "Build brief before generating draft.")
            return

        language = self._agent_call("language", f"Draft article for topic: {topic.title}\nBrief:\n{brief.text}")
        draft_text = json.dumps(language, indent=2, default=str)
        self.workspace.drafts[topic.title] = Draft(topic=topic.title, text=draft_text)
        topic.status = "Ready"
        self.status_label.setText("Draft generated")
        self._refresh_board(); self._update_detail_panels(); self._refresh_agent_fleet()

    def run_qa(self) -> None:
        topic = self._require_selected_topic()
        if not topic:
            return

        draft = self.workspace.drafts.get(topic.title)
        if not draft:
            QMessageBox.information(self, "Draft missing", "Generate draft before QA.")
            return

        evaluation = self._agent_call("evaluation", {"draft": draft.text, "topic": topic.title})
        safety = self._agent_call("safety", draft.text, {"task": "content_safety"})
        alignment = self._agent_call("alignment", {"draft": draft.text, "topic": topic.title})
        collab = {"status": "runtime_unavailable"}
        if self.collab is not None:
            collab = self.collab.perform_task({
                "mode": "assess",
                "risk_score": 0.4,
                "task_type": "editorial_qa",
                "source_agent": "autopublisher",
                "context": {"topic": topic.title},
            })

        scores = self._compute_scores(evaluation, safety, alignment)
        final_score = round(sum(scores.values()) / len(scores), 3)
        verdict = "publish-ready" if final_score >= 0.8 else "needs human review"
        if 0.65 <= final_score < 0.8:
            verdict = "auto-revised / needs human review"

        report = QualityReport(
            coverage=scores["coverage"],
            coherence=scores["coherence"],
            evidence=scores["evidence"],
            readability=scores["readability"],
            compliance=scores["compliance"],
            final_score=final_score,
            verdict=verdict,
            raw={"evaluation": evaluation, "safety": safety, "alignment": alignment, "collab": collab},
        )
        self.workspace.quality_reports[topic.title] = report
        if verdict == "publish-ready":
            topic.status = "Approved"
        self.status_label.setText(f"QA complete: {verdict}")
        self._refresh_board(); self._update_detail_panels(); self._refresh_agent_fleet()

    def _compute_scores(self, evaluation: Dict[str, Any], safety: Dict[str, Any], alignment: Dict[str, Any]) -> Dict[str, float]:
        def score_from_payload(payload: Dict[str, Any], fallback: float = 0.6) -> float:
            text = json.dumps(payload, default=str).lower()
            if "error" in text or payload.get("status") in {"error", "unavailable", "runtime_unavailable"}:
                return 0.4
            if "success" in text or payload.get("status") == "ok":
                return fallback
            return 0.5

        compliance = min(1.0, (score_from_payload(safety, 0.75) + score_from_payload(alignment, 0.7)) / 2)
        base = score_from_payload(evaluation, 0.72)
        return {
            "coverage": base,
            "coherence": max(0.0, base - 0.03),
            "evidence": max(0.0, base - 0.06),
            "readability": min(1.0, base + 0.02),
            "compliance": compliance,
        }

    def repurpose(self) -> None:
        topic = self._require_selected_topic()
        if not topic:
            return
        draft = self.workspace.drafts.get(topic.title)
        if not draft:
            QMessageBox.information(self, "Draft missing", "Generate draft before repurposing.")
            return
        repurpose_out = self._agent_call("language", f"Repurpose into social and email variants:\n{draft.text}")
        draft.variants.append(json.dumps(repurpose_out, default=str))
        self.status_label.setText("Repurpose complete")
        self._update_detail_panels(); self._refresh_agent_fleet()

    def export_package(self) -> None:
        topic = self._require_selected_topic()
        if not topic:
            return
        brief = self.workspace.briefs.get(topic.title)
        draft = self.workspace.drafts.get(topic.title)
        qa = self.workspace.quality_reports.get(topic.title)
        shared_memory_keys = []
        if self.shared_memory is not None:
            try:
                shared_memory_keys = list(self.shared_memory.get_all_keys())[:50]
            except Exception:
                shared_memory_keys = []

        package = PublishPackage(
            topic=topic.title,
            package_text=json.dumps({
                "brief": brief,
                "draft": draft,
                "quality": qa,
                "shared_memory_keys": shared_memory_keys,
            }, indent=2, default=lambda x: x.__dict__),
            exported_at=datetime.utcnow().isoformat(),
        )
        self.workspace.packages[topic.title] = package
        if self.shared_memory is not None:
            self.shared_memory.set(f"autopublisher:package:{topic.title}", package.package_text)
        self.status_label.setText(f"Exported package @ {package.exported_at}")
        self._update_detail_panels()

    def _on_board_selection(self) -> None:
        for column, lw in self.board_lists.items():
            selected = lw.selectedItems()
            if selected:
                item = selected[0]
                self.selected_topic = item.data(Qt.UserRole)
                self.status_label.setText(f"Selected: {self.selected_topic} ({column})")
                break
        self._update_detail_panels()

    def _refresh_board(self) -> None:
        grouped = {k: [] for k in self.BOARD_COLUMNS}
        for topic in self.workspace.topics:
            grouped.setdefault(topic.status, []).append(topic)

        for col, lw in self.board_lists.items():
            lw.clear()
            for t in grouped.get(col, []):
                item = QListWidgetItem(f"{t.title}\n{t.rationale}")
                item.setData(Qt.UserRole, t.title)
                lw.addItem(item)
                if self.selected_topic == t.title:
                    item.setSelected(True)

        topics = len(self.workspace.topics)
        self.stat_labels["Topics"].setText(str(topics))
        self.stat_labels["In Progress"].setText(str(len(grouped.get("In Progress", []))))
        self.stat_labels["Ready"].setText(str(len(grouped.get("Ready", []))))
        self.stat_labels["Approved"].setText(str(len(grouped.get("Approved", []))))

        avg_quality = 0.0
        if self.workspace.quality_reports:
            avg_quality = sum(r.final_score for r in self.workspace.quality_reports.values()) / len(self.workspace.quality_reports)
        self.stat_labels["Avg Quality"].setText(f"{avg_quality:.2f}")

    def _update_detail_panels(self) -> None:
        if not self.selected_topic:
            self.brief_text.setPlainText("Select a board item to load details.")
            self.draft_text.setPlainText("")
            self.qa_text.setPlainText("")
            self.policy_text.setPlainText("")
            self.qa_score.setValue(0)
            return

        brief = self.workspace.briefs.get(self.selected_topic)
        draft = self.workspace.drafts.get(self.selected_topic)
        qa = self.workspace.quality_reports.get(self.selected_topic)

        self.brief_text.setPlainText((brief.text + "\n\n--- Research ---\n" + brief.research) if brief else "Brief not generated yet.")
        self.draft_text.setPlainText(
            draft.text + ("\n\n--- Variants ---\n" + "\n\n".join(draft.variants) if draft and draft.variants else "")
            if draft else "Draft not generated yet."
        )

        if qa:
            self.qa_score.setValue(int(qa.final_score * 100))
            self.qa_text.setPlainText(json.dumps(qa.__dict__, indent=2, default=str))
            self.policy_text.setPlainText(
                f"Verdict: {qa.verdict}\nCompliance: {qa.compliance:.2f}\n\nRaw policy/alignment output:\n"
                + json.dumps(qa.raw.get("alignment", {}), indent=2, default=str)
            )
        else:
            self.qa_score.setValue(0)
            self.qa_text.setPlainText("QA not run yet.")
            self.policy_text.setPlainText("No policy diagnostics yet.")

    def _refresh_agent_fleet(self) -> None:
        lines = []
        lines.append(f"UI status: {self.runtime_components.get('ui', 'ready')}")
        lines.append(f"Lightweight runtime: {self.runtime_components.get('lightweight_runtime', 'unknown')}")
        lines.append(f"Torch-dependent subsystem: {self.runtime_components.get('torch_subsystem', 'unknown')}")
        lines.append("")
        if not self.runtime_initialized:
            lines.append("Agent runtime status: not initialized")
            if self.runtime_init_error:
                lines.append(f"Last initialization error: {self.runtime_init_error}")
            lines.append("Use 'Initialize Agent Runtime' to load SharedMemory/AgentFactory/agents.")
        else:
            lines.append(f"Agent runtime status: {self.agent_runtime_status}")
            lines.append(f"Registry status: {self.runtime_components.get('registry', 'unknown')}")
            lines.append(f"Core agents available: {self.runtime_components.get('core_agents', 'unknown')}")
            lines.append(f"Optional agents available: {self.runtime_components.get('optional_agents', 'unknown')}")
            if self.agent_runtime_error:
                lines.append(f"Runtime note: {self.agent_runtime_error}")
            for name, agent in self.agents.items():
                if agent is None:
                    lines.append(f"- {name}: {self.agent_statuses.get(name, 'unavailable')}")
                    continue
                methods = [m for m in ["perform_task", "predict", "execute", "act", "get_action"] if hasattr(agent, m)]
                lines.append(f"- {name}: online | methods={', '.join(methods) if methods else 'none'}")
            if self.collab is not None:
                lines.append("\nCollaborative metrics:")
                lines.append(json.dumps(self.collab.get_metrics(), indent=2, default=str))
        self.fleet_text.setPlainText("\n".join(lines))


def launch_autopublisher() -> None:
    existing_app = QApplication.instance()
    app = existing_app or QApplication(sys.argv)
    win = AutopublisherWindow()
    win.show()
    if existing_app is None:
        app.exec_()


if __name__ == "__main__":
    try:
        launch_autopublisher()
    except Exception:
        traceback.print_exc()
