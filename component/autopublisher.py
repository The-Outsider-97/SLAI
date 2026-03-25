"""
Autonomous content operations platform that plans, researches, drafts, QA-checks, and packages publish-ready marketing content using SLAI agents.

Core problem solved:
Content operations are slow because research, outlining, drafting, review, and reformatting happen in disconnected tools with heavy manual handoffs.
ContentOps Autopublisher centralizes and automates this pipeline while preserving human final editorial control.
"""

from __future__ import annotations

import sys

from dataclasses import dataclass, field
from typing import Dict, List

from component.styles.main_style import metric_card_style
from component.styles.autopublisher_style import AUTOPUBLISHER_STYLE
from component.utils.main_utils import fmt_pct, utc_timestamp_label

from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.collaborative_agent import CollaborativeAgent
from src.agents.planning.planning_types import Task, TaskType
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("ContentOps Autopublisher")
printer = PrettyPrinter

@dataclass
class AutopublisherWindow(QMainWindow):
    initialized_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    """
    It automates:
    - Topic discovery
    - Keyword and SERP-informed brief creation
    - Multi-format drafting (blog, social snippets, email variants)
    - Policy and quality checks
    - Publication package assembly
    """
    def __init__(self) -> None:
        self.shared_memory = SharedMemory()
        self.factory = AgentFactory()
        self.collab = CollaborativeAgent(shared_memory=self.shared_memory, agent_factory=self.factory) # Coordinates entire editorial job across stages. Handles failover and retries for blocked steps

        self.adaptive_agent = self.factory.create("adaptive", self.shared_memory) # Tunes prompts, outline strategy, and scoring thresholds over time
        self.browser_agent = self.factory.create("browser", self.shared_memory) # Collects source evidence from SERPs, competitor pages, docs, reports. Extracts supporting facts for outlines and claims
        self.evaluation_agent = self.factory.create("evaluation", self.shared_memory) # Scores readability, topical coverage, structure quality, and evidence density. Rejects low-score drafts or routes for revision
        self.language_agent = self.factory.create("language", self.shared_memory) # Generates briefs, outlines, drafts, social cuts, newsletter blurbs, CTA variants. Adapts tone by audience/persona
        self.learning_agent = self.factory.create("learning", self.shared_memory) # Learns from accepted/rejected drafts and performance feedback
        self.knowledge_agent = self.factory.create("knowledge", self.shared_memory) # Stores brand voice rules, product facts, positioning, and prior content. Maintains reusable snippet/claims library
        self.planning_agent = self.factory.create("planning", self.shared_memory) # Builds editorial calendar and execution order. Decomposes each content job into research -> outline -> draft -> QA -> package
        self.reasoning_agent = self.factory.create("reasoning", self.shared_memory) # Structures argument flow and narrative logic. Checks claim consistency and avoids contradiction
        self.safety_agent = self.factory.create("safety", self.shared_memory) # Enforces safety and policy checks (restricted claims, risky phrasing). Applies sanitization where needed

        self._planning_task_registered = False
        self._planning_enabled = True

def launch_autopublisher() -> None:
    existing_app = QApplication.instance()
    app = existing_app or QApplication(sys.argv)
    win = AutopublisherWindow()
    win.show()
    if existing_app is None:
        app.exec_()


if __name__ == "__main__":
    launch_autopublisher()
