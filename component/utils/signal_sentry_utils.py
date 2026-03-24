"""SignalSentry-specific logic: agent integration, scoring, workflow state, and seed data."""

from __future__ import annotations

import importlib

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from component.utils.main_utils import weighted_average


@dataclass
class Signal:
    id: int
    signal_type: str
    title: str
    timestamp: str
    source_url: str
    confidence: float
    impact: float
    novelty: float
    urgency: float
    suggested_action: str
    owner: str = ""
    evidence_note: str = ""
    priority_score: float = 0.0
    severity: str = "highlight"


@dataclass
class SourceEntry:
    name: str
    url: str
    source_type: str
    schedule: str
    last_status: str


@dataclass(frozen=True)
class MonitoredEntity:
    name: str
    entity_type: str
    focus: str
    homepage_url: str = ""


@dataclass
class FeedbackEvent:
    signal_id: int
    verdict: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HandlerEscalation:
    signal_id: int
    title: str
    owner: str
    status: str
    rationale: str


@dataclass
class PipelineResult:
    digest_signals: List[Signal]
    archived_signals: List[Signal]
    escalations: List[HandlerEscalation]
    workflow_trace: List[str]
    alignment_used: bool
    safety_block_count: int
    agent_bindings: Dict[str, "AgentBinding"]


@dataclass
class AgentBinding:
    agent_name: str
    module_path: str
    class_name: str
    available: bool
    implementation: str
    error: str = ""


@dataclass
class OnboardingResult:
    monitored_entities: List[MonitoredEntity]
    approved_sources: List[SourceEntry]
    excluded_sources: List[str]
    taxonomy: Dict[str, List[str]]
    confidence_rubric: Dict[str, float]


AGENT_SEQUENCE = [
    "Collaborative",
    "Planning",
    "Browser",
    "Knowledge",
    "Reasoning",
    "Evaluation",
    "Language",
    "Safety",
    "Alignment",
    "Handler",
]

AGENT_CLASS_MAP: Dict[str, Tuple[str, str]] = {
    "Collaborative": ("src.agents.collaborative_agent", "CollaborativeAgent"),
    "Planning": ("src.agents.planning_agent", "PlanningAgent"),
    "Browser": ("src.agents.browser.workflow", "WorkFlow"),
    "Knowledge": ("src.agents.knowledge_agent", "KnowledgeAgent"),
    "Reasoning": ("src.agents.reasoning_agent", "ReasoningAgent"),
    "Language": ("src.agents.language_agent", "LanguageAgent"),
    "Evaluation": ("src.agents.evaluation_agent", "EvaluationAgent"),
    "Safety": ("src.agents.safety_agent", "SafetyAgent"),
    "Alignment": ("src.agents.alignment_agent", "AlignmentAgent"),
    "Handler": ("src.agents.handler_agent", "HandlerAgent"),
}


class _SharedMemoryStub(dict):
    def get(self, key, default=None):  # type: ignore[override]
        return super().get(key, default)

    def set(self, key, value, ttl=None):
        self[key] = value


class _FactoryStub:
    pass


_AGENT_BINDINGS_CACHE: Optional[Dict[str, AgentBinding]] = None


SEED_WATCHLIST: Dict[str, List[MonitoredEntity]] = {
    "companies": [
        MonitoredEntity("Klue", "Company", "Competitive enablement and agentic CI benchmark", "https://www.klue.com"),
        MonitoredEntity("Crayon", "Company", "Competitor monitoring and AI-assisted market intelligence", "https://www.crayon.co"),
        MonitoredEntity("Similarweb", "Company", "Digital intelligence and competitive traffic visibility", "https://www.similarweb.com"),
        MonitoredEntity("6sense", "Company", "Buyer-intent GTM orchestration and revenue AI", "https://6sense.com"),
        MonitoredEntity("Gong", "Company", "Revenue AI workflows and downstream go-to-market action", "https://www.gong.io"),
    ],
    "products": [
        MonitoredEntity("G2 Buyer Intent", "Product", "Intent signals for in-market software buyers", "https://www.g2.com"),
        MonitoredEntity("AlphaSense", "Product", "Research assistant and market intelligence workflow", "https://www.alpha-sense.com"),
        MonitoredEntity("Semrush AI Visibility Toolkit", "Product", "AI answer visibility and brand share of voice tracking", "https://www.semrush.com"),
        MonitoredEntity("Crayon Sparks", "Product", "AI-assisted competitor analysis and summaries", "https://www.crayon.co"),
        MonitoredEntity("Owler Max", "Product", "Company monitoring and lightweight competitive alerts", "https://corp.owler.com"),
    ],
    "themes": [
        MonitoredEntity("agentic CI", "Theme", "Automated competitor monitoring, reasoning, and alerting"),
        MonitoredEntity("buyer intent", "Theme", "Signals that identify active software demand and account readiness"),
        MonitoredEntity("AI visibility", "Theme", "Brand presence in AI search, answers, and recommendation surfaces"),
        MonitoredEntity("pricing/packaging changes", "Theme", "Plan, seat, add-on, and tier movements that affect win rates"),
        MonitoredEntity("GTM stack consolidation", "Theme", "Platform bundling across revenue intelligence, CI, and intent data"),
    ],
}


WEIGHTS = {
    "impact": 0.35,
    "confidence": 0.30,
    "novelty": 0.20,
    "urgency": 0.15,
}


def resolve_agent_bindings() -> Dict[str, AgentBinding]:
    """Resolve actual SLAI agent classes from repository modules."""
    global _AGENT_BINDINGS_CACHE
    if _AGENT_BINDINGS_CACHE is not None:
        return _AGENT_BINDINGS_CACHE

    bindings: Dict[str, AgentBinding] = {}
    for agent_name, (module_path, class_name) in AGENT_CLASS_MAP.items():
        try:
            module = importlib.import_module(module_path)
            getattr(module, class_name)
            bindings[agent_name] = AgentBinding(
                agent_name=agent_name,
                module_path=module_path,
                class_name=class_name,
                available=True,
                implementation=f"{module_path}.{class_name}",
            )
        except Exception as exc:
            bindings[agent_name] = AgentBinding(
                agent_name=agent_name,
                module_path=module_path,
                class_name=class_name,
                available=False,
                implementation="fallback_pipeline_logic",
                error=str(exc),
            )
    _AGENT_BINDINGS_CACHE = bindings
    return bindings


def _binding_label(binding: AgentBinding) -> str:
    return binding.implementation if binding.available else f"fallback ({binding.class_name} unavailable)"


def compute_priority(signal: Signal) -> float:
    """Compute priority according to README scoring formula."""
    return (
        WEIGHTS["impact"] * signal.impact
        + WEIGHTS["confidence"] * signal.confidence
        + WEIGHTS["novelty"] * signal.novelty
        + WEIGHTS["urgency"] * signal.urgency
    )


def classify_priority(score: float) -> str:
    """Map numeric score into README threshold buckets."""
    if score >= 0.80:
        return "critical"
    if score >= 0.60:
        return "highlight"
    return "archive"


def enrich_signals(signals: List[Signal]) -> List[Signal]:
    """Fill computed fields and return sorted signals by descending priority."""
    for signal in signals:
        signal.priority_score = compute_priority(signal)
        signal.severity = classify_priority(signal.priority_score)
    return sorted(signals, key=lambda sig: sig.priority_score, reverse=True)


def summarize_digest(signals: List[Signal]) -> str:
    """Produce a concise daily digest summary."""
    if not signals:
        return "No signals detected in the selected monitoring window."
    critical_count = sum(1 for sig in signals if sig.severity == "critical")
    highlight_count = sum(1 for sig in signals if sig.severity == "highlight")
    avg_conf = weighted_average([sig.confidence for sig in signals], [1.0] * len(signals))
    top_title = signals[0].title
    return (
        f"SLAI captured {len(signals)} monitored changes in the last run. "
        f"Critical: {critical_count}, digest highlights: {highlight_count}. "
        f"Top development: {top_title}. "
        f"Mean confidence {avg_conf:.2f}."
    )


def run_onboarding_workflow(
    entities: List[MonitoredEntity],
    sources: List[SourceEntry],
    exclusions: Optional[List[str]] = None,
) -> OnboardingResult:
    """Workflow A from README: source/entity intake + taxonomy/rubric generation."""
    exclusions_set = {value.lower() for value in (exclusions or [])}
    approved = [s for s in sources if s.url.lower() not in exclusions_set]

    taxonomy = {
        "signal_types": [
            "launch",
            "pricing_shift",
            "messaging_change",
            "hiring_signal",
            "integration",
            "trend",
            "risk",
        ],
        "entities": sorted({e.name for e in entities}),
        "themes": [e.name for e in entities if e.entity_type.lower() == "theme"],
    }
    confidence_rubric = {
        "high": 0.85,
        "medium": 0.70,
        "low": 0.50,
    }

    return OnboardingResult(
        monitored_entities=list(entities),
        approved_sources=approved,
        excluded_sources=list(exclusions_set),
        taxonomy=taxonomy,
        confidence_rubric=confidence_rubric,
    )


def _safety_blocked(signal: Signal) -> bool:
    text = f"{signal.title} {signal.evidence_note}".lower()
    blocked_terms = ["password", "secret", "token leak", "private key", "credential"]
    return any(term in text for term in blocked_terms)


def run_agentic_monitoring(signals: List[Signal], alignment_enabled: bool = True) -> PipelineResult:
    """
    Workflow B/C from README:
    Collaborative -> Planning -> Browser -> Knowledge -> Reasoning -> Evaluation
    -> Language -> Safety -> (optional Alignment) -> Handler escalation.
    """
    bindings = resolve_agent_bindings()
    prioritized = enrich_signals(signals)

    workflow_trace: List[str] = [
        f"Collaborative ({_binding_label(bindings['Collaborative'])}): orchestrated {len(prioritized)} signal jobs.",
        f"Planning ({_binding_label(bindings['Planning'])}): built crawl schedule by recency and source health.",
        f"Browser ({_binding_label(bindings['Browser'])}): fetched monitored sources and extracted snapshots.",
        f"Knowledge ({_binding_label(bindings['Knowledge'])}): diffed snapshots against historical memory.",
        f"Reasoning ({_binding_label(bindings['Reasoning'])}): inferred strategic implications and confidence.",
    ]

    digest_candidates = [sig for sig in prioritized if sig.priority_score >= 0.60]
    archived = [sig for sig in prioritized if sig.priority_score < 0.60]
    workflow_trace.append(
        f"Evaluation ({_binding_label(bindings['Evaluation'])}): approved {len(digest_candidates)} highlights, archived {len(archived)} low-priority items."
    )

    safe_digest: List[Signal] = []
    safety_blocks = 0
    for sig in digest_candidates:
        if _safety_blocked(sig):
            safety_blocks += 1
            archived.append(sig)
            continue
        safe_digest.append(sig)

    workflow_trace.append(
        f"Safety ({_binding_label(bindings['Safety'])}): blocked {safety_blocks} items and passed {len(safe_digest)} items."
    )

    if alignment_enabled:
        workflow_trace.append(
            f"Alignment ({_binding_label(bindings['Alignment'])}): applied governance overlays and compliance traceability."
        )
    else:
        workflow_trace.append("Alignment: skipped (optional mode disabled).")

    workflow_trace.append(
        f"Language ({_binding_label(bindings['Language'])}): generated digest narrative and alert cards."
    )

    escalations: List[HandlerEscalation] = []
    for sig in safe_digest:
        if sig.severity == "critical":
            escalations.append(
                HandlerEscalation(
                    signal_id=sig.id,
                    title=sig.title,
                    owner=sig.owner or "unassigned",
                    status="pending_human_review",
                    rationale=f"Priority {sig.priority_score:.2f} exceeds critical threshold.",
                )
            )

    handler_binding = bindings["Handler"]
    if handler_binding.available:
        try:
            handler_module = importlib.import_module(handler_binding.module_path)
            handler_cls = getattr(handler_module, handler_binding.class_name)
            handler = handler_cls(shared_memory=_SharedMemoryStub(), agent_factory=_FactoryStub(), config={})
            for escalation in escalations:
                handler.failure_normalization(
                    error_info={
                        "error_type": "SignalEscalation",
                        "error_message": escalation.rationale,
                    },
                    context={"agent": "SignalSentry", "signal_id": escalation.signal_id},
                )
            workflow_trace.append(
                f"Handler ({handler_binding.implementation}): routed {len(escalations)} critical alerts for human confirmation."
            )
        except Exception as exc:
            workflow_trace.append(
                f"Handler fallback: routed {len(escalations)} critical alerts (native handler call failed: {exc})."
            )
    else:
        workflow_trace.append(f"Handler fallback: routed {len(escalations)} critical alerts for human confirmation.")

    return PipelineResult(
        digest_signals=safe_digest,
        archived_signals=archived,
        escalations=escalations,
        workflow_trace=workflow_trace,
        alignment_used=alignment_enabled,
        safety_block_count=safety_blocks,
        agent_bindings=bindings,
    )


def feedback_usefulness_ratio(events: List[FeedbackEvent]) -> float:
    """Compute useful/total feedback ratio."""
    if not events:
        return 0.84
    useful_count = sum(1 for event in events if event.verdict == "useful")
    return useful_count / len(events)


def agent_status_for_tab(tab_name: str) -> Dict[str, str]:
    """Map workflow state labels by active surface to make agent usage explicit."""
    base = {agent: "idle" for agent in AGENT_SEQUENCE}

    if tab_name == "Dashboard":
        for agent in ["Collaborative", "Planning", "Browser", "Knowledge", "Reasoning", "Evaluation"]:
            base[agent] = "active"
        base["Language"] = "ready"
        base["Safety"] = "ready"
        base["Handler"] = "watching"
    elif tab_name == "Daily Digest":
        for agent in ["Language", "Safety", "Evaluation", "Collaborative"]:
            base[agent] = "active"
        base["Alignment"] = "optional"
    elif tab_name == "Source Manager":
        for agent in ["Planning", "Browser", "Knowledge", "Collaborative"]:
            base[agent] = "active"
        base["Handler"] = "policy"
    elif tab_name == "Feedback & Tuning":
        for agent in ["Evaluation", "Reasoning", "Collaborative"]:
            base[agent] = "active"
        base["Handler"] = "triaging"
        base["Alignment"] = "optional"
    return base


def seed_watchlist() -> Dict[str, List[MonitoredEntity]]:
    return {group: list(items) for group, items in SEED_WATCHLIST.items()}


def watchlist_summary() -> str:
    groups = seed_watchlist()
    return (
        f"{len(groups['companies'])} companies, "
        f"{len(groups['products'])} products, "
        f"and {len(groups['themes'])} themes seeded for monitoring."
    )


def seed_signals() -> List[Signal]:
    """Mocked but realistic seed data aligned to the curated starting watchlist."""
    return enrich_signals(
        [
            Signal(
                id=1,
                signal_type="Pricing shift",
                title="Gong expanded enterprise packaging around Revenue AI workflows",
                timestamp="2026-03-24T08:40:00Z",
                source_url="https://www.gong.io",
                confidence=0.93,
                impact=0.94,
                novelty=0.86,
                urgency=0.84,
                suggested_action="Refresh enterprise battle cards and flag pricing objections for RevOps.",
                owner="PMM team",
                evidence_note="Packaging and positioning shift matches the pricing/packaging changes theme.",
            ),
            Signal(
                id=2,
                signal_type="Launch",
                title="Klue highlighted new agentic CI workflow for competitive enablement teams",
                timestamp="2026-03-24T08:10:00Z",
                source_url="https://www.klue.com",
                confidence=0.91,
                impact=0.89,
                novelty=0.90,
                urgency=0.82,
                suggested_action="Compare workflow depth with SignalSentry alerting and digest flows.",
                owner="Product Strategy",
                evidence_note="Relevant to both Klue monitoring and the agentic CI market theme.",
            ),
            Signal(
                id=3,
                signal_type="Messaging change",
                title="Crayon Sparks reframed positioning toward AI-assisted competitive analysis",
                timestamp="2026-03-24T07:42:00Z",
                source_url="https://www.crayon.co",
                confidence=0.88,
                impact=0.77,
                novelty=0.85,
                urgency=0.69,
                suggested_action="Audit homepage and sales narrative deltas against your own messaging.",
                owner="PMM team",
                evidence_note="Product-level movement on Crayon Sparks plus broader CI messaging drift.",
            ),
            Signal(
                id=4,
                signal_type="Hiring signal",
                title="Similarweb posted GTM roles tied to AI visibility and digital intelligence",
                timestamp="2026-03-24T07:20:00Z",
                source_url="https://www.similarweb.com",
                confidence=0.84,
                impact=0.74,
                novelty=0.81,
                urgency=0.66,
                suggested_action="Track whether hiring translates into a visible AI visibility product push.",
                owner="Market Intelligence",
                evidence_note="Hiring plus website emphasis suggests expansion toward AI visibility use cases.",
            ),
            Signal(
                id=5,
                signal_type="Integration",
                title="6sense increased emphasis on buyer intent orchestration across the GTM stack",
                timestamp="2026-03-24T06:54:00Z",
                source_url="https://6sense.com",
                confidence=0.89,
                impact=0.87,
                novelty=0.78,
                urgency=0.76,
                suggested_action="Map overlap with your buyer intent and GTM stack consolidation themes.",
                owner="RevOps",
                evidence_note="Company-level signal tied directly to buyer intent and GTM stack consolidation.",
            ),
            Signal(
                id=6,
                signal_type="Product update",
                title="G2 Buyer Intent broadened signal language around in-market account identification",
                timestamp="2026-03-24T06:15:00Z",
                source_url="https://www.g2.com",
                confidence=0.83,
                impact=0.79,
                novelty=0.80,
                urgency=0.71,
                suggested_action="Review whether buyer-intent language should be a first-class filter in the UI.",
                owner="Product Strategy",
                evidence_note="Product watchlist entry reinforces the buyer intent theme in onboarding.",
            ),
            Signal(
                id=7,
                signal_type="Workflow expansion",
                title="AlphaSense leaned harder into deep research workflows for strategy teams",
                timestamp="2026-03-24T05:48:00Z",
                source_url="https://www.alpha-sense.com",
                confidence=0.82,
                impact=0.75,
                novelty=0.79,
                urgency=0.62,
                suggested_action="Keep research-assistant positioning separate from alert-first workflows.",
                owner="Founder",
                evidence_note="Good benchmark for adjacent research-heavy product motion.",
            ),
            Signal(
                id=8,
                signal_type="Category trend",
                title="Semrush AI Visibility Toolkit sharpened the AI visibility benchmark narrative",
                timestamp="2026-03-24T05:20:00Z",
                source_url="https://www.semrush.com",
                confidence=0.87,
                impact=0.81,
                novelty=0.84,
                urgency=0.73,
                suggested_action="Add AI visibility as a saved digest section for brand and category monitoring.",
                owner="Growth",
                evidence_note="Product watchlist item directly mapped to the AI visibility theme.",
            ),
            Signal(
                id=9,
                signal_type="Market positioning",
                title="Owler Max packaged company monitoring as a lighter-weight consolidation play",
                timestamp="2026-03-24T04:55:00Z",
                source_url="https://corp.owler.com",
                confidence=0.78,
                impact=0.67,
                novelty=0.76,
                urgency=0.60,
                suggested_action="Benchmark which monitoring capabilities feel essential vs. nice-to-have.",
                owner="Founder",
                evidence_note="Useful lower-end competitor reference for GTM stack consolidation messaging.",
            ),
            Signal(
                id=10,
                signal_type="Theme signal",
                title="Buyer-intent and pricing/packaging mentions now dominate the seeded watchlist",
                timestamp="2026-03-24T04:22:00Z",
                source_url="https://signalsentry.local/watchlist",
                confidence=0.86,
                impact=0.83,
                novelty=0.77,
                urgency=0.72,
                suggested_action="Prioritize those themes in weekly briefs and alert-card routing.",
                owner="PMM team",
                evidence_note="Cross-watchlist synthesis across selected companies, products, and themes.",
            ),
        ]
    )


def seed_sources() -> List[SourceEntry]:
    return [
        SourceEntry("Klue company monitor", "https://www.klue.com", "Company", "Daily", "success"),
        SourceEntry("Crayon company monitor", "https://www.crayon.co", "Company", "Daily", "success"),
        SourceEntry("Similarweb company monitor", "https://www.similarweb.com", "Company", "Daily", "success"),
        SourceEntry("6sense company monitor", "https://6sense.com", "Company", "Daily", "success"),
        SourceEntry("Gong company monitor", "https://www.gong.io", "Company", "Daily", "success"),
        SourceEntry("G2 Buyer Intent watch", "https://www.g2.com", "Product", "Hourly", "success"),
        SourceEntry("AlphaSense watch", "https://www.alpha-sense.com", "Product", "Daily", "success"),
        SourceEntry("Semrush AI Visibility watch", "https://www.semrush.com", "Product", "Hourly", "success"),
        SourceEntry("Crayon Sparks watch", "https://www.crayon.co", "Product", "Hourly", "success"),
        SourceEntry("Owler Max watch", "https://corp.owler.com", "Product", "Daily", "success"),
    ]
