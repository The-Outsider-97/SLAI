# Parallax Protocol

**Parallax Protocol** is a sci-fi, turn-based cognitive strategy game concept for the R-Games ecosystem.

Players act as orbital systems operators trying to prevent a multi-sector cascade failure while balancing three competing pressures:

- **Array Stability** (technical resilience)
- **Faction Trust** (social/political alignment)
- **Cascade Risk** (systemic failure probability)

The game blends tactical planning, uncertainty management, and socio-emotional decision-making in a mission-driven interface.

---

## Table of Contents

- [Concept Summary](#concept-summary)
- [Design Goals](#design-goals)
- [Core Gameplay Loop](#core-gameplay-loop)
- [Win / Loss Conditions](#win--loss-conditions)
- [Primary Systems](#primary-systems)
- [Mode Variants](#mode-variants)
- [User Interface Overview](#user-interface-overview)
- [Narrative Framing](#narrative-framing)
- [AI Layer (Future)](#ai-layer-future)
- [Technical Notes](#technical-notes)
- [How to View the Visual Mockup](#how-to-view-the-visual-mockup)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Concept Summary

In **Parallax Protocol**, a deteriorating orbital relay network supports the last stable settlements in a fractured near-future system. Every turn, players must interpret incomplete telemetry, commit limited interventions, and negotiate with semi-autonomous factions whose claims may be truthful, partial, or deceptive.

The central design question is:

> Can you keep a complex system alive when information is imperfect and every intervention has tradeoffs?

This game is intended to align with the R-Games theme of cognitive challenge, strategic depth, and reflective learning.

---

## Design Goals

1. **Cognitive depth over mechanical complexity**
   - Emphasize planning, forecasting, and adaptation.
2. **Meaningful socio-emotional choices**
   - Trust decisions should be strategically relevant, not cosmetic.
3. **Transparent risk with uncertain outcomes**
   - Show probability and confidence indicators while preserving ambiguity.
4. **Metacognitive reflection**
   - Include a short debrief stage that asks players why they made key decisions.
5. **Replayable mission structure**
   - Rotating objectives, event chains, and faction behaviors.

---

## Core Gameplay Loop

Each round follows a four-phase cycle:

1. **Forecast**
   - Inspect relay statuses, confidence intervals, and anomaly alerts.
   - Review faction offers/intel with reliability scores.

2. **Commit**
   - Select a fixed number of actions (e.g., 2 actions per round).
   - Actions can target infrastructure, diplomacy, routing, or contingency setup.

3. **Resolve**
   - Outcomes execute simultaneously.
   - Hidden modifiers (event drift, faction intent, accumulated strain) are applied.
   - System indicators update: Stability / Trust / Cascade Risk.

4. **Debrief**
   - Brief prompt asks what decision model the player used.
   - Optional coaching insights connect outcomes to strategy patterns.

---

## Win / Loss Conditions

### Win (example profile)
Achieve one of:

- Maintain **Array Stability > 65%** for `N` consecutive rounds.
- Complete `3` major objectives without any trust collapse.
- Stabilize all critical relays before the scenario timer expires.

### Loss (example profile)
Lose if any occur:

- **Cascade Risk reaches 100%**.
- **Array Stability drops to 0%**.
- All factions enter hostile/withdrawn state (global trust collapse).
- Critical objective deadline expires with unresolved catastrophic event.

> Numeric thresholds are tuning placeholders during prototyping.

---

## Primary Systems

### 1) Relay Network State
- Relays can be: `Stable`, `Degrading`, `Compromised`, or `Offline`.
- Neighboring relay states influence propagation pressure.
- Certain actions reduce local strain but can increase global load elsewhere.

### 2) Resource Economy
- Core resources may include: `Energy`, `Bandwidth`, `Ops Credits`, `Cooldown Slots`.
- High-impact interventions consume scarce resources and introduce future opportunity costs.

### 3) Faction Trust Model
- Factions provide aid, intel, sabotage immunity, or strategic leverage.
- Reliability is imperfect; short-term gains can damage long-term trust.
- Trust modifies event outcomes and available action cards.

### 4) Event & Uncertainty Engine
- Scheduled events (predictable pressure) + stochastic anomalies (uncertain shock).
- Confidence indicators communicate partial observability without complete certainty.

### 5) Debrief / Reflection Layer
- Round-end prompts reinforce transfer learning:
  - *Which signal did you prioritize and why?*
  - *Was your move robust to deception?*
  - *What would you do differently next round?*

---

## Mode Variants

### Solo Campaign
- Story-linked scenarios with persistent modifiers and escalating complexity.

### Daily Operations Challenge
- Deterministic seed + leaderboard scoring based on stability efficiency and risk handling.

### Co-op Command (future)
- 2 players divide responsibilities (Infrastructure Lead / Diplomatic Lead).
- Shared objective, asymmetric information.

---

## User Interface Overview

The visual mockup currently demonstrates:

- **Left panel:** Mission feed + round objectives.
- **Center panel:**
  - Top HUD metrics (Stability / Trust / Cascade Risk)
  - Orbital relay board centered around a core node
  - Phase tracker (Forecast → Commit → Resolve → Debrief)
- **Right panel:** Action commit cards with tradeoffs and risk language.

This layout is designed for fast comprehension of system state + decision options.

---

## Narrative Framing

You are the lead operator of the **Parallax Array**, a legacy orbital system keeping distributed habitats synchronized. Historical conflicts fragmented governance, leaving human factions and autonomous protocol blocs in fragile cooperation.

As degradation accelerates, your role shifts from maintenance to triage under uncertainty.

Narrative pillars:
- Fragile interdependence
- Costly compromises
- Technical and social resilience

---

## AI Layer (Future)

A future AI architecture may include:

- **Signal Analyst Agent**: Prioritizes telemetry interpretation.
- **Planning Agent**: Builds short-horizon intervention plans.
- **Diplomacy Agent**: Evaluates trust and negotiation strategies.
- **Learning Agent**: Adapts scenario tuning and opponent behaviors from outcomes.

These agents can support both NPC behavior and optional player-coaching overlays.

---

## Technical Notes

Current status:

- This is a **concept-stage mockup**, not a playable game module yet.
- A static HTML visual is available for UX direction and stakeholder review.

Potential implementation direction:

- **Frontend:** React or vanilla JS (consistent with existing R-Games modules)
- **Backend:** Python service for scenario state evaluation and AI inference
- **State Model:** Turn-seeded deterministic resolver + stochastic event table

---

## How to View the Visual Mockup

The current visual prototype lives at:

- `concepts/parallax-protocol-mockup.html`

Quick local preview:

```bash
python -m http.server 8000
```

Then open:

- `http://localhost:8000/concepts/parallax-protocol-mockup.html`

---

## Roadmap

### Phase 1 — Foundation
- [ ] Formalize game state schema and action taxonomy
- [ ] Define scenario JSON format and event vocabulary
- [ ] Implement deterministic turn resolver

### Phase 2 — Playable Prototype
- [ ] Interactive board with selectable actions
- [ ] Runtime metric updates and phase transitions
- [ ] Basic scenario progression and score screen

### Phase 3 — Intelligence & Narrative
- [ ] Faction intent modeling
- [ ] Dialogue/negotiation interactions
- [ ] Adaptive debrief and coaching prompts

### Phase 4 — R-Games Integration
- [ ] Launcher integration card + metadata
- [ ] Shared audio/visual polish pipeline
- [ ] Documentation and onboarding flow

---

## Contributing

If you'd like to help shape this game concept:

1. Open an issue describing gameplay, UX, balance, or narrative suggestions.
2. Include concrete examples (turn snapshot, metric changes, proposed alternatives).
3. Keep proposals consistent with R-Games themes: strategic cognition, reflection, and replayable challenge.

---

## License

This concept documentation is distributed under the repository's root license.

