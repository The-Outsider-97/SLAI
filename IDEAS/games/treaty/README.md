# Treaty Machine — Expanded Product & Technical Design

Date: 2026-03-23  
Status: Concept expansion for implementation planning

---

## 1) One-line pitch
A diplomacy strategy simulator where players negotiate clause-level treaties, and a multi-agent system continuously models legal consistency, geopolitical reaction, and long-horizon consequences.

---

## 2) Why this concept is viable for current SLAI

### Stack fit
- Reasoning agent supports rule/contradiction-heavy gameplay.
- Planning agent supports scenario simulation over turn horizons.
- Language agent can draft, explain, and summarize treaty text naturally.
- Alignment/safety/evaluation layers can constrain abusive policy space and preserve fair play.

### 90%+ self-reliance rationale
- Content scales from procedural factions, goals, and treaty clause combinations.
- High replayability comes from simulation depth, not high-cost assets.

---

## 3) Target audience and market position
- **Primary audience:** diplomacy/grand-strategy players
- **Secondary audience:** narrative systems/policy enthusiasts
- **Market position:** “Clause-level treaty sandbox with explainable AI simulations”

---

## 4) Core gameplay fantasy
You are a strategic architect of peace, leverage, and coercion. You don’t just choose “yes/no war”—you engineer legal language that shapes future world trajectories.

---

## 5) Core gameplay loops

### Loop A: Negotiation loop
1. Review faction priorities, constraints, and trust levels.
2. Draft or modify treaty clauses.
3. Run AI negotiation rounds and counteroffers.
4. Ratify, reject, or defer.
5. Observe immediate strategic consequences.

### Loop B: Consequence loop
1. Advance turns/time.
2. Observe treaty compliance/non-compliance and spillover effects.
3. Trigger dispute resolution or sanction pathways.
4. Amend clauses to restore stability or gain leverage.

### Loop C: Meta-strategy loop
1. Build long-term diplomatic doctrine.
2. Shape alliance architecture and dependency structures.
3. Optimize for chosen victory profile (stability, influence, prosperity, survival).

---

## 6) Pillars and non-goals

### Design pillars
- **Clause-level agency** (fine-grained legal design matters)
- **Transparent causality** (players can inspect why outcomes happened)
- **Multipolar simulation** (no static factions)
- **Ethical gameplay constraints** through alignment/safety

### Non-goals
- Reflex-combat gameplay
- AAA cinematic campaign production
- Massive synchronous MMO diplomacy backend at launch

---

## 7) Systems design (detailed)

### 7.1 Faction model
- Attributes: ideology, economic profile, risk appetite, military posture, legitimacy
- Dynamic utilities: security, growth, prestige, autonomy, stability
- Memory: prior betrayals, fulfilled obligations, negotiation style

### 7.2 Treaty clause system (DSL)
- Clause types:
  - Trade access and tariffs
  - Resource quotas
  - Security guarantees and patrol rights
  - Verification/audit rights
  - Sanction triggers and arbitration rules
  - Humanitarian carve-outs
- Clause parameters include duration, scope, exceptions, penalties

### 7.3 Compliance and enforcement model
- Compliance probability by faction and context
- Detection fidelity and audit mechanics
- Dispute and arbitration flow
- Escalation ladder: warning -> sanctions -> intervention -> conflict risk

### 7.4 Consequence simulation engine
- Turn-based simulation of economy/security/social effects
- Chain reactions from clause interactions
- Cross-treaty interference checks
- Time-delayed consequences with confidence intervals

### 7.5 Player strategy layer
- Diplomatic capital and reputation as core currencies
- Secret objectives and public commitments
- Optional asymmetric information modes

### 7.6 Explainability layer
- “Clause impact report” before ratification
- “Why this happened” timeline after every turn
- Counterfactual compare for alternative clause sets

---

## 8) Agent mapping (explicit)

### Collaborative agent
- Coordinates negotiation, simulation, and reporting passes

### Reasoning agent
- Validates clause consistency and detects contradictions
- Computes logical implications and conflict points

### Planning agent
- Simulates forward strategic outcomes and fallback plans

### Knowledge agent
- Stores treaties, historical compliance, faction memory vectors

### Language agent
- Drafts treaty text, translates clause semantics to readable briefs
- Generates faction communiqués and player advisories

### Alignment agent
- Enforces design constraints around prohibited behavior spaces

### Safety agent
- Prevents exploitative or out-of-bounds policy combos

### Evaluation agent
- Scores treaty quality, simulation stability, and fairness metrics

### Adaptive/Learning agents
- Tune negotiation heuristics and scenario balancing over time

---

## 9) Autonomy boundary (what AI does vs human does)

### Autonomous
- Generate faction agendas and strategic priorities
- Produce draft treaty structures and alternatives
- Simulate long-horizon outcomes
- Audit compliance and trigger dispute workflows
- Build narrative turn reports and advisories

### Human oversight required
- Global balance philosophy (e.g., anti-snowball rules)
- Ethical/political sensitivity review
- Expansion content direction
- Ranked mode rules and anti-abuse policy

---

## 10) Modes and progression

### Single-player campaign-lite
- Procedural world setup with scenario constraints
- Milestone crises and strategic forks

### Endless sandbox
- Open-ended multipolar simulation with configurable complexity

### Async competitive mode (later)
- Players submit doctrine/treaty packages between turns
- AI resolves and scores rounds asynchronously

### Progression
- Unlock advanced clause types and negotiation tools
- Faction dossier depth and analytics upgrades

---

## 11) Monetization strategy

### Primary
- Premium base game

### Secondary
- Paid scenario packs (regions, eras, scarcity models)
- Expansion packs with new clause families and faction archetypes

### Optional
- Cosmetic UI themes and report styles

### Avoid
- Pay-to-win diplomatic leverage boosts

---

## 12) Platform and UX
- **Best initial platform:** PC
- **Secondary:** tablet/web companion for async play
- **UX must-haves:**
  - readable clause editor
  - consequence previews with uncertainty visualization
  - clear conflict/debug traces

---

## 13) Risks and mitigations

1. **Risk:** Overwhelming complexity for new players  
   **Mitigation:** guided templates, beginner treaty presets, staged tutorials

2. **Risk:** Outcome opacity and player mistrust  
   **Mitigation:** mandatory explainability outputs and counterfactual compare

3. **Risk:** Dominant strategy stagnation  
   **Mitigation:** adaptive faction behavior + scenario modifiers + periodic rebalancing

4. **Risk:** sensitive geopolitical interpretation  
   **Mitigation:** fictionalized settings, robust content policy, review workflows

---

## 14) Complexity and feasibility
- **Complexity level:** Medium core, High full product with async competition
- **Feasibility score:** 9/10
- **Autonomy score:** 9/10
- **Business potential score:** 8/10

---

## 15) Build roadmap

### Phase 1 (6–8 weeks): Core prototype
- Faction model
- Clause DSL and validator
- Turn simulation core
- Basic negotiation UI

### Phase 2 (8–10 weeks): Alpha
- Explainability and counterfactual tools
- Compliance/arbitration systems
- 5–8 scenario archetypes

### Phase 3 (8–12 weeks): Launch candidate
- Progression and unlock systems
- Balance pass and telemetry loops
- Async mode prototype (optional)
- Launch packaging and analytics

---

## 16) MVP acceptance criteria
- Players can author, ratify, and amend treaties with at least 20 meaningful clause permutations
- Every clause has inspectable consequence metadata pre-ratification
- Simulation remains stable for 100+ turns across test seeds
- At least 3 distinct winning strategic archetypes are viable
