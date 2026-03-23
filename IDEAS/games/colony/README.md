# Colony Clerk — Expanded Product & Technical Design

Date: 2026-03-23  
Status: Concept expansion for implementation planning

---

## 1) One-line pitch
A systems-first colony management sim where players author policy and priorities while an autonomous multi-agent operations core runs the colony minute-to-minute, generating incidents, tradeoffs, and postmortems.

---

## 2) Why this concept is viable for current SLAI

### Stack fit
- Strong use of planning, scheduling, and decomposition for colony task chains.
- Execution agent can model low-level action/state transitions.
- Reasoning agent can simulate cascading consequences and consistency checks.
- Safety/evaluation/alignment layers can constrain hazardous or degenerate policies.

### 90%+ self-reliance rationale
- Most player value comes from procedural operations (incidents, logistics, staffing, priority conflicts), not hand-authored levels.
- Content scale can come from simulation grammars, parameterized templates, and scenario seeds.

---

## 3) Target audience and market position
- **Primary audience:** strategy/management players (RimWorld/Frostpunk/Oxygen Not Included overlap)
- **Secondary audience:** productivity simulation and “AI operators” audience
- **Market position:** “AI colony operations sim” with heavy explainability and incident storytelling

---

## 4) Core gameplay fantasy
You are not micromanaging every worker. You are the **policy designer** and **operations auditor**. The AI handles execution; your mastery is in designing robust operating doctrine under uncertainty.

---

## 5) Core gameplay loops

### Loop A: Shift planning loop (macro)
1. Review colony status and constraints.
2. Set policy cards (risk tolerance, labor allocation, emergency doctrine).
3. Approve strategic priorities and budget allocations.
4. Run shift/day simulation.
5. Analyze outcomes and revise doctrine.

### Loop B: Incident response loop (mid)
1. Receive incident alert (fire, contamination, morale collapse, supply shock).
2. Select response doctrine (aggressive, conservative, balanced).
3. AI dispatches tasks and fallback plans.
4. Monitor response and intervene if needed.
5. Post-incident report updates trust/performance metrics.

### Loop C: Optimization loop (long)
1. Identify recurring bottlenecks.
2. Invest in upgrades/training/process changes.
3. Compare KPI trends over multiple cycles.
4. Iterate toward resilient colony architecture.

---

## 6) Pillars and non-goals

### Design pillars
- **Explainable AI operations** (why each decision happened)
- **Tradeoff-rich decisions** (efficiency vs safety vs morale)
- **Replayable procedural crises**
- **Low-asset, high-system depth**

### Non-goals
- Twitch combat
- Massive handcrafted campaign maps
- MMO-scale synchronous multiplayer

---

## 7) Systems design (detailed)

### 7.1 World and economy
- Resource categories: food, water, oxygen, energy, parts, medicine, research
- Production chains: extraction -> processing -> distribution -> consumption
- Failure states: shortage spirals, infrastructure downtime, social unrest
- Economy sinks/sources tuned by scenario presets

### 7.2 Workforce and staffing
- Role types: engineers, medics, technicians, logistics, governance staff
- Dynamic skills and fatigue modifiers
- Shift scheduling with emergency override policies
- Labor conflicts and morale feedback loops

### 7.3 Incident engine
- Incident classes: environmental, technical, social, external
- Trigger model: threshold triggers + probabilistic shocks + latent risk accumulation
- Escalation paths with branch probabilities
- Mandatory postmortem object for every major incident

### 7.4 Policy/doctrine system
- Player chooses doctrine cards that alter planner heuristics
- Example doctrine axes:
  - Safety-first vs throughput-first
  - Local autonomy vs centralized control
  - Preventive maintenance vs reactive repair
- Doctrine effects are explicit and visible before commit

### 7.5 Mission/task model
- Tasks represented as abstract goals decomposed into primitive actions
- Deadlines, prerequisites, and resource envelopes included
- Planner can propose alternatives with confidence and risk scores

### 7.6 Scoring/KPIs
- Stability index
- Sustainability index
- Safety compliance score
- Morale and trust score
- Throughput/profit score

### 7.7 Replay and explainability
- Timeline replay by tick/incident
- Decision trace view (“what was chosen and why”)
- Counterfactual viewer (“what if policy B had been active”)

---

## 8) Agent mapping (explicit)

### Collaborative agent
- Tick orchestration and agent handoff management

### Planning agent
- Goal decomposition, schedule generation, fallback planning

### Execution agent
- Applies action sequences to colony state and validates transitions

### Reasoning agent
- Contradiction detection (resource impossibility, policy conflicts)
- Cascading consequence estimates

### Knowledge agent
- Maintains scenario memory, historical outcomes, reusable doctrine metadata

### Safety agent
- Blocks unsafe action sequences and policy combinations

### Alignment agent
- Ensures policy/system behavior remains within design constraints

### Evaluation agent
- Measures decision quality, policy efficiency, and scenario solvability

### Language agent
- Alert text, shift reports, postmortems, advisor briefings

### Adaptive/Learning agents
- Suggest parameter tuning and policy optimizations over repeated runs

---

## 9) Autonomy boundary (what AI does vs human does)

### Autonomous
- Incident generation and escalation
- Task decomposition and dispatch
- Workload rebalance during shocks
- KPI auditing and report generation
- Automated balancing suggestions

### Human oversight required
- Economy tuning baselines
- Scenario/theme curation
- Difficulty progression targets
- Final moderation for harmful narrative edge cases
- Monetization and retention ethics decisions

---

## 10) Live operations model (small-team practical)

### Minimal viable live ops
- Weekly scenario seed rotation
- Monthly doctrine pack
- Quarterly expansion map

### Automation-first ops
- Auto-generated daily challenge
- Auto-generated incident packs constrained by quality gates
- Evaluation-driven patch recommendations

---

## 11) Monetization strategy

### Primary
- Premium base game

### Secondary
- Paid scenario/biome doctrine expansions

### Optional
- Cosmetic UI themes
- Creator map/seed bundle revenue share

### Avoid
- Pay-to-win policy boosts
- Exploitative stamina loops

---

## 12) Platform and UX
- **Best initial platform:** PC desktop (mouse-heavy dashboards)
- **Secondary:** tablet after UI simplification
- **UX requirements:**
  - clear incident severity hierarchy
  - explainability at all major decision points
  - low-friction policy editing

---

## 13) Risks and mitigations

1. **Risk:** Repetitive incidents  
   **Mitigation:** incident grammar diversity + novelty constraints + monthly content themes

2. **Risk:** Simulation opacity  
   **Mitigation:** first-class explanation UI and postmortem traces

3. **Risk:** Overcomplex onboarding  
   **Mitigation:** staged tutorial scenarios and prebuilt doctrines

4. **Risk:** Economy instability  
   **Mitigation:** sandbox telemetry + guardrail caps + controlled sinks/sources

---

## 14) Complexity and feasibility
- **Complexity level:** Medium (core), Medium-High (full progression + liveops)
- **Feasibility score:** 10/10
- **Autonomy score:** 9/10
- **Business potential score:** 8/10

---

## 15) Build roadmap

### Phase 1 (6–8 weeks): Core prototype
- Colony state model
- Task/incident loop
- Basic policy cards
- KPI dashboard

### Phase 2 (8–10 weeks): Productization alpha
- Replay and explanation layer
- 4–6 scenario seeds
- Balancing tools and telemetry

### Phase 3 (8–12 weeks): Launch candidate
- Progression/meta layer
- Expanded incident grammar
- Stability/performance pass
- Store/packaging + launch analytics

---

## 16) MVP acceptance criteria
- Runs 100+ simulation days without deadlock or catastrophic balance collapse
- Every major incident has an explainable decision trace
- At least 3 distinct viable playstyles (safe/balanced/aggressive)
- Positive retention signal in closed test (D1/D7 cohort benchmark defined by team)
