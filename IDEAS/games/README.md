# Game Ideas Viability Matrix for Current SLAI Agent Stack

Date: 2026-03-23

This README is a detailed, practical portfolio of game ideas that can be built and operated using the SLAI agents in their current state, with a target of 90%+ code self-reliance and minimal external dependencies.

---

## 1) Stack-fit assessment (before ideas)

### What this stack is best at in games
- Multi-agent orchestration and modular role separation (planning/reasoning/execution/language/safety/evaluation).
- Task decomposition, scheduling, and policy-based operation loops.
- Rule/probability/consistency-heavy simulation and event generation.
- Text-forward interaction, report generation, and explainability loops.
- Safety/alignment gating for automated operations.

### What this stack is weak at in games
- High-fidelity real-time twitch gameplay and low-latency precision controls.
- AAA-scale 3D content production and cinematic asset pipelines.
- Large synchronous live operations (MMO-like concurrency + anti-cheat ops).
- Deep action-combat animation systems requiring large asset teams.

### Realistic game types
- Strategy/management sims
- Turn-based and async tactics
- Procedural text-first RPG/investigation titles
- Automation, dispatch, and logistics puzzles
- Async competitive simulation games

### Game types to exclude
- Open-world AAA action RPGs
- Hero shooter/MOBA-scale live service
- Massive MMO persistent worlds
- Motion-capture/cinematic-heavy productions

---

## 2) 20 game ideas grouped by category (ranked by viability)

Score format: **Feasibility / Autonomy / Business Potential** (1–10)

---

## Category A — Operations & Management Sims (Most viable)

## 1) Colony Clerk
- **Genre/category:** Colony management sim
- **Core gameplay concept:** Run a fragile colony using policy directives, automated task queues, and incident response.
- **Player audience:** Strategy/management players (RimWorld/Frostpunk-lite overlap)
- **Why it fits this agent stack:** Planning + execution + safety/evaluation map directly to resource triage.
- **Agent(s) needed:** Collaborative, Planning, Execution, Reasoning, Safety, Evaluation, Language
- **Role of each agent:**
  - Collaborative: orchestrates scenario ticks and handoffs
  - Planning: builds daily/weekly colony action plans
  - Execution: simulates workforce actions and outcomes
  - Reasoning: predicts cascade failures and conflicts
  - Safety: enforces hazard thresholds and blocked actions
  - Evaluation: scores stability, efficiency, and resilience
  - Language: surfaces logs, alerts, and reports
- **What the game can do autonomously:** generate incidents, adapt schedules, rebalance contracts, produce debriefs
- **What still requires human oversight:** macro-economy balancing, progression pacing, live-event curation
- **Core gameplay loops:** plan shift -> execute tasks -> resolve incident -> postmortem -> upgrade policy
- **Core systems/features:** resource graph, dispatch board, incident engine, doctrine cards, AI debriefs
- **Why it is viable with 90%+ self-reliance:** value comes from procedural systems rather than manual content
- **Main risks/bottlenecks:** repetitive incidents if scenario grammars are shallow
- **Complexity level:** Medium
- **Monetization options:** Premium + scenario DLC
- **Best format/platform:** PC / desktop
- **Scores:** **10 / 9 / 8**

## 2) Treaty Machine
- **Genre/category:** Diplomacy strategy sim
- **Core gameplay concept:** Negotiate and enforce treaties; every clause has modeled downstream effects.
- **Player audience:** Grand-strategy and policy sim players
- **Why it fits this agent stack:** rule and contradiction reasoning is central gameplay.
- **Agent(s) needed:** Reasoning, Planning, Language, Alignment, Evaluation, Knowledge
- **Role of each agent:** clause consistency, scenario simulation, treaty drafting, ethics constraints, outcome scoring, historical memory
- **Autonomous:** faction stance generation, negotiation cycles, consequence forecasts
- **Human oversight:** faction personality tuning, sensitive-policy constraints
- **Loops:** draft treaty -> simulate turns -> renegotiate -> enforce
- **Systems:** treaty DSL, trust model, sanctions engine, causal timeline
- **90% viability rationale:** primarily text/rules simulation with low asset overhead
- **Risks:** readability and UX complexity
- **Complexity:** Medium
- **Monetization:** Premium + geopolitical scenario packs
- **Platform:** PC/web
- **Scores:** **9 / 9 / 8**

## 3) Cargo Cartel Terminal
- **Genre/category:** Logistics/economy management
- **Core gameplay concept:** Optimize cargo contracts under disruptions and uncertainty.
- **Player audience:** Tycoon and optimization players
- **Why fit:** planner/scheduler plus probabilistic reasoning are directly monetizable gameplay.
- **Agent(s):** Planning, Execution, Reasoning, Adaptive, Evaluation
- **Roles:** contract planning, dispatch simulation, disruption modeling, auto-tuning, KPI scoring
- **Autonomous:** route generation, disruption injections, pricing suggestions
- **Human oversight:** inflation/sink tuning and retention design
- **Loops:** accept contracts -> route optimize -> execute -> reinvest
- **Systems:** network graph, SLA penalties, stochastic delays, fleet states
- **90% viability:** procedural contract volume replaces manual content work
- **Risks:** economy tuning sensitivity
- **Complexity:** Medium
- **Monetization:** Premium + expansion packs
- **Platform:** PC/tablet
- **Scores:** **9 / 8 / 8**

## 4) Incident Commander
- **Genre/category:** Emergency response strategy
- **Core gameplay concept:** Coordinate emergency teams through cascading city events.
- **Audience:** Strategy/simulation audiences
- **Fit:** task routing + safety-aware planning under constraints
- **Agents:** Planning, Execution, Reasoning, Safety, Evaluation, Language
- **Roles:** triage, dispatch, forecast, guardrails, scoring, communication
- **Autonomous:** scenario chains, mission assignments, after-action reports
- **Oversight:** realism boundaries, educational framing
- **Loops:** detect -> allocate -> stabilize -> review
- **Systems:** heatmaps, queue prioritization, fatigue/resource model
- **90% viability:** scenario generators can produce endless sessions
- **Risks:** balancing realism and fun
- **Complexity:** Medium-High
- **Monetization:** Premium + mission packs
- **Platform:** PC
- **Scores:** **9 / 8 / 7**

## 5) Rogue Process Manager
- **Genre/category:** Workflow/ops satire sim
- **Core gameplay concept:** Keep a dysfunctional AI-driven company operational.
- **Audience:** Sim + satire players
- **Fit:** handler/policy/adaptive routing is native capability
- **Agents:** Handler, Planning, Execution, Evaluation, Language
- **Roles:** policy strategy selection, remediation plans, execution, KPI audit, narrative flavor
- **Autonomous:** ticket generation, failure mode chaining, KPI report writing
- **Oversight:** writing tone and policy boundaries
- **Loops:** incident -> diagnose -> patch -> metric review
- **Systems:** process graph, escalation ladder, morale metrics
- **90% viability:** procedural ops incidents drive replayability
- **Risks:** humor can become repetitive
- **Complexity:** Medium
- **Monetization:** Premium + themed expansions
- **Platform:** PC/web
- **Scores:** **8 / 9 / 7**

---

## Category B — Narrative Logic & Investigation

## 6) Audit & Alibi
- **Genre/category:** Detective contradiction game
- **Core gameplay concept:** Break alibis by finding statement/evidence inconsistencies.
- **Audience:** Deduction and courtroom fans
- **Why fit:** contradiction reasoning is core stack strength
- **Agents:** Reasoning, Knowledge, Language, Evaluation
- **Roles:** evidence graph, consistency checks, testimony generation, solvability scoring
- **Autonomous:** case generation, witness variants, clue threading
- **Oversight:** narrative quality and tone
- **Loops:** interrogate -> map claims -> disprove -> close case
- **Systems:** evidence graph, confidence weights, contradiction alerts
- **90% viability:** procedural cases reduce manual script burden
- **Risks:** puzzle repetition if template set too narrow
- **Complexity:** Medium
- **Monetization:** Episodic packs
- **Platform:** PC/mobile
- **Scores:** **8 / 9 / 8**

## 7) Courtroom Contradictions
- **Genre/category:** Legal argument strategy
- **Core gameplay concept:** Build legal argument trees and dismantle opponent logic.
- **Audience:** Narrative strategy players
- **Fit:** rule engine + language generation fit legal-play loops
- **Agents:** Reasoning, Language, Safety, Evaluation
- **Roles:** argument validation, transcript generation, guardrails, verdict quality
- **Autonomous:** trial scenarios, witness statements, objection opportunities
- **Oversight:** legal realism and sensitivity review
- **Loops:** prepare brief -> hearing -> objections -> verdict
- **Systems:** argument tree, precedent memory, credibility score
- **90% viability:** low asset, high logic depth
- **Risks:** niche appeal
- **Complexity:** Medium
- **Monetization:** Premium + season content
- **Platform:** PC/web
- **Scores:** **8 / 8 / 7**

## 8) City of Claims
- **Genre/category:** Civic narrative strategy
- **Core gameplay concept:** Govern a city where conflicting narratives alter behavior.
- **Audience:** Policy sim players
- **Fit:** alignment + safety + reasoning supports claim moderation gameplay
- **Agents:** Reasoning, Alignment, Safety, Planning, Language
- **Roles:** claim scoring, policy simulation, guardrails, public briefing generation
- **Autonomous:** rumor cycles, social response simulations, policy event chains
- **Oversight:** ethics and communications boundaries
- **Loops:** detect claim waves -> choose interventions -> monitor trust impact
- **Systems:** trust map, provenance graph, intervention toolkit
- **90% viability:** agent-generated events provide scalable content
- **Risks:** hard to keep it entertaining vs didactic
- **Complexity:** High
- **Monetization:** Premium + scenario packs + institutional license
- **Platform:** PC
- **Scores:** **8 / 8 / 8**

## 9) The Last Bureaucrat
- **Genre/category:** Branching narrative management
- **Core gameplay concept:** Process high-stakes files in a collapsing state apparatus.
- **Audience:** Narrative/strategy crossover
- **Fit:** language + planning + evaluation are a direct fit for decision narratives
- **Agents:** Language, Planning, Evaluation, Safety
- **Roles:** event generation, consequence planning, scorecards, guardrails
- **Autonomous:** dossier creation, chain consequences, ending variations
- **Oversight:** canon writing and pacing polish
- **Loops:** review file -> choose action -> absorb consequences -> manage reputation
- **Systems:** faction trust, compliance risk, morale and corruption metrics
- **90% viability:** text/system heavy and low production overhead
- **Risks:** UI clarity and consequence transparency
- **Complexity:** Medium
- **Monetization:** Premium narrative title
- **Platform:** PC/mobile
- **Scores:** **8 / 8 / 7**

## 10) Evidence Weave
- **Genre/category:** Procedural mystery sandbox
- **Core gameplay concept:** Infinite mystery generation via causal evidence webs.
- **Audience:** replay-heavy mystery players
- **Fit:** probabilistic/graph reasoning core
- **Agents:** Reasoning, Planning, Knowledge, Evaluation
- **Roles:** causal graph generation, event planning, memory persistence, solvability checks
- **Autonomous:** full-case generation and validation
- **Oversight:** difficulty scaling and frustration control
- **Loops:** gather evidence -> hypothesize -> verify chain -> conclude
- **Systems:** causal DAG, suspicion score, solve validator
- **90% viability:** procedural generation is central product value
- **Risks:** high design burden for fair puzzle generation
- **Complexity:** High
- **Monetization:** Premium + puzzle season packs
- **Platform:** PC/web
- **Scores:** **7 / 9 / 7**

---

## Category C — Automation, Dispatch & Puzzle Strategy

## 11) StoryOps Dungeon
- **Genre/category:** Auto-run tactical roguelite
- **Core gameplay concept:** Players define doctrines; AI parties execute runs and report outcomes.
- **Audience:** auto-battler/roguelite players
- **Fit:** planning/execution/reporting loop is native to stack
- **Agents:** Planning, Execution, Reasoning, Language, Evaluation
- **Roles:** route planning, turn simulation, risk checks, battle logs, run scoring
- **Autonomous:** quest generation, encounter tuning, post-run summaries
- **Oversight:** progression and reward pacing
- **Loops:** define doctrine -> run expedition -> analyze logs -> revise doctrine
- **Systems:** doctrine editor, encounter generator, loot economy
- **90% viability:** procedural runs replace manual level design volume
- **Risks:** combat can feel abstract without visual feedback
- **Complexity:** Medium-High
- **Monetization:** Premium + expansion biomes
- **Platform:** PC/mobile
- **Scores:** **7 / 8 / 8**

## 12) Salvage Contract Grid
- **Genre/category:** Turn-based salvage tactics
- **Core gameplay concept:** Complete salvage contracts on hazard grids with limited actions.
- **Audience:** tactics puzzle players
- **Fit:** execution grid-state and planner deadlines align directly
- **Agents:** Execution, Planning, Reasoning, Safety
- **Roles:** action resolution, contract scheduling, hazard prediction, safety rules
- **Autonomous:** mission generation, hazard variation, contract balancing
- **Oversight:** reward balance and tutorial design
- **Loops:** choose contract -> execute turn plan -> extract -> upgrade
- **Systems:** hazard tiles, toolkits, SLA penalties, extraction logic
- **90% viability:** systematic mission generation supports long tail
- **Risks:** content variety pressure
- **Complexity:** Medium
- **Monetization:** Premium + contract packs
- **Platform:** PC/tablet
- **Scores:** **7 / 8 / 7**

## 13) Blackout Dispatcher
- **Genre/category:** Infrastructure outage sim
- **Core gameplay concept:** Keep city utilities online during cascading failures.
- **Audience:** city/infrastructure sim players
- **Fit:** planning under uncertainty + event queues
- **Agents:** Planning, Reasoning, Execution, Evaluation
- **Roles:** dispatch planning, propagation forecasts, work-order simulation, KPI scoring
- **Autonomous:** outage generation, weather events, load spikes
- **Oversight:** realism controls and onboarding UX
- **Loops:** monitor -> dispatch -> stabilize -> optimize budget
- **Systems:** network graph, outage propagation, crew logistics
- **90% viability:** procedural outage events provide durable replayability
- **Risks:** steep learning curve
- **Complexity:** High
- **Monetization:** Premium + regional map packs
- **Platform:** PC
- **Scores:** **7 / 8 / 8**

## 14) Factory Exception Simulator
- **Genre/category:** Industrial troubleshooting puzzle
- **Core gameplay concept:** Diagnose emergent process exceptions in automated plants.
- **Audience:** logic and systems puzzle players
- **Fit:** handler + execution recovery + planning are direct feature match
- **Agents:** Handler, Execution, Planning, Evaluation, Safety
- **Roles:** remediation strategy, action sequence execution, plan checks, scoring, guardrails
- **Autonomous:** defect synthesis, bottleneck emergence, corrective recommendations
- **Oversight:** domain difficulty calibration
- **Loops:** alert -> isolate cause -> patch -> verify throughput
- **Systems:** process map, rollback/checkpoints, exception taxonomy
- **90% viability:** generated failure states are reusable content
- **Risks:** niche audience if presented too technically
- **Complexity:** Medium-High
- **Monetization:** Premium + pro edition
- **Platform:** PC/web
- **Scores:** **7 / 9 / 7**

## 15) Signal Triager
- **Genre/category:** Queue triage strategy
- **Core gameplay concept:** Prioritize noisy signals under uncertainty and limited analyst time.
- **Audience:** short-session strategy players
- **Fit:** evaluation and uncertainty-aware planning as core mechanics
- **Agents:** Planning, Reasoning, Evaluation, Language
- **Roles:** queue optimization, confidence modeling, outcome scoring, report generation
- **Autonomous:** signal stream generation, false-positive control, debrief creation
- **Oversight:** fairness checks and mode balancing
- **Loops:** classify -> allocate attention -> act/escalate -> review
- **Systems:** confidence pipeline, escalation ladders, analyst fatigue
- **90% viability:** generated data streams drive infinite sessions
- **Risks:** weak progression can reduce retention
- **Complexity:** Medium
- **Monetization:** Premium + daily challenge pass
- **Platform:** PC/mobile
- **Scores:** **7 / 9 / 6**

---

## Category D — Async Competitive & High-Upside Concepts

## 16) Syndicate Accountant
- **Genre/category:** Async economic strategy
- **Core gameplay concept:** Build covert economic networks while managing detection risk.
- **Audience:** high-complexity strategy players
- **Fit:** risk modeling and hidden-state reasoning
- **Agents:** Planning, Reasoning, Evaluation, Safety, Knowledge
- **Roles:** portfolio planning, anomaly detection, score auditing, policy guardrails, memory
- **Autonomous:** rival moves, market shocks, risk events
- **Oversight:** thematic boundaries and compliance framing
- **Loops:** allocate assets -> run turn -> avoid detection -> compound returns
- **Systems:** ledger graph, suspicion index, rival simulation
- **90% viability:** simulation-heavy core; low art dependency
- **Risks:** ethical framing and platform policy concerns
- **Complexity:** High
- **Monetization:** Premium + ranked seasons
- **Platform:** PC/web
- **Scores:** **6 / 8 / 8**

## 17) Diplomat of Dust
- **Genre/category:** Scarcity diplomacy sim
- **Core gameplay concept:** Negotiate survival treaties among post-collapse factions.
- **Audience:** strategy narrative fans
- **Fit:** treaty generation and consequence simulation strengths
- **Agents:** Reasoning, Planning, Language, Alignment
- **Roles:** clause logic, scenario planning, dialogue output, alignment constraints
- **Autonomous:** faction agendas, scarcity events, negotiation rounds
- **Oversight:** lore consistency and writing polish
- **Loops:** negotiate -> ratify -> monitor impacts -> renegotiate
- **Systems:** scarcity model, faction memory, treaty engine
- **90% viability:** mostly text/systems with procedural depth
- **Risks:** needs strong writing direction
- **Complexity:** High
- **Monetization:** Premium + faction expansions
- **Platform:** PC
- **Scores:** **6 / 8 / 7**

## 18) Counterintel Desk
- **Genre/category:** Async deduction PvP
- **Core gameplay concept:** Players submit intelligence operations; AI adjudicates uncertainty.
- **Audience:** competitive strategy players
- **Fit:** evaluation + reasoning + safety moderation for async play
- **Agents:** Reasoning, Evaluation, Safety, Alignment, Language
- **Roles:** hidden-state reasoning, fair scoring, moderation guardrails, match briefings
- **Autonomous:** scenario seeds, turn adjudication, replay narratives
- **Oversight:** anti-cheat policy and dispute resolution
- **Loops:** submit operation -> resolve turn -> analyze report -> adapt strategy
- **Systems:** hidden info model, confidence reports, replay inspector
- **90% viability:** async model lowers live ops pressure
- **Risks:** competitive integrity complexity
- **Complexity:** High
- **Monetization:** F2P + pass + cosmetics
- **Platform:** web/mobile
- **Scores:** **6 / 7 / 8**

## 19) Procedural Guildmaster
- **Genre/category:** Idle guild economy RPG
- **Core gameplay concept:** Manage an autonomous guild economy with AI-generated quests and market cycles.
- **Audience:** idle/RPG economy players
- **Fit:** high autonomy through planning+execution+adaptive tuning
- **Agents:** Planning, Execution, Adaptive, Language, Evaluation
- **Roles:** quest planning, quest resolution, difficulty tuning, narrative reports, KPI balance
- **Autonomous:** quest board generation, AI recruit generation, economy drift correction
- **Oversight:** monetization ethics and retention design
- **Loops:** assign teams -> resolve quests -> upgrade -> prestige
- **Systems:** roster, quest engine, market simulation, prestige meta
- **90% viability:** procedural live content from core agents
- **Risks:** economy inflation and pay-to-win optics
- **Complexity:** Medium
- **Monetization:** F2P + convenience + cosmetics
- **Platform:** mobile/web
- **Scores:** **6 / 9 / 9**

## 20) Policy Panic
- **Genre/category:** Multiplayer party strategy sim
- **Core gameplay concept:** Players propose policy cards; AI simulates city outcomes each round.
- **Audience:** social/streamer-friendly strategy audiences
- **Fit:** rapid simulation + language briefings + evaluation
- **Agents:** Planning, Reasoning, Language, Evaluation, Safety
- **Roles:** policy simulation, interaction resolution, round narration, scorekeeping, guardrails
- **Autonomous:** event generation, scoring, consequence narration
- **Oversight:** moderation and balance patches
- **Loops:** vote -> simulate -> react -> optimize coalition strategy
- **Systems:** policy deck, city-state variables, faction sentiment
- **90% viability:** low asset burden and high replay potential
- **Risks:** balancing kingmaking and party fairness
- **Complexity:** Medium
- **Monetization:** Premium party game + creator packs
- **Platform:** PC/web
- **Scores:** **6 / 8 / 7**

---

## 3) Final ranked subsets

### Top 10 most viable overall
1. Colony Clerk
2. Treaty Machine
3. Cargo Cartel Terminal
4. Incident Commander
5. Rogue Process Manager
6. Audit & Alibi
7. Courtroom Contradictions
8. City of Claims
9. The Last Bureaucrat
10. StoryOps Dungeon

### Top 5 easiest to launch
1. Audit & Alibi
2. Courtroom Contradictions
3. The Last Bureaucrat
4. Rogue Process Manager
5. Signal Triager

### Top 5 most autonomous
1. Colony Clerk
2. Treaty Machine
3. Factory Exception Simulator
4. Signal Triager
5. Procedural Guildmaster

### Top 5 highest-upside businesses
1. Procedural Guildmaster
2. Counterintel Desk
3. Cargo Cartel Terminal
4. City of Claims
5. Blackout Dispatcher

---

## 4) Opportunity pattern summary

- The strongest opportunities are **systems-first strategy products** where planning, simulation, and autonomous event operations are the core fun.
- The stack is most commercially efficient when it generates **procedural contracts/incidents/cases** instead of requiring large authored content teams.
- Highest upside comes from **async multiplayer or creator-layer extensions** added to an already viable single-player simulation core.
