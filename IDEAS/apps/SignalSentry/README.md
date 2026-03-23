# SignalSentry

Autonomous market and competitor intelligence platform powered primarily by SLAI agents.

---

## 1) Product definition

**SignalSentry** continuously monitors selected companies, products, categories, and market themes, then produces:
- Daily and weekly intelligence briefs
- High-priority alert cards (launches, pricing shifts, messaging changes, hiring signals)
- Trend and risk summaries with recommended follow-up actions

The product is designed so SLAI agents perform the majority of value-generating work (crawl -> extract -> reason -> summarize -> alert), with humans focused on strategic interpretation and decision-making.

---

## 2) Target customers

### Primary ICP
- B2B SaaS founders and PMM teams
- Product strategy teams at SMB/mid-market companies
- Agencies running competitive intelligence for clients

### Secondary ICP
- Investor analysts (early-stage trend watch)
- RevOps teams tracking competitor GTM movements

---

## 3) Core problem solved

Teams miss important external changes because manual monitoring is fragmented and inconsistent.

SignalSentry solves this by turning dispersed web activity into a **single prioritized intelligence stream** with source evidence and trend context.

---

## 4) SLAI agent architecture mapping

## Required agents
- **Collaborative Agent**
- **Planning Agent**
- **Browser Agent**
- **Knowledge Agent**
- **Reasoning Agent**
- **Language Agent**
- **Evaluation Agent**
- **Safety Agent**
- **Alignment Agent** (optional but recommended in regulated setups)
- **Handler Agent** (optional for escalation/routing policy)

## Agent-by-agent role breakdown

### Collaborative Agent
- Orchestrates end-to-end job runs
- Routes task types to best agent
- Handles fallback to alternative agents/workflows when one path fails

### Planning Agent
- Creates crawl plans by source priority and recency windows
- Decomposes jobs into subtasks (fetch -> parse -> compare -> score)
- Replans if source fetch/parsing fails

### Browser Agent
- Visits monitored websites, blogs, changelogs, docs, pricing pages, social/news pages
- Extracts text/DOM/PDF content
- Handles retries and dynamic browsing steps

### Knowledge Agent
- Stores source snapshots and extracted facts
- Maintains historical memory for “what changed” detection
- Applies rule templates for categorization and relevance

### Reasoning Agent
- Distinguishes meaningful signal from noise
- Infers implications (e.g., “new enterprise pricing page suggests upmarket motion”)
- Builds confidence scores from corroborating evidence

### Language Agent
- Produces human-readable briefings and alert summaries
- Formats outputs for email, dashboard cards, and API payloads

### Evaluation Agent
- Scores output quality (coverage, confidence, novelty)
- Detects weak/low-evidence items for suppression

### Safety Agent
- Enforces content safety and leakage prevention
- Flags potentially unsafe outputs before distribution

### Alignment Agent (optional)
- Applies organization-specific fairness/compliance policies
- Adds governance traceability for enterprise customers

### Handler Agent (optional)
- Policy-driven escalation routing
- Human-in-the-loop assignment for edge alerts

---

## 5) Autonomous scope (90%+)

## Fully autonomous work
- Scheduled source crawling
- Change detection against previous snapshots
- Signal extraction and categorization
- Priority scoring and deduplication
- Brief and alert generation
- Delivery to configured channels (email/dashboard/API)

## Human oversight points
- Approve initial source list and taxonomy
- Tune scoring thresholds for “critical” alerts
- Validate high-impact strategic conclusions
- Quarterly quality audits of signal relevance

---

## 6) Core workflows

## Workflow A: Onboarding
1. User defines monitored entities (competitors, themes, products)
2. User provides source list + optional exclusions
3. SLAI generates initial taxonomy and confidence rubric
4. User reviews/approves

## Workflow B: Daily monitoring run
1. Planner builds crawl schedule
2. Browser fetches source content
3. Knowledge stores snapshots + extracted facts
4. Reasoning determines significance and implication
5. Evaluation ranks and filters
6. Language generates digest + alert cards
7. Safety/alignment gates output
8. Collaborative publishes results

## Workflow C: Alert escalation
1. Critical signal detected (score threshold exceeded)
2. Handler assigns to human owner/team
3. System attaches evidence bundle and suggested actions
4. Human marks outcome (true positive / low value)
5. Adaptive/Learning components can use feedback to tune scoring over time

---

## 7) Feature specification (MVP -> V2)

## MVP
- Monitored source registry
- Scheduled crawl jobs
- Snapshot diff engine
- Signal scoring + dedup
- Daily digest email
- Dashboard feed with evidence links

## V1.5
- Theme-level trend lines
- Multi-competitor comparison cards
- Slack/Teams notifications
- Basic analyst notes and pinning

## V2
- Sector-specific rule packs (SaaS, fintech, health)
- Confidence calibration dashboard
- API for downstream BI integration
- Cross-language source monitoring

---

## 8) Data model (practical)

## Core entities
- `Organization`
- `Source`
- `CrawlJob`
- `Snapshot`
- `ExtractedSignal`
- `SignalCluster`
- `Alert`
- `Digest`
- `FeedbackEvent`

## Minimal schema hints
- `ExtractedSignal`: type, title, entity, timestamp, source_url, confidence, novelty_score, impact_score
- `Alert`: severity, rationale, linked_signals[], owner, status
- `Digest`: period_start/end, top_signals[], summary_text, generated_at

---

## 9) Scoring and prioritization logic

Use weighted scoring:
- **Impact** (business relevance)
- **Confidence** (evidence quality)
- **Novelty** (new vs repeated)
- **Urgency** (time sensitivity)

Example:
`priority_score = 0.35*impact + 0.30*confidence + 0.20*novelty + 0.15*urgency`

Then apply thresholding:
- `>= 0.80` critical alert
- `0.60–0.79` digest highlight
- `< 0.60` archive unless linked to trend cluster

---

## 10) UX surfaces

- **Ops dashboard:** Live signal feed, trend modules, unresolved alerts
- **Digest view:** Daily/weekly brief
- **Source manager:** Add/edit sources and crawl policy
- **Feedback panel:** Rate signal usefulness and tune rules

---

## 11) Deployment and operations

## Runtime requirements
- Scheduler for recurring jobs
- Persistent storage for snapshots/signals
- Queue for async crawl + extraction tasks
- Observability (latency, failures, false-positive rates)

## Reliability patterns
- Retry with backoff on fetch/parsing failures
- Circuit breaking for repeatedly failing sources
- Fallback parsing pipeline per content type (HTML/PDF)

---

## 12) Security, compliance, and governance

- Source allowlist + robots/policy considerations
- Secrets management for private feeds/APIs
- Output safety gating before alerts are sent
- Audit trails for enterprise (who changed rules/thresholds)

---

## 13) KPIs and success metrics

## Product KPIs
- Weekly active analysts/teams
- Alert open rate
- “Useful signal” feedback ratio
- Time-to-awareness reduction vs baseline

## Model/agent KPIs
- Precision@K for top alerts
- False-positive rate
- Coverage of monitored entities
- Crawl success rate

---

## 14) Main risks and mitigations

- **Risk:** CAPTCHA/source blocking  
  **Mitigation:** diversified sources, retry policy, source health scoring

- **Risk:** Noisy/low-value alerts  
  **Mitigation:** stronger dedup + confidence gating + human feedback loop

- **Risk:** Overinterpretation by automation  
  **Mitigation:** explicit confidence labels + mandatory human sign-off on strategic recommendations

---

## 15) Monetization and packaging

## Pricing models
- Tiered subscription by monitored entities/source volume
- Add-on for premium alerting + API access
- Enterprise package with governance/audit controls

## Suggested plans
- Starter: 10 entities, daily digest
- Growth: 50 entities, hourly scans, collaboration
- Enterprise: custom policy packs, SSO, audit exports

---

## 16) Launch plan (first 60 days)

## Days 1–15
- Build source ingestion + crawl scheduler + raw snapshot storage

## Days 16–30
- Implement extraction, signal scoring, and digest generation

## Days 31–45
- Add dashboard + alerting + safety/evaluation gates

## Days 46–60
- Pilot with 3–5 teams; calibrate thresholds and feedback loop

---

## 17) Why this can be 90%+ self-reliant

The central value chain is entirely digital and repeatable: monitor web sources, identify deltas, infer importance, and produce intelligence outputs. Human time is focused on interpretation and action—not data gathering or first-pass analysis.
