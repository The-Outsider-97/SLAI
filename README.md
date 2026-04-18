![Startup](component/assets/startup.png)
---



[//]: <![Flow Diagram](component/assets/flow_diagram.png)>
## What is SLAI?

SLAI is a modular, distributed AI assistant framework composed of autonomous agents. It decomposes complex tasks into specialized processes handled by modular agents. Each agent is specialized (e.g., perception, planning, reasoning) and collectively they:

- Process multimodal user input (text, voice, images)
- Retrieve and validate knowledge from memory and external sources
- Plan and execute complex tasks using tools or APIs
- Perform logical reasoning and generate fluent natural language responses
- Ensure output safety, ethical alignment, and continual improvement via learning
- Adapt over time through user feedback, monitoring, and meta-learning

---

## Flow Diagram

[//]: < See `slai_flowchart.png` in the repo for the full data and agent pipeline.>

```mermaid
graph TD
    U[User / External Trigger] --> CA[Collaborative Agent]
    CA --> AF[Agent Factory + Registry]

    AF --> P[Perception Agent]
    AF --> K[Knowledge Agent]
    AF --> PL[Planning Agent]
    AF --> R[Reasoning Agent]
    AF --> EX[Execution Agent]
    AF --> L[Language Agent]
    AF --> LE[Learning Agent]
    AF --> AD[Adaptive Agent]
    AF --> SA[Safety Agent]
    AF --> EV[Evaluation Agent]
    AF --> AL[Alignment Agent]
    AF --> OB[Observability Agent]
    AF --> RE[Reader Agent]
    AF --> QL[Quality Agent]
    AF --> PR[Privacy Agent]
    AF --> NW[Network Agent]

    P --> ORCH[Task Orchestration]
    K --> ORCH
    PL --> ORCH
    R --> ORCH
    EX --> ORCH
    L --> ORCH

    ORCH --> SAFE[Safety + Alignment Gate]
    SA --> SAFE
    AL --> SAFE

    SAFE --> EVAL[Evaluation & Scoring]
    EV --> EVAL

    EVAL --> OUT[Final Response / Action]
    OUT --> FB[User Feedback + Telemetry]
    FB --> LE
    FB --> AD
    LE --> CA
    AD --> CA

    EVAL --> LOGS[(Memory / Metrics / Logs)]
    LOGS --> K
    LOGS --> LE
```

### Flow Explanation

SLAI begins when a user request or external event reaches the **Collaborative Agent**, which coordinates execution through the **Agent Factory + Registry**. The factory activates specialized agents (perception, knowledge, planning, reasoning, execution, and language) so each stage of understanding and task handling is performed by a focused component.

Outputs from these core agents are merged by **Task Orchestration**, producing a unified intermediate result. Before any user-visible response is released, this result is passed through a dedicated **Safety + Alignment Gate**, informed by both the **Safety Agent** and **Alignment Agent** to enforce policy, risk controls, and behavioral constraints.

The latest agent expansion adds focused support for:

- **Observability Agent** for telemetry, tracing, and runtime diagnostics.
- **Reader Agent** for parsing and recovering structured/unstructured documents.
- **Quality Agent** for semantic, structural, and statistical quality checks.
- **Privacy Agent** for consent, retention, minimization, and auditability controls.
- **Network Agent** for stream/reliability/policy orchestration across distributed components.

After gating, the result proceeds to **Evaluation & Scoring**, where quality and correctness are assessed with support from the **Evaluation Agent**. The system then emits the **Final Response / Action**.

SLAI closes the loop through two feedback channels:

- **User Feedback + Telemetry** updates the **Learning Agent** and **Adaptive Agent**, which in turn inform future coordination decisions through the Collaborative Agent.
- **Memory / Metrics / Logs** generated from evaluation are stored and reused by the **Knowledge Agent** and **Learning Agent** to improve retrieval quality, decision-making, and long-term system performance.

In practice, this forms a continuous cycle: **specialized processing -> orchestration -> safety/alignment enforcement -> evaluation -> output -> learning and adaptation**.

---

---

## Minimum System Requirements

### Hardware

| Component   | Minimum Requirement         | Recommended                                              |
| ----------- | --------------------------- | -------------------------------------------------------- |
| **CPU**     | 4-core (Intel i5 / Ryzen 5) | 8-core (Intel i7 / Ryzen 7) for intensive tasks.         |
| **RAM**     | 16 GB                       | 32 GB (for multitasking)                                 |
| **GPU**     | NVIDIA GTX 1060             | NVIDIA RTX 3060 or higher (for large tasks and training) |
| **Storage** | 10 GB SSD                   | 50 GB SSD (for models + vector DBs + logs)              |

### Software

- **OS:** Ubuntu 22.04+ / Windows 10/11 (with WSL for Linux compatibility)
- **Python:** 3.10+
- **Dependencies:**
   - Core libraries include 'torch' (PyTorch), 'transformers', 'sentence-transformers',
   - 'faiss-cpu (or faiss-gpu)', 'numpy', 'pandas', and 'pydantic'. Optional tools: 'langchain',
   - 'openai' (LLM APIs), 'gradio' (UI prototype), 'flask/uvicorn' (web services), 'graphviz' (visualizations).
- **Additional Libraries:** For specialized features:
  - 'librosa' (audio), 'music21' (music theory/MIDI), 'mido' (MIDI I/O), 'tensorflow-cpu' (optional ML tasks),
  - 'pypdf' (PDF parsing), and 'nltk' (natural language; run nltk.download('punkt')).

---

# How to run

## Clone this repo.
   ```bash
   git clone https://github.com/The-Outsider-97/SLAI.git
   cd SLAI
   ```

2. Set Up a Virtual Environment (Recommended):

On **Linux/MacOS**:
   ```console
   python3 -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

On **Windows (PowerShell)**:
   ```console
   py -3.10 -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

If `py -3.10` is unavailable, use the default launcher target instead:
   ```console
   py -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

To verify which Python versions are installed and visible to the launcher:
   ```console
   py -0p
   ```

Note: If you see an error about execution policy:
   ```console
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\venv\Scripts\Activate.ps1
   ```

3. Install requirements:
   ```console
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Optional: if you are using GPU-enabled PyTorch builds, install the CUDA wheel matching your system from the official PyTorch index.

4. Having trouble with TensorFlow?
   ```console
   py -3.10 -m venv tfenv
   .\tfenv\Scripts\activate
   ```
   ```console
   pip install --upgrade pip setuptools wheel
   pip install tensorflow
   ```

5. Run Model:

   ```console
   python main.py
   ```


## Continuous Integration
- GitHub Actions workflow runs on each push/PR.
- To trigger manually:
  ```bash
  gh workflow run test.yml
   ```

   If user experiencing errors at this stage, run this command to install PyTorch inside the virtual environment.
   CPU-Only Version (lighter):

   ```console
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

   Confirm Torch Installed:

   ```console
   pip list
   ```
   
6. Run the Tests (Optional but Recommended):

   ```console
   pytest tests/
   ```

---
# SLAI: Roadmap to Autonomous Self-Improvement and Refactoring

This document outlines the current state of SLAI (Safe Learning AI) and identifies the required features and improvements for it to reach its goal of safe, mostly autonomous self-improvement and self-refactoring.

---

## Vision

SLAI aims to be an autonomous, modular AI assistant that can:
- Learn and improve over time via reinforcement and user feedback.
- Edit and refactor its own Python source code to enhance performance.
- Operate autonomously with minimal user intervention.
- Ensure safety, reliability, and continuous adaptation.

SLAI is designed to evolve over time based on the user’s hardware capabilities and engagement. It is a long-term project targeting AI autonomy while prioritizing transparency, control, and safe alignment with human users.

---

## Capability Summary

| **Aspect** | **Current Capabilities** | **Missing / Underdeveloped** |
|-----------|--------------------------|-------------------------------|
| **Performance Optimization & Refactoring** | DQN, MAML, RSI, and Evo agents with hyperparameter tuning. RSI module can adjust model structure/config. | No advanced code analysis, static profiling, or AI-guided refactoring. |
| **User Feedback Utilization** | Feedback conceptually embedded in architecture. | No actual ingestion, parsing, or processing of user feedback into actions. |
| **Self-Editing (Code)** | RSI module can modify code (e.g. layers) with rollback. | Edits are hardcoded templates. No intelligent edits, testing, or Git integration. |
| **Autonomous Operation** | TaskRouter + Collaboration Agent coordinate workflows. Automated R&D loop trains agents. | No proactive task generation or idle self-improvement. Minimal long-term memory or preference learning. |

---

## Current Features

### Modular Agent Framework
- Perception, Planning, Reasoning, Language, SafeAI, RSI, and more.
- Agents are dynamically routed based on task input.

### Recursive Self-Improvement (RSI)
- Rewrites model architecture on stagnated reward.
- Can insert new layers and modify config files.
- Hot reloads agents at runtime with backup handling.

### Automated R&D Engine
- Performs hyperparameter tuning (grid/Bayesian).
- Can evaluate and select top-performing agents.

### Learning Agent
- Implements online learning during runtime.
- Supports lifelong improvement through training from new inputs.

### Safety & Rollback System
- Backs up Python files before applying RSI modifications.
- Recovers original code if modifications fail or result in errors.

---

## Recommendations for Advancing SLAI

### 1. Robust Self-Assessment & Testing
- Integrate and auto-run unit and integration tests post-edit.
- Run benchmark queries to measure real-world performance before/after edits.
- Create detailed metrics for agent scoring.

### 2. Version Control Integration
- Implement `GitPython` to version all RSI edits.
- Auto-commit with timestamp and reason for change.
- Store edit results in an experiment log.
- Optional: Merge successful edits into `main` branch only after passing tests.

### 3. Autonomous Refactoring Engine
- Add `CodeImprovementAgent` using LLM for advanced suggestions.
- Profile slow functions and optimize logic beyond neural layers.
- Integrate PEP8 linter and performance cost analyzer.

### 4. User Feedback System
- Frontend: rating system, correction field, feature request queue.
- Backend: interpret ratings as reward modifiers.
- Enable user preferences to persist via embeddings or logs.

### 5. Deeper Autonomy & Scheduling
- Task scheduler to initiate self-checks and improvements.
- Allow idle-time training, code audit, and model tuning.
- Agents should suggest improvements or experiments.
- Develop persistent memory across sessions.

---

## Goals for Long-Term Autonomy

| **Goal** | **Milestone** |
|---------|---------------|
| Self-healing system | Detects faults and patches own modules |
| Meta-learning loop | Learns from feedback on its own suggestions |
| Zero-touch maintenance | Operates for long durations without intervention |
| Preference-aligned behavior | Adjusts based on learned user traits or habits |
| Agent democracy | Voting system for agents to decide structural changes |

---

## Development Roadmap (Proposed)

### Phase 1: Foundation
- Refactor agent factory and task router for extensibility.
- Ensure all agent outputs are logged and rated.

### Phase 2: Safety & Testing
- Enable live testing post-modification.
- Connect Git versioning to rollback logic.

### Phase 3: Feedback Integration
- Launch UI input fields for feedback.
- Train an LLM-based interpreter for feedback-to-action mapping.

### Phase 4: Autonomous Research
- Add research goals as tasks (e.g., tune a model, explore algorithm X).
- Implement automatic literature retrieval and concept learning.

### Phase 5: Real-Time Adaptation
- Support real-time changes and learning during execution.
- Add a runtime UI for agent performance insights.


---
# SLAI v1.6 Roadmap

**Milestone Focus:**  
Moving from modular execution to autonomous collaboration and introspection.

---

## 🎯 Objectives

1. Enable agents to:
   - Analyze their own performance
   - Propose changes to hyperparameters or policies

2. Build a persistent experiment memory:
   - Evaluation history
   - Configs, scores, and logs over time

3. Expand frontend to support:
   - Live experiment management
   - Leaderboards and real-time comparisons
   - Security/compliance logs display

---

## ✅ Checklist: Agent & System Intelligence

| Task | Description | Status |
|------|-------------|--------|
| Agent self-analysis | Each agent can evaluate and log its own weaknesses | [x] |
| Shared scoring memory | All evaluation results pushed to a central ranking list | ☐ |
| Recursive retraining | Underperforming agents can request tuning | ☐ |
| Agent voting mechanism | Agents can vote on proposed actions (task democracy) | ☐ |

---

## ✅ Checklist: Frontend Enhancements

| Task | Description | Status |
|------|-------------|--------|
| Leaderboard panel | Real-time sortable agent leaderboard | ☐ |
| Agent introspection viewer | Show logs, tuning, and outcomes per agent | ☐ |
| Security & compliance logs view | Render violations and audit reports | ☐ |
| Terminal/metric toggle | Switch between live logs and metrics in UI | ☐ |

---

## ✅ Checklist: Experiment Persistence

| Task | Description | Status |
|------|-------------|--------|
| Evaluation history storage | Save each run with config, agent, metrics | ☐ |
| Historical graphs | Plot accuracy/reward/risk score over time | ☐ |
| Per-agent config/version history | Track changes per agent class | ☐ |
| Save/Restore experiment sessions | Export session as JSON or re-load it later | ☐ |

---

## 🧪 Proposed New Modules

| Module | Purpose |
|--------|---------|
| `agent_introspector.py` | Let agents self-reflect on failure conditions |
| `scoreboard.py` | Central registry of all agent scores |
| `session_manager.py` | Save, restore, and replay sessions |

---

## 🚀 Timeline Suggestion

| Week | Goals |
|------|-------|
| Week 1 | Build `scoreboard` + enable live leaderboard UI |
| Week 2 | Add introspection hooks to top 3 agents |
| Week 3 | Enable persistent evaluation logging (DB or JSONL) |
| Week 4 | UI upgrade: views for logs, scores, history, toggles |

---

## How to Contribute

We welcome collaborators interested in AGI safety, reinforcement learning, or AI-driven automation. You can:
- Fork the repository and submit a PR.
- Open issues for feature ideas or design reviews.
- Help with documentation, examples, and agent tutorials.

  
---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

### Contact

Developed by [@The-Outsider-97](https://github.com/The-Outsider-97)

---

> SLAI is an experiment in building safe, scalable, intelligent systems that learn and grow with every user interaction.
