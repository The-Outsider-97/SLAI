# Planning Agent System

## Overview
The `src/agents/planning/` package implements a configurable planning stack that combines:

- **Typed planning primitives** (`planning_types.py`).
- **Task scheduling and deadline logic** (`task_scheduler.py`).
- **Probabilistic planning and repair** (`probabilistic_planner.py`).
- **Safety validation + distributed resource orchestration** (`safety_planning.py`, `utils/resource_monitor.py`).
- **Plan execution controls** (`planning_executor.py`).
- **State persistence and checkpoints** (`planning_memory.py`).
- **Metrics and monitoring** (`planning_metrics.py`, `planning_monitor.py`).
- **Adaptive heuristic selection** across ML and experience-based approaches (`heuristic_selector.py` and heuristic modules).

This package is intended for robust plan generation + execution in dynamic, resource-constrained environments.

---

## Directory Layout

```text
src/agents/planning/
├── __init__.py
├── planning_types.py
├── task_scheduler.py
├── probabilistic_planner.py
├── safety_planning.py
├── planning_executor.py
├── planning_memory.py
├── planning_metrics.py
├── planning_monitor.py
├── heuristic_selector.py
├── decision_tree_heuristic.py
├── gradient_boosting_heuristic.py
├── reinforcement_learning_heuristic.py
├── uncertainty_aware_heuristic.py
├── case_based_reasoning_heuristic.py
├── configs/
│   └── planning_config.yaml
├── templates/
│   ├── planning_db.json
│   └── academic_planning_error.json
└── utils/
    ├── __init__.py
    ├── base_heuristic.py
    ├── config_loader.py
    ├── planning_calculations.py
    ├── planning_errors.py
    └── resource_monitor.py
```

---

## Core Data Model (`planning_types.py`)

`planning_types.py` defines the shared types used across the planning system:

- `TaskStatus`, `TaskType`: task lifecycle + abstraction level.
- `Task`: primary unit of work (goal state, priority, parent link, status, constraints).
- `ResourceProfile`, `ClusterResources`: required/available compute resources.
- `TemporalConstraints`: scheduling and dependency timing rules.
- `SafetyMargins` + `SafetyViolation`: guardrail model for resource/temporal constraints.
- `PlanSnapshot`, `PerformanceMetrics`, `Adjustment`, `RepairCandidate`: structured objects used by execution, repair, and analytics.
- `Any`: a strict wrapper type with runtime constraints and serialization helpers.

These dataclasses act as the contract between schedulers, planners, executor, and monitoring components.

---

## Planning Pipeline

A typical control flow looks like this:

1. **Task and method candidates** are prepared.
2. **`HeuristicSelector`** chooses the best heuristic based on budget/context.
3. Selected heuristic estimates the best method for the task.
4. **`TaskScheduler` / `DeadlineAwareScheduler`** determine assignment viability and timing.
5. **`SafetyPlanning` + `ResourceMonitor`** validate resource and temporal margins.
6. **`PlanningExecutor`** executes, validates preconditions/state divergence, and triggers replanning if needed.
7. **`PlanningMemory`** records snapshots/checkpoints.
8. **`PlanningMetrics` + `PlanningMonitor`** aggregate quality/performance/anomaly indicators.

---

## Heuristics and Method Selection

### `HeuristicSelector`
`heuristic_selector.py` orchestrates dynamic strategy choice among:

- **RL** (`reinforcement_learning_heuristic.py`)
- **GB** (`gradient_boosting_heuristic.py`)
- **DT** (`decision_tree_heuristic.py`)
- **UA** (`uncertainty_aware_heuristic.py`)
- **CBR** (`case_based_reasoning_heuristic.py`)

Selector behavior is controlled by `heuristic_selector` config (priority ordering, time budget, accuracy/speed weighting, CBR readiness, etc.).

### Heuristic Modules
All heuristic modules implement a common conceptual interface based on `utils/base_heuristic.py`:

- training/loading from planning history (`templates/planning_db.json`)
- feature extraction from task + world state + method stats
- method ranking / probability estimation
- persistence of trained models under `heuristic_model_path`

#### Short role summary
- **Decision Tree (DT):** interpretable classifier for success-probability scoring.
- **Gradient Boosting (GB):** boosted ensemble for stronger predictive ranking.
- **Reinforcement Learning (RL):** sequential policy adaptation for task chains.
- **Uncertainty-Aware (UA):** confidence-sensitive selection to reduce risky picks.
- **Case-Based Reasoning (CBR):** nearest-case retrieval/adaptation from historical trajectories.

---

## Scheduling and Plan Construction

### `task_scheduler.py`
Provides scheduler classes for capability/deadline-aware assignment:

- `TaskScheduler`
- `DeadlineAwareScheduler`

Core behavior includes risk thresholding, retry policy integration, and estimated duration calculations based on task requirements.

### `probabilistic_planner.py`
Contains probabilistic plan structures and planner logic:

- `ProbabilisticAction`
- `ProbabilisticPlanner`

Used for action likelihood modeling, uncertainty-aware branching, and plan-repair support.

---

## Safety and Resource Control

### `safety_planning.py`
Safety-focused validation/orchestration components:

- `SafetyPlanning`
- `DistributedOrchestrator`

Enforces safety margins and guards against unsafe temporal/resource plans.

### `utils/resource_monitor.py`
`ResourceMonitor` continuously tracks cluster and node resources and supports reservation:

- background refresh loop (`update_interval`)
- static/consul/k8s service-discovery modes
- resource acquisition checks with structured acquisition errors
- available-vs-allocated resource accounting

Exceptions are represented with typed errors from `utils/planning_errors.py` (e.g., `ResourceViolation`, `SafetyMarginError`, `ReplanningError`, `TemporalViolation`).

---

## Execution, Memory, and Observability

### `planning_executor.py`
`PlanningExecutor` handles runtime plan progression and integrity checks:

- precondition lookahead
- state deviation monitoring
- divergence threshold management
- execution snapshots and replanning triggers

### `planning_memory.py`
`PlanningMemory` provides retention-aware checkpointing and history buffering:

- checkpoint directory management
- bounded history
- optional compression
- cleanup policies by age/count

### `planning_metrics.py`
`PlanningMetrics` computes quality/performance indicators, including weighted efficiency composition (success/cost/time).

### `planning_monitor.py`
`PlanningMonitor` performs rolling-window trend analysis and anomaly checks (e.g., success rate, CPU peaks, memory peaks).

---

## Configuration
File: `configs/planning_config.yaml`

Configuration is loaded with `utils/config_loader.py` and drives every subsystem.

### Important sections
- `service_discovery`: resource monitor node discovery mode.
- `safety_margins`: resource + temporal reserve constraints.
- `task_scheduler`: risk threshold and retry policy.
- `heuristic_selector`: selector strategy and performance weighting.
- `global_heuristic` + per-heuristic blocks: model parameters.
- `planning_executor`: runtime integrity checks and thresholds.
- `planning_memory`: checkpoint retention and autosave settings.
- `planning_metrics`: scoring weights and fallback behaviors.
- `planning_monitor`: windows, anomaly thresholds, scan intervals.

> Note: `templates/planning_db.json` contains seed data for method statistics and example tasks/world states used by heuristic components.

---

## Minimal Usage Example

```python
from src.agents.planning.heuristic_selector import HeuristicSelector
from src.agents.planning.planning_executor import PlanningExecutor
from src.agents.planning.planning_memory import PlanningMemory

task = {
    "name": "navigation",
    "goal_state": {"position": "target"},
    "priority": 0.8,
    "parent": None,
}
world_state = {"cpu_available": 0.7, "memory_available": 2048}
candidate_methods = ["A*", "RRT"]
method_stats = {
    ("navigation", "A*"): {"success": 45, "total": 50},
    ("navigation", "RRT"): {"success": 38, "total": 50},
}

selector = HeuristicSelector()
method, probability = selector.select_best_method(
    task=task,
    world_state=world_state,
    candidate_methods=candidate_methods,
    method_stats=method_stats,
)

executor = PlanningExecutor()
memory = PlanningMemory()

checkpoint_id = memory.save_checkpoint({"task": task, "method": method})
# executor.execute_plan(...)  # integrate with your runtime plan format
```

---

## Operational Recommendations
- Keep `planning_db.json` synchronized with production outcomes; stale histories degrade heuristic quality.
- Tune `heuristic_selector.time_budget`, `accuracy_weight`, and `speed_weight` to your latency envelope.
- Set conservative `safety_margins` first, then relax with monitored evidence.
- Persist and rotate checkpoints to support incident replay and postmortem analysis.
- Use monitor anomalies as triggers for retraining or policy fallback activation.
