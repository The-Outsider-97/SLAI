# Learning Utils Overview

This document describes the utility layer that supports the learning subsystem.

## Purpose

The utilities in this folder provide reusable primitives so agents and orchestrators can share behavior consistently:

- Input/state normalization and preprocessing.
- Policy/network definitions and optimizer setup.
- Error typing and fault recovery flow.
- Multi-task adaptation support.
- RL engine-level helper abstractions.

## Utility modules

| File | Role | Typical consumers |
|---|---|---|
| `config_loader.py` | Loads learning config sections from YAML. | agents, factories, strategy selectors |
| `state_processor.py` | Converts raw observations into model-ready tensors/features. | `LearningAgent`, `rl_engine` |
| `policy_network.py` | Defines policy network and optimizer factories. | `StrategySelector`, actor-style learners |
| `neural_network.py` | Shared neural building blocks and training helpers. | model-centric learners |
| `learning_calculations.py` | Numeric/statistical helper methods for adaptation metrics. | `LearningAgent`, selectors |
| `multi_task_learner.py` | Task-aware adaptation and multi-task support methods. | orchestration/meta-learning flows |
| `rl_engine.py` | RL execution/runtime support for policy updates and table operations. | RL-variant agents |
| `recovery_system.py` | Recovery policy hooks for degraded learning behavior. | `LearningAgent` |
| `error_calls.py` | Structured domain exceptions and learning-specific error taxonomy. | all subsystem components |

## Operational conventions

1. Keep utility APIs deterministic and side-effect-light unless explicitly stateful.
2. Raise typed errors from `error_calls.py` for recoverable failures.
3. Prefer shared utility helpers over duplicating logic in individual agents.
4. Keep config keys aligned with `configs/learning_config.yaml`.

## Extension checklist

When adding a new utility:

- Add the module here under **Utility modules**.
- Document expected input/output types.
- Specify which subsystem(s) consume it.
- Add or update config keys (if needed) in `configs/learning_config.yaml`.
- Ensure naming conventions follow current patterns (`*_threshold`, `*_size`, `*_rate`).
