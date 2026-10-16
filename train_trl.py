"""SLAI v2.3 operational tabular RL training.

Runs against src.agents.learning.rl_agent.RLAgent (no replacement algorithm).
Checkpoint format remains RLAgent pickle version 2, despite .pt extension.
This is trusted-local-only: the upstream RLAgent loader uses pickle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import statistics
import sys
import tempfile
import time

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = "trl-operational-1"
ACTIONS = {
    "scheduling": ("fifo", "shortest_job", "earliest_deadline", "highest_priority"),
    "allocation": ("least_loaded", "lowest_cost", "highest_reliability"),
}
STATE_SIZE = 7


@dataclass
class Job:
    arrival: float
    duration: float
    deadline: float
    priority: int
    resources: int
    attempts: int = 0
    ident: int = 0


@dataclass
class Worker:
    speed: float
    cost: float
    reliability: float
    capacity: int
    available: float = 0.0


class OperationalEnv:
    """Finite event-driven workload; actions choose dispatch rules or workers.

    Each step schedules one job. Jobs cannot be discarded to inflate reward.
    All random draws use this environment's own RNG, never the global RNG.
    """
    def __init__(self, profile: str, stage: int = 3, max_steps: int = 128):
        if profile not in ACTIONS or stage not in range(1, 6) or max_steps < 2:
            raise ValueError("invalid environment profile, stage or step bound")
        self.profile, self.stage, self.max_steps = profile, stage, max_steps
        self.rng = random.Random()
        self.queue: list[Job] = []
        self.pending: list[Job] = []
        self.workers: list[Worker] = []
        self.clock = 0.0
        self.steps = 0
        self.total = 0
        self.stats: dict[str, float] = {}

    def reset(self, *, seed: int) -> tuple[tuple, dict]:
        self.rng.seed(int(seed))
        r, stage = self.rng, self.stage
        count = r.randint(10, 17) if stage < 3 else r.randint(16, 27)
        workers = 2 if stage < 3 else r.randint(2, 4)
        self.workers = [Worker(speed=r.uniform(.75, 1.5), cost=r.uniform(.5, 1.7),
                               reliability=(1.0 if stage < 4 else r.uniform(.86, .995)),
                               capacity=(3 if i == 0 or stage < 3 else r.randint(1, 3)))
                        for i in range(workers)]
        self.pending, self.queue = [], []
        self.clock, self.steps, self.total = 0.0, 0, count
        self.stats = dict(completed=0, failed=0, missed=0, waiting=0., tardiness=0.,
                          cost=0., busy=0., invalid=0., attempts=0.)
        for i in range(count):
            duration = r.uniform(.8, 6.) if stage == 1 else r.uniform(.5, 11.)
            arrival = 0.0 if stage == 1 else r.uniform(0, count * (1.3 if stage >= 3 else .5))
            deadline = arrival + duration * r.uniform(1.25, 3.5 if stage < 4 else 2.8)
            priority = r.randint(1, 4) if stage >= 2 else 1
            req = r.randint(1, 3) if stage >= 3 else 1
            self.pending.append(Job(arrival, duration, deadline, priority, req, ident=i))
        self.pending.sort(key=lambda j: (j.arrival, j.ident))
        self._arrive()
        return self.observe(), {"seed": int(seed), "profile": self.profile}

    def _arrive(self) -> None:
        while self.pending and self.pending[0].arrival <= self.clock + 1e-10:
            self.queue.append(self.pending.pop(0))
        if not self.queue and self.pending:
            self.clock = max(self.clock, self.pending[0].arrival)
            while self.pending and self.pending[0].arrival <= self.clock + 1e-10:
                self.queue.append(self.pending.pop(0))

    @staticmethod
    def _bin(x: float, cuts: tuple[float, ...]) -> int:
        return sum(x >= c for c in cuts)

    def observe(self) -> tuple:
        self._arrive()
        q = self.queue
        return encode_operational_state(
            self.profile, len(q),
            min((j.deadline - max(self.clock, j.arrival) - j.duration for j in q), default=0.),
            (max(j.priority for j in q) - min(j.priority for j in q)) if q else 0,
            (max(j.duration for j in q) - min(j.duration for j in q)) if q else 0.,
            max((w.available - self.clock for w in self.workers), default=0.),
            len(self.pending))

    def _job(self, action: int) -> Job:
        if self.profile == "allocation":
            return min(self.queue, key=lambda j: (j.deadline, -j.priority, j.ident))
        name = ACTIONS[self.profile][action]
        key = {
            "fifo": lambda j: (j.arrival, j.ident),
            "shortest_job": lambda j: (j.duration, j.ident),
            "earliest_deadline": lambda j: (j.deadline, j.ident),
            "highest_priority": lambda j: (-j.priority, j.deadline, j.ident),
        }[name]
        return min(self.queue, key=key)

    def _worker(self, action: int, job: Job) -> Worker:
        eligible = [w for w in self.workers if w.capacity >= job.resources]
        if not eligible:
            raise RuntimeError("No eligible worker; simulator config invalid")
        if self.profile == "allocation":
            name = ACTIONS[self.profile][action]
            key = {
                "least_loaded": lambda w: (max(0., w.available - self.clock), w.cost),
                "lowest_cost": lambda w: (w.cost, w.available),
                "highest_reliability": lambda w: (-w.reliability, w.available),
            }[name]
        else:
            key = lambda w: (max(self.clock, w.available) + job.duration / w.speed,
                             w.cost, -w.reliability)
        return min(eligible, key=key)

    def step(self, action: int) -> tuple[tuple, float, bool, bool, dict]:
        if not isinstance(action, int) or action not in range(len(ACTIONS[self.profile])):
            raise ValueError("invalid action index")
        if not self.queue:
            raise RuntimeError("step after terminal episode")
        job = self._job(action)
        worker = self._worker(action, job)
        self.queue.remove(job)
        self.steps += 1
        self.stats["attempts"] += 1
        start = max(self.clock, worker.available, job.arrival)
        delay = (1.0 if self.stage < 4 else self.rng.uniform(.8, 1.5))
        duration = job.duration / worker.speed * delay
        finish = start + duration
        wait = max(0., start - job.arrival)
        tardy = max(0., finish - job.deadline)
        cost = duration * worker.cost
        worker.available = finish
        self.clock = max(self.clock, min(w.available for w in self.workers))
        failed = self.stage >= 4 and self.rng.random() > worker.reliability
        self.stats["waiting"] += wait
        self.stats["cost"] += cost
        self.stats["busy"] += duration
        self.stats["tardiness"] += job.priority * tardy
        if tardy > 0:
            self.stats["missed"] += 1
        if failed:
            self.stats["failed"] += 1
            if job.attempts < 1:
                job.attempts += 1
                job.arrival = finish
                self.pending.append(job)
                self.pending.sort(key=lambda j: (j.arrival, j.ident))
        else:
            self.stats["completed"] += 1
        # All jobs incur a cost and a tardiness charge; failure is penalized.
        # No drop action exists. Reward cannot increase by skipping hard work.
        reward = (1.0 if not failed else -1.0) - (
            .030 * job.priority * tardy + .012 * wait + .010 * cost)
        self._arrive()
        terminated = not self.queue and not self.pending
        truncated = self.steps >= self.max_steps and not terminated
        info = {"completed": self.stats["completed"], "failed": self.stats["failed"],
                "missed": self.stats["missed"], "waiting": self.stats["waiting"],
                "weighted_tardiness": self.stats["tardiness"], "cost": self.stats["cost"],
                "violation": False, "done": terminated or truncated}
        return self.observe(), float(reward), terminated, truncated, info

    def metrics(self) -> dict[str, float]:
        attempts = max(self.stats["attempts"], 1)
        end = max([self.clock] + [w.available for w in self.workers])
        return {"completion_rate": self.stats["completed"] / self.total,
                "deadline_adherence": 1.0 - self.stats["missed"] / attempts,
                "mean_wait": self.stats["waiting"] / attempts,
                "weighted_tardiness": self.stats["tardiness"],
                "resource_utilization": min(1., self.stats["busy"] / max(1., len(self.workers) * end)),
                "constraint_violation_rate": self.stats["invalid"] / attempts,
                "failures": self.stats["failed"]}


def encode_operational_state(profile: str, queue_length: int, min_slack: float,
                             priority_spread: float, duration_spread: float,
                             max_worker_wait: float, pending: int) -> tuple:
    """Versioned bounded encoder for both training and inference telemetry."""
    if profile not in ACTIONS or min(queue_length, pending) < 0:
        raise ValueError("unknown profile or negative queue/pending count")
    values = (min_slack, priority_spread, duration_spread, max_worker_wait)
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError("nonfinite telemetry feature")
    b = OperationalEnv._bin
    return (VERSION + ":" + profile, b(queue_length, (2, 5, 10)),
            b(min_slack, (-4, 0, 4)), b(priority_spread, (1, 3)),
            b(duration_spread, (3, 7)), b(max_worker_wait, (1, 5)),
            b(pending, (1, 6)))


def scenario_seed(master: int, split: str, index: int, stage: int) -> int:
    data = f"{VERSION}|{master}|{split}|{index}|{stage}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verified(path: Path) -> None:
    signature = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not signature.is_file() or digest(path) != signature.read_text().strip():
        raise ValueError(f"Missing/corrupt checkpoint or SHA-256 manifest: {path}")


def safe_replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def save_agent(agent: Any, path: Path, info: dict) -> None:
    """Save upstream version-2 pickle checkpoint, with metadata in extra keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(".__trl_staging_" + str(os.getpid()) + "_" + path.name)
    staging_mem = staging.with_name(staging.stem + "_memory.pt")
    dest_mem = path.with_name(path.name + ".memory")
    try:
        agent.save_checkpoint(str(staging))
        with staging.open("rb") as f:
            payload = pickle.load(f)  # local self-authored checkpoint only
        if not isinstance(payload, dict) or payload.get("version") != 2:
            raise RuntimeError("Unrecognized RLAgent version-2 checkpoint")
        payload["trl_metadata"] = info
        payload["learning_memory_checkpoint"] = str(dest_mem)
        with staging.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        if not staging_mem.exists():
            raise RuntimeError("RLAgent did not write accompanying memory checkpoint")
        if path.exists() and info.get("status") == "validated":
            verified(path)
            shutil.copy2(path, path.with_name(path.name + ".previous"))
            shutil.copy2(path.with_suffix(path.suffix + ".sha256"),
                         path.with_name(path.name + ".previous.sha256"))
        safe_replace(staging_mem, dest_mem)
        safe_replace(staging, path)
        sig_stage = path.with_suffix(path.suffix + ".sha256.tmp")
        sig_stage.write_text(digest(path))
        safe_replace(sig_stage, path.with_suffix(path.suffix + ".sha256"))
    finally:
        staging.unlink(missing_ok=True)
        staging_mem.unlink(missing_ok=True)


def read_metadata(path: Path) -> dict:
    verified(path)
    with path.open("rb") as f:
        payload = pickle.load(f)  # WARNING: only checkpoints from trusted local SLAI
    if not isinstance(payload, dict):
        raise ValueError("Unsupported checkpoint payload")
    meta = payload.get("trl_metadata")
    if not isinstance(meta, dict) or meta.get("version") != VERSION:
        raise ValueError("Checkpoint has no compatible TRL metadata")
    if payload.get("possible_actions") != list(range(len(ACTIONS[meta["profile"]]))):
        raise ValueError("Checkpoint action indices mismatch")
    if payload.get("state_size") != STATE_SIZE or meta.get("actions") != list(ACTIONS[meta["profile"]]):
        raise ValueError("Checkpoint observation/action contract mismatch")
    if meta.get("algorithm") != "q-learning-lambda0":
        raise ValueError("Unsupported algorithm/checkpoint")
    return meta


def create_agent(profile: str, learning_rate: float, gamma: float, epsilon: float,
                 epsilon_min: float, epsilon_decay: float) -> Any:
    from src.agents.learning.rl_agent import RLAgent
    agent = RLAgent(agent_id="trl_" + profile, possible_actions=list(range(len(ACTIONS[profile]))),
                    state_size=STATE_SIZE)
    if agent.double_q:
        raise RuntimeError("RLAgent double_q=True has an initialization/serialization defect; disable in learning_config.yaml")
    agent.learning_rate, agent.discount_factor = learning_rate, gamma
    agent.epsilon, agent.min_epsilon, agent.epsilon_decay = epsilon, epsilon_min, epsilon_decay
    agent.trace_decay = 0.0  # Q(lambda=0), avoids cross-episode eligibility issues
    agent.replay_interval = 0  # prevent unverified off-policy replay contamination
    agent.q_optimizer_compression = False
    agent.exploration_strategy = "epsilon_greedy"
    return agent


def agent_action(agent: Any, state: tuple) -> int:
    values = agent.get_action_values(state)
    if not values or not all(math.isfinite(float(v)) for v in values.values()):
        raise ValueError("Nonfinite or missing Q values")
    # deterministic tie-break, important for reproducible inference
    return max(sorted(values), key=lambda a: values[a])


def evaluate(agent: Any, profile: str, seeds: list[int], stages: list[int]) -> dict:
    candidates: dict[str, list[dict]] = {"trl": [], "random": []}
    for heuristic in ACTIONS[profile]:
        candidates[heuristic] = []
    for seed, stage in zip(seeds, stages):
        for name in candidates:
            env = OperationalEnv(profile, stage)
            state, _ = env.reset(seed=seed)
            rng = random.Random(seed ^ 0xC0FFEE)
            total = 0.0
            done = False
            inference_ns = 0
            while not done:
                tick = time.perf_counter_ns()
                action = agent_action(agent, state) if name == "trl" else (
                    rng.randrange(len(ACTIONS[profile])) if name == "random" else ACTIONS[profile].index(name))
                inference_ns += time.perf_counter_ns() - tick
                state, reward, term, trunc, _ = env.step(action)
                total += reward
                done = term or trunc
            latency_us = inference_ns / max(env.steps, 1) / 1000.0
            candidates[name].append(dict(reward=total, latency_us=latency_us, **env.metrics()))
    report = {}
    for name, episodes in candidates.items():
        report[name] = {k: statistics.mean(e[k] for e in episodes) for k in episodes[0]}
        report[name]["median_reward"] = statistics.median(e["reward"] for e in episodes)
        report[name]["std_reward"] = statistics.stdev(e["reward"] for e in episodes) if len(episodes) > 1 else 0.
    reward_model = [x["reward"] for x in candidates["trl"]]
    for baseline in ("random",) + ACTIONS[profile]:
        differences = [m - b["reward"] for m, b in zip(reward_model, candidates[baseline])]
        mean = statistics.mean(differences)
        std = statistics.stdev(differences) if len(differences) > 1 else 0.0
        report["trl"]["diff_" + baseline] = mean
        report["trl"]["lower95_" + baseline] = mean - 1.96 * std / math.sqrt(len(differences))
    return report



def eligible(report: dict, profile: str, updates: int) -> bool:
    """Require supported superiority over every tested heuristic."""
    if updates <= 0:
        return False

    result = report["trl"]
    if result["completion_rate"] < 0.95:
        return False

    if result["constraint_violation_rate"] > 0.0:
        return False

    if result["lower95_random"] <= 0.0:
        return False

    for heuristic in ACTIONS[profile]:
        if result["lower95_" + heuristic] <= 0.0:
            return False

    return True



def load_agent(path: Path, args: argparse.Namespace) -> tuple[Any, dict]:
    meta = read_metadata(path)
    if meta["profile"] != args.environment:
        raise ValueError("Checkpoint is for a different operational profile")
    cfg = meta["hyperparameters"]
    agent = create_agent(args.environment, cfg["learning_rate"], cfg["gamma"], cfg["epsilon"],
                         cfg["epsilon_min"], cfg["epsilon_decay"])
    agent.load_checkpoint(str(path))
    if not agent.q_table or agent.possible_actions != list(range(len(ACTIONS[args.environment]))):
        raise ValueError("Checkpoint did not restore a usable Q table")
    return agent, meta


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=.15)
    p.add_argument("--gamma", type=float, default=.95)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-min", type=float, default=.03)
    p.add_argument("--epsilon-decay", type=float, default=.9996)
    p.add_argument("--max-steps", type=int, default=128)
    p.add_argument("--environment", choices=list(ACTIONS), default="scheduling")
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--validation-interval", type=int, default=500)
    p.add_argument("--validation-episodes", type=int, default=80)
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-new", action="store_true", help="discard old working/eligible candidates (not deployment)")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--inspect", action="store_true", help="inspect checkpoint learned values without training")
    p.add_argument("--model-path", type=Path, default=Path("src/agents/learning/models/tabularRL.pt"))
    p.add_argument("--device", choices=["cpu"], default="cpu", help="Tabular Q learning needs no GPU")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    p.add_argument("--max-q-entries", type=int, default=25000)
    return p


def validate_args(a: argparse.Namespace) -> None:
    if (a.episodes < 1 or a.seed < 0 or a.max_steps < 2 or a.validation_interval < 1
        or a.validation_episodes < 10 or a.checkpoint_interval < 1 or a.max_q_entries < 100):
        raise ValueError("invalid positive integer option")
    if not (0 < a.learning_rate <= 1 and 0 <= a.gamma < 1 and 0 <= a.epsilon <= 1
            and 0 <= a.epsilon_min <= a.epsilon and 0 < a.epsilon_decay <= 1):
        raise ValueError("invalid learning/exploration hyperparameters")
    if a.environment == "allocation" and a.model_path == Path("src/agents/learning/models/tabularRL.pt"):
        raise ValueError("Allocation uses incompatible actions: choose --model-path tabularRL_allocation.pt")


def metadata(a: argparse.Namespace, *, episode: int, stage: int, status: str,
             report: dict | None = None) -> dict:
    return {"version": VERSION, "profile": a.environment, "actions": list(ACTIONS[a.environment]),
            "state_size": STATE_SIZE, "algorithm": "q-learning-lambda0", "episode": episode,
            "stage": stage, "status": status, "seed": a.seed,
            "split_versions": {"train": "train", "validation": "validation", "test": "test"},
            "hyperparameters": dict(learning_rate=a.learning_rate, gamma=a.gamma, epsilon=a.epsilon,
                                    epsilon_min=a.epsilon_min, epsilon_decay=a.epsilon_decay),
            "metrics": report}


def evaluate_split(agent: Any, a: argparse.Namespace, split: str, count: int) -> dict:
    seeds = [scenario_seed(a.seed, split, i, 5) for i in range(count)]
    stages = [5 if split == "test" else (3 + i % 3) for i in range(count)]
    return evaluate(agent, a.environment, seeds, stages)


def report_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Public JSON metadata; preserve full RNG state inside the checkpoint.

    Python and NumPy RNG state are restoration details, not evaluation metrics.
    In particular, NumPy's state contains an ndarray that json.dumps rejects.
    """
    return {key: value for key, value in meta.items()
            if key not in {"python_rng", "numpy_rng"}}


def main(argv: list[str] | None = None) -> int:
    a = args_parser().parse_args(argv)
    validate_args(a)
    output = a.model_path.resolve()
    suffix = output.with_suffix(output.suffix + ".resume")  # stable working checkpoint path
    candidate = output.with_suffix(output.suffix + ".candidate")
    eligible_path = output.with_suffix(output.suffix + ".eligible")
    if a.inspect:
        agent, meta = load_agent(output, a)
        samples = sorted(agent.q_table.items(), key=lambda item: abs(float(item[1])), reverse=True)[:12]
        print(json.dumps({"metadata": report_metadata(meta), "q_entries": len(agent.q_table),
                          "updates": agent.total_learning_updates,
                          "top_learned_entries": [{"state": repr(k[0]), "action": ACTIONS[a.environment][k[1]],
                                                   "value": v} for k, v in samples]}, indent=2))
        return 0
    if a.evaluate_only:
        agent, meta = load_agent(output, a)
        report = evaluate_split(agent, a, "test", max(a.validation_episodes, 100))
        print(json.dumps({"artifact": str(output), "metadata": report_metadata(meta), "test": report}, indent=2))
        return 0
    if a.resume and a.force_new:
        raise ValueError("--resume and --force-new conflict")
    if not a.resume:
        if output.is_file():
            verified(output)  # do not silently overwrite invalid deployment
        if a.force_new:
            for artifact in (candidate, eligible_path, suffix):
                for suffix_ in ("", ".sha256", ".memory"):
                    artifact.with_name(artifact.name + suffix_).unlink(missing_ok=True)
        elif any(p.exists() for p in (candidate, eligible_path, suffix)):
            raise ValueError("Existing training state found: use --resume or --force-new")
    if a.resume:
        agent, meta = load_agent(suffix, a)
        episode = int(meta["episode"])
        stage = int(meta["stage"])
        a.seed = int(meta["seed"])
        if a.max_steps != int(meta.get("max_steps", a.max_steps)) or a.curriculum != bool(meta.get("curriculum", a.curriculum)):
            raise ValueError("Resume max-steps/curriculum differ from checkpoint")
        for k, v in meta["hyperparameters"].items():
            # fail closed: restore settings from saved training run
            setattr(a, {"gamma": "gamma", "learning_rate": "learning_rate",
                        "epsilon": "epsilon", "epsilon_min": "epsilon_min",
                        "epsilon_decay": "epsilon_decay"}[k], v)
        best = float(meta.get("best_score", -math.inf))
        best_eligible = float(meta.get("best_eligible", -math.inf))
        if "python_rng" in meta:
            random.setstate(meta["python_rng"])
        if "numpy_rng" in meta:
            import numpy as np # type: ignore
            np.random.set_state(meta["numpy_rng"])
    else:
        random.seed(a.seed)
        try:
            import numpy as np # type: ignore
            np.random.seed(a.seed % (2 ** 32))
        except ImportError:
            pass
        agent = create_agent(a.environment, a.learning_rate, a.gamma, a.epsilon,
                             a.epsilon_min, a.epsilon_decay)
        episode, stage, best, best_eligible = 0, (1 if a.curriculum else 5), -math.inf, -math.inf
    starting_episode = episode
    started = time.monotonic()
    recent = deque(maxlen=100)
    recent_success = deque(maxlen=100)
    validations = 0
    completed_episode = episode
    active_episode = False
    if output.exists():
        try:
            deployed = read_metadata(output)
            deployed_score = float(deployed["metrics"]["trl"]["reward"])
            best_eligible = max(best_eligible, deployed_score)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Existing deployment is invalid: {exc}") from exc
    def save(path: Path, status: str, report: dict | None = None) -> None:
        md = metadata(a, episode=episode, stage=stage, status=status, report=report)
        md["best_score"] = best
        md["best_eligible"] = best_eligible
        md["max_steps"] = a.max_steps
        md["curriculum"] = bool(a.curriculum)
        md["python_rng"] = random.getstate()
        try:
            import numpy as np # type: ignore
            md["numpy_rng"] = np.random.get_state()
        except ImportError:
            pass
        save_agent(agent, path, md)
    try:
        for episode in range(episode + 1, a.episodes + 1):
            active_episode = True
            env = OperationalEnv(a.environment, stage, a.max_steps)
            state, _ = env.reset(seed=scenario_seed(a.seed, "train", episode, stage))
            total_reward = 0.
            while True:
                action = int(agent.step(state))
                nxt, reward, terminated, truncated, _ = env.step(action)
                agent.receive_reward(reward)
                agent.learn_transition(state, action, reward, nxt, done=terminated)
                total_reward += reward
                state = nxt
                if terminated or truncated:
                    if truncated:
                        agent.end_episode(state, done=False)
                    break
            active_episode = False
            completed_episode = episode
            recent.append(total_reward)
            recent_success.append(env.metrics()["completion_rate"])
            if len(agent.q_table) > a.max_q_entries:
                raise RuntimeError("Q-table budget exceeded; revisit discretization")
            due_validate = episode % a.validation_interval == 0 or episode == a.episodes
            if due_validate:
                before = random.getstate()
                try:
                    import numpy as np # type: ignore
                    np_before = np.random.get_state()
                except ImportError:
                    np_before = None
                report = evaluate_split(agent, a, "validation", a.validation_episodes)
                random.setstate(before)
                if np_before is not None:
                    np.random.set_state(np_before) # type: ignore
                validations += 1
                score = report["trl"]["reward"]
                qualifies = eligible(report, a.environment, agent.total_learning_updates)
                if score > best:
                    best = score
                    save(candidate, "experimental", report)
                if qualifies and score > best_eligible:
                    best_eligible = score
                    save(eligible_path, "validation_passed", report)
                if a.curriculum and stage < 5:
                    level_seeds = [scenario_seed(a.seed, "stage_validation", j, stage) for j in range(a.validation_episodes)]
                    level = evaluate(agent, a.environment, level_seeds, [stage] * a.validation_episodes)
                    if (level["trl"]["lower95_random"] > 0.0 and
                            level["trl"]["completion_rate"] >= .95):
                        stage += 1
                elapsed = time.monotonic() - started
                eta = elapsed * (a.episodes - episode) / max(1, episode - starting_episode)
                print(f"episode={episode}/{a.episodes} stage={stage} reward100={statistics.mean(recent):.3f} "
                      f"completion100={statistics.mean(recent_success):.3f} epsilon={agent.epsilon:.4f} "
                      f"q_entries={len(agent.q_table)} updates={agent.total_learning_updates} "
                      f"validation={score:.3f} best={best:.3f} eligible={qualifies} "
                      f"ETA_s={eta:.0f}", flush=True)
                if a.log_level == "DEBUG":
                    print(json.dumps(report, indent=2))
            if episode % a.checkpoint_interval == 0 or episode == a.episodes:
                save(suffix, "training")
    except (KeyboardInterrupt, Exception):
        if not active_episode and completed_episode > 0:
            try:
                episode = completed_episode
                save(suffix, "interrupted")
                print(f"Working checkpoint saved: {suffix}", file=sys.stderr)
            except Exception as save_exc:
                print(f"Emergency checkpoint failed: {save_exc}", file=sys.stderr)
        else:
            print("Interrupted mid-episode; resume from last complete checkpoint, not partial updates.", file=sys.stderr)
        raise
    final_test = None
    if eligible_path.exists():
        eligible_agent, eligible_meta = load_agent(eligible_path, a)
        test_count = max(100, a.validation_episodes)
        final_test = evaluate_split(eligible_agent, a, "test", test_count)
        if eligible(final_test, a.environment, eligible_agent.total_learning_updates):
            prev = read_metadata(output) if output.exists() else None
            current = final_test["trl"]["reward"]
            old = float(prev["metrics"]["trl"]["reward"]) if prev else -math.inf
            if current > old:
                promoted = dict(eligible_meta)
                promoted["status"] = "validated"
                promoted["validation_metrics"] = eligible_meta.get("metrics")
                promoted["metrics"] = final_test
                save_agent(eligible_agent, output, promoted)
                print(f"PROMOTED validated checkpoint: {output}", flush=True)
        else:
            print("Held-out test promotion gate failed; deployment unchanged", flush=True)
    summary = {"profile": a.environment, "episodes": episode, "best_validation_reward": best,
               "q_entries": len(agent.q_table), "updates": agent.total_learning_updates,
               "duration_s": round(time.monotonic() - started, 2), "validations": validations,
               "eligible_checkpoint": str(output) if output.exists() else None,
               "working_checkpoint": str(suffix), "candidate": str(candidate),
               "best_validation_eligible": str(eligible_path) if eligible_path.exists() else None,
               "heldout_test": final_test}
    report_path = output.with_suffix(".training.json")
    report_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"TRAIN_TRL FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
