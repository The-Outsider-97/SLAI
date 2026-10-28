"""A small, fully inspectable trainable policy for discrete actions: ``residual softmax + REINFORCE with a baseline``.

    logits(x) = prior(x) + W @ x + b            (illegal actions masked to -inf)
    pi(a|x)   = softmax(logits / temperature)

``prior`` is optional environment-supplied guidance (e.g. an offline-trained or hand-written policy).  With ``W = 0, b = 0``
the policy *is* its prior; learning moves only the residual ``(W, b)``, and ``max_param_norm`` bounds how far it may move,
so learned guidance can refine a prior but never silently replace it.

Why this exists: SLAI's torch learners (dqn/maml/rl/rsi) have no entry point for transitions that come from outside SLAI
(see docs/learning_diagnosis.md), and they require torch.  This policy is the generic, numpy-only learner behind that entry
point.  It has no knowledge of any environment.

Identity is derived from state: ``parameter_hash`` covers the inference parameters, ``state_hash`` also the learner's
baseline statistics, ``policy_id`` embeds the first 12 hex characters of ``parameter_hash``.  Two different parameter
sets therefore cannot share a policy id, and an unchanged policy keeps its id.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np  # type: ignore

from .contracts import (
    SCHEMA_VERSION,
    Action,
    CheckpointIntegrityError,
    ContractError,
    EpisodeRecord,
    LearningUpdateResult,
    Observation,
)

ALGORITHM = "residual-softmax-reinforce"


@dataclass(frozen=True)
class PolicyHyperparameters:
    learning_rate: float = 0.05
    gamma: float = 1.0                    # discount; 1.0 suits episodic tasks whose reward arrives at the end
    baseline_momentum: float = 0.9        # EMA coefficient of the (state-independent) return baseline
    entropy_coef: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float = 5.0
    max_param_norm: Optional[float] = None  # Frobenius bound on (W, b); None = unbounded
    advantage_clip: float = 5.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        for name in ("learning_rate", "gamma", "baseline_momentum", "entropy_coef", "weight_decay", "max_grad_norm",
                     "advantage_clip", "temperature"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"hyperparameter {name} must be a finite number, got {v!r}")
        if self.learning_rate <= 0 or self.temperature <= 0 or self.max_grad_norm <= 0 or self.advantage_clip <= 0:
            raise ValueError("learning_rate, temperature, max_grad_norm and advantage_clip must be > 0")
        if not (0.0 <= self.gamma <= 1.0) or not (0.0 <= self.baseline_momentum < 1.0):
            raise ValueError("gamma must be in [0, 1] and baseline_momentum in [0, 1)")
        if self.entropy_coef < 0 or self.weight_decay < 0:
            raise ValueError("entropy_coef and weight_decay must be >= 0")
        if self.max_param_norm is not None and not (math.isfinite(float(self.max_param_norm)) and self.max_param_norm > 0):
            raise ValueError("max_param_norm must be a positive finite number or None")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def _canonical(arr: np.ndarray) -> bytes:
    """Bytes of an array independent of memory layout; -0.0 and 0.0 hash the same."""
    return np.ascontiguousarray(np.asarray(arr, dtype="<f8") + 0.0).tobytes()


class ResidualSoftmaxPolicy:
    def __init__(self, feature_dim: int, action_dim: int, hyper: Optional[PolicyHyperparameters] = None, *,
                 seed: int = 0, name: str = "") -> None:
        for n, v in (("feature_dim", feature_dim), ("action_dim", action_dim)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{n} must be a positive integer, got {v!r}")
        if action_dim < 2:
            raise ValueError("action_dim must be >= 2")
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hyper = hyper or PolicyHyperparameters()
        self.name = name
        self.seed = seed
        self.W = np.zeros((action_dim, feature_dim), dtype=np.float64)
        self.b = np.zeros(action_dim, dtype=np.float64)
        self.baseline_mean = 0.0
        self.baseline_count = 0
        self.training_step = 0
        self.consumed_transitions = 0
        self.consumed_episodes = 0
        self._rng = np.random.default_rng(seed)
        self._hash_cache: Optional[Tuple[str, str]] = None

    # ------------------------------------------------------------------ inference
    def _logits(self, X: np.ndarray, masks: Optional[np.ndarray], priors: Optional[np.ndarray]) -> np.ndarray:
        z = X @ self.W.T + self.b
        if priors is not None:
            z = z + priors
        z = z / self.hyper.temperature
        if masks is not None:
            z = np.where(masks, z, -np.inf)
        return z

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        m = np.max(z, axis=1, keepdims=True)
        e = np.exp(z - m)
        return e / e.sum(axis=1, keepdims=True)

    def _check(self, obs: Observation) -> None:
        if obs.features.shape != (self.feature_dim,):
            raise ContractError("invalid_observation", f"expected {self.feature_dim} features, got shape {obs.features.shape}")
        if obs.action_mask is not None and obs.action_mask.shape != (self.action_dim,):
            raise ContractError("invalid_action_mask", f"expected mask of {self.action_dim}, got {obs.action_mask.shape}")
        if obs.prior_logits is not None and obs.prior_logits.shape != (self.action_dim,):
            raise ContractError("invalid_prior_logits", f"expected {self.action_dim} prior logits, got {obs.prior_logits.shape}")

    def probabilities(self, obs: Observation) -> np.ndarray:
        self._check(obs)
        z = self._logits(obs.features[None, :], None if obs.action_mask is None else obs.action_mask[None, :],
                         None if obs.prior_logits is None else obs.prior_logits[None, :])
        return self._softmax(z)[0]

    def act(self, obs: Observation, *, greedy: bool) -> Action:
        p = self.probabilities(obs)
        if greedy:
            idx = int(np.argmax(p))          # ties -> lowest index: deterministic
        else:
            idx = int(self._rng.choice(self.action_dim, p=p))
        return Action(index=idx, probability=float(p[idx]), probabilities=tuple(float(v) for v in p), greedy=greedy)

    # ------------------------------------------------------------------ identity
    def _hashes(self) -> Tuple[str, str]:
        if self._hash_cache is None:
            head = f"{SCHEMA_VERSION}|{ALGORITHM}|{self.action_dim}|{self.feature_dim}|".encode()
            hp = hashlib.sha256(head + _canonical(self.W) + b"|" + _canonical(self.b)).hexdigest()
            hs = hashlib.sha256(hp.encode() + struct.pack("<dq", self.baseline_mean + 0.0, int(self.baseline_count))).hexdigest()
            self._hash_cache = (hp, hs)
        return self._hash_cache

    @property
    def parameter_hash(self) -> str:
        return self._hashes()[0]

    @property
    def state_hash(self) -> str:
        return self._hashes()[1]

    @property
    def policy_id(self) -> str:
        return f"{self.name + '-' if self.name else ''}{ALGORITHM}-{self.parameter_hash[:12]}"

    def param_norm(self) -> float:
        return float(math.sqrt(float(np.sum(self.W ** 2)) + float(np.sum(self.b ** 2))))

    # ------------------------------------------------------------------ learning
    def update(self, episodes: Sequence[EpisodeRecord]) -> LearningUpdateResult:
        """One policy-gradient step on completed episodes.  Always returns telemetry; never fails silently."""
        t0 = time.perf_counter()
        step_before, hp_before, hs_before = self.training_step, self.parameter_hash, self.state_hash
        norm_before = self.param_norm()
        hyper = self.hyper
        n_trans = sum(len(e.transitions) for e in episodes)

        def skipped(reason: str, **extra: Any) -> LearningUpdateResult:
            return LearningUpdateResult(
                updated=False, skip_reason=reason, step_before=step_before, step_after=self.training_step,
                episodes=len(episodes), transitions=n_trans, learning_rate=hyper.learning_rate, param_norm_before=norm_before,
                param_norm_after=self.param_norm(), parameter_hash_before=hp_before, parameter_hash_after=self.parameter_hash,
                state_hash_before=hs_before, state_hash_after=self.state_hash, baseline=self.baseline_mean,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0, **extra)

        if not episodes or n_trans == 0:
            return skipped("empty_batch")

        xs: List[np.ndarray] = []; ms: List[np.ndarray] = []; ps: List[np.ndarray] = []
        acts: List[int] = []; rets: List[float] = []; credit: List[bool] = []
        invalid = 0
        for ep in episodes:
            ordered = sorted(ep.transitions, key=lambda t: t.sequence)
            g = 0.0
            ep_returns = [0.0] * len(ordered)
            for i in range(len(ordered) - 1, -1, -1):
                g = ordered[i].reward + hyper.gamma * g
                ep_returns[i] = g
            for t, ret in zip(ordered, ep_returns):
                o = t.observation
                ok = (o.features.shape == (self.feature_dim,) and 0 <= t.action < self.action_dim and math.isfinite(ret)
                      and (o.action_mask is None or (o.action_mask.shape == (self.action_dim,) and bool(o.action_mask[t.action])))
                      and (o.prior_logits is None or o.prior_logits.shape == (self.action_dim,)))
                if not ok:
                    invalid += 1
                    continue
                xs.append(o.features)
                ms.append(np.ones(self.action_dim, dtype=bool) if o.action_mask is None else o.action_mask)
                ps.append(np.zeros(self.action_dim) if o.prior_logits is None else o.prior_logits)
                acts.append(t.action); rets.append(ret); credit.append(t.on_policy)
        if not xs:
            self.consumed_episodes += len(episodes); self.consumed_transitions += n_trans
            return skipped("no_valid_transitions", invalid_values=invalid)

        X = np.asarray(xs); M = np.asarray(ms); P0 = np.asarray(ps)
        A = np.asarray(acts); G = np.asarray(rets); C = np.asarray(credit, dtype=bool)
        mean_return = float(G.mean())
        adv = np.clip(G - self.baseline_mean, -hyper.advantage_clip, hyper.advantage_clip)

        Z = self._logits(X, M, P0)
        Pi = self._softmax(Z)
        if not (np.all(np.isfinite(Pi)) and np.all(np.isfinite(adv))):
            self.consumed_episodes += len(episodes); self.consumed_transitions += n_trans
            return skipped("non_finite_values", invalid_values=invalid + int((~np.isfinite(Pi).all(axis=1)).sum()))

        n_credit = int(C.sum())
        # the baseline always learns from the returns it saw, even if no decision is creditable
        self.baseline_mean = hyper.baseline_momentum * self.baseline_mean + (1.0 - hyper.baseline_momentum) * mean_return
        self.baseline_count += len(G)
        self.consumed_episodes += len(episodes); self.consumed_transitions += n_trans
        self._hash_cache = None

        if n_credit == 0:
            return skipped("no_on_policy_transitions", mean_return=mean_return, invalid_values=invalid)

        Y = np.zeros_like(Pi); Y[np.arange(len(A)), A] = 1.0
        logp = np.log(np.clip(Pi[np.arange(len(A)), A], 1e-300, None))
        Gz = (adv * C)[:, None] * (Y - Pi)
        ent = -np.sum(np.where(Pi > 0, Pi * np.log(np.clip(Pi, 1e-300, None)), 0.0), axis=1)
        if hyper.entropy_coef > 0:
            Hz = -Pi * (np.log(np.clip(Pi, 1e-300, None)) + ent[:, None])
            Gz = Gz + hyper.entropy_coef * (C[:, None] * Hz)
        T = hyper.temperature
        gW = (Gz.T @ X) / (T * n_credit)
        gb = Gz.sum(axis=0) / (T * n_credit)
        if hyper.weight_decay > 0:
            gW = gW - hyper.weight_decay * self.W
            gb = gb - hyper.weight_decay * self.b
        grad_norm = float(math.sqrt(float(np.sum(gW ** 2)) + float(np.sum(gb ** 2))))
        loss = float(-np.sum((adv * logp)[C]) / n_credit - hyper.entropy_coef * float(np.sum(ent[C])) / n_credit)
        scale = 1.0 if grad_norm <= hyper.max_grad_norm else hyper.max_grad_norm / grad_norm
        gW *= scale; gb *= scale
        grad_clipped = grad_norm * scale

        W_old, b_old = self.W.copy(), self.b.copy()
        self.W = self.W + hyper.learning_rate * gW
        self.b = self.b + hyper.learning_rate * gb
        if hyper.max_param_norm is not None:
            n = self.param_norm()
            if n > hyper.max_param_norm:
                f = hyper.max_param_norm / n
                self.W = self.W * f; self.b = self.b * f
        if not (np.all(np.isfinite(self.W)) and np.all(np.isfinite(self.b))):
            self.W, self.b = W_old, b_old                       # never keep a poisoned update
            self._hash_cache = None
            return skipped("non_finite_parameters", invalid_values=invalid, mean_return=mean_return)
        self._hash_cache = None
        delta = float(math.sqrt(float(np.sum((self.W - W_old) ** 2)) + float(np.sum((self.b - b_old) ** 2))))
        changed = self.parameter_hash != hp_before
        if changed:
            self.training_step += 1
        return LearningUpdateResult(
            updated=changed, skip_reason=None if changed else "zero_gradient", step_before=step_before, step_after=self.training_step,
            episodes=len(episodes), transitions=n_trans, credited_transitions=n_credit, loss=loss, grad_norm=grad_norm,
            grad_norm_clipped=grad_clipped, param_norm_before=norm_before, param_norm_after=self.param_norm(), param_delta_norm=delta,
            learning_rate=hyper.learning_rate, mean_return=mean_return, mean_advantage=float(adv[C].mean()), baseline=self.baseline_mean,
            entropy=float(ent[C].mean()), invalid_values=invalid, parameter_hash_before=hp_before, parameter_hash_after=self.parameter_hash,
            state_hash_before=hs_before, state_hash_after=self.state_hash, inference_uses_update=changed,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0)

    # ------------------------------------------------------------------ persistence
    def export_arrays(self) -> Dict[str, np.ndarray]:
        return {"W": self.W.copy(), "b": self.b.copy(), "baseline": np.array([self.baseline_mean, float(self.baseline_count)])}

    def counters(self) -> Dict[str, Any]:
        return {"training_step": self.training_step, "consumed_transitions": self.consumed_transitions,
                "consumed_episodes": self.consumed_episodes}

    def load_state(self, arrays: Mapping[str, Any], counters: Mapping[str, Any], *, expect_parameter_hash: Optional[str] = None,
                   expect_state_hash: Optional[str] = None) -> None:
        """Replace the learned state after validating it; the live state is untouched if anything is wrong."""
        try:
            W = np.array(arrays["W"], dtype=np.float64, copy=True)
            b = np.array(arrays["b"], dtype=np.float64, copy=True)
            base = np.array(arrays["baseline"], dtype=np.float64, copy=True).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointIntegrityError(f"checkpoint arrays are missing or malformed: {exc}") from exc
        if W.shape != self.W.shape or b.shape != self.b.shape or base.shape != (2,):
            raise CheckpointIntegrityError(
                f"checkpoint shape mismatch: W{W.shape} b{b.shape} vs expected W{self.W.shape} b{self.b.shape}")
        if not (np.all(np.isfinite(W)) and np.all(np.isfinite(b)) and np.all(np.isfinite(base))):
            raise CheckpointIntegrityError("checkpoint contains non-finite values")
        old = (self.W, self.b, self.baseline_mean, self.baseline_count, self._hash_cache)
        self.W, self.b, self.baseline_mean, self.baseline_count = W, b, float(base[0]), int(base[1])
        self._hash_cache = None
        if (expect_parameter_hash and self.parameter_hash != expect_parameter_hash) or (
                expect_state_hash and self.state_hash != expect_state_hash):
            self.W, self.b, self.baseline_mean, self.baseline_count, self._hash_cache = old
            raise CheckpointIntegrityError("restored parameters do not hash to the checkpoint's recorded hash")
        self.training_step = int(counters.get("training_step", 0))
        self.consumed_transitions = int(counters.get("consumed_transitions", 0))
        self.consumed_episodes = int(counters.get("consumed_episodes", 0))
