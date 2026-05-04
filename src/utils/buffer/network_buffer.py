from __future__ import annotations

import random
import uuid

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence

from .buffer_telemetry import BufferTelemetry
from .eviction_policies import EvictionContext, build_eviction_policy
from .utils.config_loader import get_config_section, load_global_config


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class NetworkMessage:
    """Normalized network message held by NetworkBuffer."""

    message_id: str
    payload: Any
    channel: str
    protocol: str
    fairness_key: str
    priority: int = 0
    enqueued_at: datetime = field(default_factory=_utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= _utcnow()

    @property
    def age_seconds(self) -> float:
        return max(0.0, (_utcnow() - self.enqueued_at).total_seconds())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "channel": self.channel,
            "protocol": self.protocol,
            "fairness_key": self.fairness_key,
            "priority": self.priority,
            "enqueued_at": self.enqueued_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at is not None else None,
            "expired": self.expired,
            "metadata": dict(self.metadata),
            "payload": self.payload,
        }


@dataclass(slots=True)
class NetworkBufferConfig:
    """Config contract for network-facing backpressure and fairness control."""

    capacity: int = 2048
    fairness_strategy: str = "weighted_round_robin"
    default_fairness_key: str = "global"
    drop_strategy: str = "drop_oldest"
    eviction_policy: str = "fifo"
    random_early_drop_min_fill: float = 0.8
    random_early_drop_max_probability: float = 0.4
    high_watermark_ratio: float = 0.9
    low_watermark_ratio: float = 0.6
    default_ttl_seconds: Optional[int] = None
    max_per_key_inflight: int = 0
    max_dequeue_batch: int = 64
    fairness_weights: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, user_config: Optional[Mapping[str, Any]] = None) -> "NetworkBufferConfig":
        load_global_config()
        cfg = dict(get_config_section("network_buffer") or {})
        if user_config:
            cfg.update(dict(user_config))

        capacity = max(1, int(cfg.get("capacity", 2048)))
        fairness_strategy = str(cfg.get("fairness_strategy", "weighted_round_robin")).strip().lower()
        default_fairness_key = str(cfg.get("default_fairness_key", "global")).strip() or "global"
        drop_strategy = str(cfg.get("drop_strategy", "drop_oldest")).strip().lower()
        eviction_policy = str(cfg.get("eviction_policy", "fifo")).strip().lower()

        red_min_fill = float(cfg.get("random_early_drop_min_fill", 0.8))
        red_max_prob = float(cfg.get("random_early_drop_max_probability", 0.4))
        high_wm = float(cfg.get("high_watermark_ratio", 0.9))
        low_wm = float(cfg.get("low_watermark_ratio", 0.6))
        default_ttl = cfg.get("default_ttl_seconds")
        max_per_key_inflight = int(cfg.get("max_per_key_inflight", 0))
        max_dequeue_batch = max(1, int(cfg.get("max_dequeue_batch", 64)))

        if not (0.0 <= red_min_fill <= 1.0):
            raise ValueError("random_early_drop_min_fill must be in [0, 1]")
        if not (0.0 <= red_max_prob <= 1.0):
            raise ValueError("random_early_drop_max_probability must be in [0, 1]")
        if not (0.0 < high_wm <= 1.0):
            raise ValueError("high_watermark_ratio must be in (0, 1]")
        if not (0.0 <= low_wm <= high_wm):
            raise ValueError("low_watermark_ratio must be <= high_watermark_ratio")

        if default_ttl is not None:
            default_ttl = int(default_ttl)
            if default_ttl < 0:
                raise ValueError("default_ttl_seconds must be >= 0")

        weights_raw = cfg.get("fairness_weights", {}) or {}
        fairness_weights: Dict[str, int] = {}
        if isinstance(weights_raw, Mapping):
            for key, value in weights_raw.items():
                key_text = str(key).strip()
                if not key_text:
                    continue
                fairness_weights[key_text] = max(1, int(value))

        return cls(
            capacity=capacity,
            fairness_strategy=fairness_strategy,
            default_fairness_key=default_fairness_key,
            drop_strategy=drop_strategy,
            eviction_policy=eviction_policy,
            random_early_drop_min_fill=red_min_fill,
            random_early_drop_max_probability=red_max_prob,
            high_watermark_ratio=high_wm,
            low_watermark_ratio=low_wm,
            default_ttl_seconds=default_ttl,
            max_per_key_inflight=max_per_key_inflight,
            max_dequeue_batch=max_dequeue_batch,
            fairness_weights=fairness_weights,
        )


class NetworkBuffer:
    """Transport-focused queue abstraction with backpressure, fairness, and drop strategies.

    Design goals:
    - keep enqueue/dequeue O(1) average for normal operation,
    - preserve fairness across producers/tenants/channels,
    - provide explicit backpressure decisions (drop, reject, evict),
    - avoid duplicating network-adapter buffering logic.
    """

    VALID_DROP_STRATEGIES = {
        "drop_oldest",
        "drop_newest",
        "drop_lowest_priority",
        "random_early_drop",
        "reject_new",
        "eviction_policy",
    }

    def __init__(
        self,
        user_config: Optional[Mapping[str, Any]] = None,
        telemetry: Optional[BufferTelemetry] = None,
    ) -> None:
        self.config = NetworkBufferConfig.from_config(user_config=user_config)
        if self.config.drop_strategy not in self.VALID_DROP_STRATEGIES:
            raise ValueError(f"Unsupported drop strategy: {self.config.drop_strategy}")

        self.telemetry = telemetry or BufferTelemetry(component_name="network_buffer")
        self._lock = RLock()
        self._rand = random.Random()

        self._global_queue: Deque[str] = deque()
        self._messages: Dict[str, NetworkMessage] = {}
        self._queue_by_key: Dict[str, Deque[str]] = defaultdict(deque)
        self._inflight_by_key: Dict[str, int] = defaultdict(int)
        self._active_fairness_keys: Deque[str] = deque()
        self._fairness_weights: Dict[str, int] = dict(self.config.fairness_weights)
        self._fairness_credits: Dict[str, int] = {}
        self._eviction_policy = build_eviction_policy({"eviction": {"policy": self.config.eviction_policy}})
        self._stats: Dict[str, int] = {
            "enqueued": 0,
            "dequeued": 0,
            "dropped": 0,
            "expired": 0,
            "acked": 0,
            "nacked": 0,
            "backpressure_signals": 0,
            "evictions": 0,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def enqueue(
        self,
        payload: Any,
        *,
        channel: str = "unknown",
        protocol: str = "unknown",
        fairness_key: Optional[str] = None,
        priority: int = 0,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.telemetry.time_block("enqueue_latency_seconds"):
            with self._lock:
                self._prune_expired_locked()
                key = self._normalize_fairness_key(fairness_key)
                if self._would_exceed_per_key_limit(key):
                    self._stats["backpressure_signals"] += 1
                    self.telemetry.increment("enqueue_rejected_per_key_inflight")
                    return self._decision_payload(False, reason="max_per_key_inflight_reached", key=key)

                if len(self._messages) >= self.config.capacity:
                    admitted, reason = self._apply_backpressure_on_enqueue_locked(incoming_priority=int(priority))
                    if not admitted:
                        self.telemetry.increment("enqueue_rejected_capacity")
                        return self._decision_payload(False, reason=reason, key=key)

                resolved_ttl = self.config.default_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
                if resolved_ttl is not None and resolved_ttl < 0:
                    raise ValueError("ttl_seconds must be >= 0")

                resolved_message_id = str(message_id).strip() if message_id is not None else ""
                if not resolved_message_id:
                    resolved_message_id = f"net_{uuid.uuid4().hex}"
                expires_at = _utcnow() + timedelta(seconds=resolved_ttl) if resolved_ttl is not None and resolved_ttl > 0 else None

                msg = NetworkMessage(
                    message_id=resolved_message_id,
                    payload=payload,
                    channel=str(channel).strip() or "unknown",
                    protocol=str(protocol).strip() or "unknown",
                    fairness_key=key,
                    priority=int(priority),
                    expires_at=expires_at,
                    metadata=dict(metadata or {}),
                )

                self._messages[msg.message_id] = msg
                self._global_queue.append(msg.message_id)
                self._queue_by_key[key].append(msg.message_id)
                self._ensure_active_key_locked(key)
                self._stats["enqueued"] += 1
                self.telemetry.increment("enqueue_count")
                self.telemetry.observe("queue_depth", float(len(self._messages)))
                self.telemetry.observe("enqueue_priority", float(priority))
                return self._decision_payload(True, reason="enqueued", key=key, message_id=msg.message_id)

    def dequeue(self, max_items: int = 1) -> List[Dict[str, Any]]:
        batch_size = max(1, min(int(max_items), self.config.max_dequeue_batch))
        outputs: List[Dict[str, Any]] = []
        with self.telemetry.time_block("dequeue_latency_seconds"):
            with self._lock:
                self._prune_expired_locked()
                for _ in range(batch_size):
                    msg = self._dequeue_one_locked()
                    if msg is None:
                        break
                    outputs.append(msg.as_dict())
                    self._inflight_by_key[msg.fairness_key] += 1
                    self._stats["dequeued"] += 1

                self.telemetry.increment("dequeue_count", len(outputs))
                self.telemetry.observe("dequeue_batch_size", float(len(outputs)))
                self.telemetry.observe("queue_depth", float(len(self._messages)))
                return outputs

    def ack(self, message_id: str) -> bool:
        normalized = str(message_id).strip()
        if not normalized:
            return False
        with self._lock:
            msg = self._messages.pop(normalized, None)
            if msg is None:
                return False
            self._remove_from_indexes_locked(msg.message_id, msg.fairness_key)
            if self._inflight_by_key[msg.fairness_key] > 0:
                self._inflight_by_key[msg.fairness_key] -= 1
            self._stats["acked"] += 1
            self.telemetry.increment("ack_count")
            return True

    def nack(self, message_id: str, *, requeue: bool = True) -> bool:
        normalized = str(message_id).strip()
        if not normalized:
            return False
        with self._lock:
            msg = self._messages.get(normalized)
            if msg is None:
                return False

            if self._inflight_by_key[msg.fairness_key] > 0:
                self._inflight_by_key[msg.fairness_key] -= 1

            if requeue:
                self._queue_by_key[msg.fairness_key].appendleft(msg.message_id)
                self._global_queue.appendleft(msg.message_id)
                self._ensure_active_key_locked(msg.fairness_key)
            else:
                self._messages.pop(msg.message_id, None)
                self._remove_from_indexes_locked(msg.message_id, msg.fairness_key)
            self._stats["nacked"] += 1
            self.telemetry.increment("nack_count")
            return True

    def set_fairness_weight(self, fairness_key: str, weight: int) -> None:
        key = self._normalize_fairness_key(fairness_key)
        normalized_weight = max(1, int(weight))
        with self._lock:
            self._fairness_weights[key] = normalized_weight
            self._fairness_credits[key] = normalized_weight
            self._ensure_active_key_locked(key)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            depth_by_key = {key: len(queue) for key, queue in self._queue_by_key.items() if queue}
            inflight_by_key = {key: count for key, count in self._inflight_by_key.items() if count > 0}
            return {
                "config": {
                    "capacity": self.config.capacity,
                    "fairness_strategy": self.config.fairness_strategy,
                    "drop_strategy": self.config.drop_strategy,
                    "eviction_policy": self.config.eviction_policy,
                    "default_ttl_seconds": self.config.default_ttl_seconds,
                    "high_watermark_ratio": self.config.high_watermark_ratio,
                    "low_watermark_ratio": self.config.low_watermark_ratio,
                    "max_per_key_inflight": self.config.max_per_key_inflight,
                },
                "size": len(self._messages),
                "capacity_remaining": max(0, self.config.capacity - len(self._messages)),
                "fill_ratio": self._fill_ratio_locked(),
                "depth_by_key": depth_by_key,
                "inflight_by_key": inflight_by_key,
                "active_fairness_keys": list(self._active_fairness_keys),
                "fairness_weights": dict(self._fairness_weights),
                "stats": dict(self._stats),
                "telemetry": self.telemetry.snapshot(),
            }

    def clear(self) -> None:
        with self._lock:
            self._global_queue.clear()
            self._messages.clear()
            self._queue_by_key.clear()
            self._inflight_by_key.clear()
            self._active_fairness_keys.clear()
            self._fairness_credits.clear()
            self._stats = {key: 0 for key in self._stats.keys()}
            self.telemetry.reset()

    # ------------------------------------------------------------------ #
    # Internal methods
    # ------------------------------------------------------------------ #
    def _decision_payload(
        self,
        admitted: bool,
        *,
        reason: str,
        key: str,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "admitted": admitted,
            "reason": reason,
            "fairness_key": key,
            "message_id": message_id,
            "queue_depth": len(self._messages),
            "fill_ratio": self._fill_ratio_locked(),
            "backpressure": self._fill_ratio_locked() >= self.config.high_watermark_ratio,
        }

    def _dequeue_one_locked(self) -> Optional[NetworkMessage]:
        if not self._messages:
            return None
        if self.config.fairness_strategy == "fifo":
            return self._dequeue_fifo_locked()
        return self._dequeue_weighted_round_robin_locked()

    def _dequeue_fifo_locked(self) -> Optional[NetworkMessage]:
        while self._global_queue:
            message_id = self._global_queue.popleft()
            msg = self._messages.get(message_id)
            if msg is None:
                continue
            if msg.expired:
                self._drop_locked(msg.message_id, reason="expired")
                continue
            self._remove_from_key_queue_locked(msg.message_id, msg.fairness_key)
            return msg
        return None

    def _dequeue_weighted_round_robin_locked(self) -> Optional[NetworkMessage]:
        if not self._active_fairness_keys:
            self._rebuild_active_keys_locked()
            if not self._active_fairness_keys:
                return None

        rounds = len(self._active_fairness_keys) * 2
        for _ in range(rounds):
            key = self._active_fairness_keys[0]
            queue = self._queue_by_key.get(key)
            if not queue:
                self._active_fairness_keys.popleft()
                continue

            if key not in self._fairness_credits or self._fairness_credits[key] <= 0:
                self._fairness_credits[key] = self._fairness_weights.get(key, 1)

            message_id = queue[0]
            msg = self._messages.get(message_id)
            if msg is None:
                queue.popleft()
                continue

            if msg.expired:
                queue.popleft()
                self._drop_locked(msg.message_id, reason="expired")
                continue

            if self._fairness_credits[key] <= 0:
                self._active_fairness_keys.rotate(-1)
                continue

            queue.popleft()
            self._remove_from_global_queue_locked(message_id)
            self._fairness_credits[key] -= 1
            if queue:
                self._active_fairness_keys.rotate(-1)
            else:
                self._active_fairness_keys.popleft()
            return msg

        return None

    def _apply_backpressure_on_enqueue_locked(self, incoming_priority: int) -> tuple[bool, str]:
        self._stats["backpressure_signals"] += 1
        strategy = self.config.drop_strategy
        if strategy == "reject_new":
            return False, "rejected_capacity"
        if strategy == "drop_newest":
            self._stats["dropped"] += 1
            self.telemetry.increment("drop_count")
            return False, "dropped_newest"
        if strategy == "random_early_drop":
            probability = self._red_probability_locked()
            if self._rand.random() < probability:
                self._stats["dropped"] += 1
                self.telemetry.increment("drop_count")
                return False, "dropped_random_early"
            self._evict_one_locked(reason="capacity_random_early_admit")
            return True, "evicted_for_admit"
        if strategy == "drop_lowest_priority":
            if not self._messages:
                return False, "empty_on_drop_lowest_priority"
            lowest = min(self._messages.values(), key=lambda msg: (msg.priority, msg.enqueued_at))
            if lowest.priority > incoming_priority:
                self._stats["dropped"] += 1
                self.telemetry.increment("drop_count")
                return False, "dropped_new_lower_priority"
            self._drop_locked(lowest.message_id, reason="drop_lowest_priority")
            return True, "evicted_lower_priority"
        if strategy == "eviction_policy":
            self._evict_one_locked(reason="eviction_policy")
            return True, "evicted_policy"
        # default -> drop_oldest
        self._evict_one_locked(reason="drop_oldest")
        return True, "evicted_oldest"

    def _evict_one_locked(self, *, reason: str) -> None:
        if not self._messages:
            return
        ids = list(self._global_queue)
        if not ids:
            candidate_id = next(iter(self._messages.keys()))
            self._drop_locked(candidate_id, reason=reason)
            return
        context = EvictionContext(
            overflow=1,
            total_items=len(ids),
            metadata={"reason": reason},
        )
        idx = self._eviction_policy.select_index(ids, context=context)
        idx = max(0, min(idx, len(ids) - 1))
        candidate_id = ids[idx]
        self._drop_locked(candidate_id, reason=reason)
        self._stats["evictions"] += 1

    def _drop_locked(self, message_id: str, *, reason: str) -> None:
        msg = self._messages.pop(message_id, None)
        if msg is None:
            return
        self._remove_from_indexes_locked(message_id, msg.fairness_key)
        if reason == "expired":
            self._stats["expired"] += 1
        else:
            self._stats["dropped"] += 1
        self.telemetry.increment("drop_count")

    def _remove_from_indexes_locked(self, message_id: str, fairness_key: str) -> None:
        self._remove_from_global_queue_locked(message_id)
        self._remove_from_key_queue_locked(message_id, fairness_key)

    def _remove_from_global_queue_locked(self, message_id: str) -> None:
        if not self._global_queue:
            return
        try:
            self._global_queue.remove(message_id)
        except ValueError:
            return

    def _remove_from_key_queue_locked(self, message_id: str, fairness_key: str) -> None:
        queue = self._queue_by_key.get(fairness_key)
        if not queue:
            return
        try:
            queue.remove(message_id)
        except ValueError:
            pass
        if not queue:
            self._queue_by_key.pop(fairness_key, None)
            self._fairness_credits.pop(fairness_key, None)
            self._active_fairness_keys = deque([key for key in self._active_fairness_keys if key != fairness_key])

    def _prune_expired_locked(self) -> None:
        expired_ids = [mid for mid, message in self._messages.items() if message.expired]
        for message_id in expired_ids:
            self._drop_locked(message_id, reason="expired")

    def _normalize_fairness_key(self, fairness_key: Optional[str]) -> str:
        candidate = str(fairness_key).strip() if fairness_key is not None else ""
        return candidate or self.config.default_fairness_key

    def _ensure_active_key_locked(self, key: str) -> None:
        if key not in self._active_fairness_keys:
            self._active_fairness_keys.append(key)
        if key not in self._fairness_credits or self._fairness_credits[key] <= 0:
            self._fairness_credits[key] = self._fairness_weights.get(key, 1)

    def _rebuild_active_keys_locked(self) -> None:
        self._active_fairness_keys = deque([key for key, queue in self._queue_by_key.items() if queue])

    def _red_probability_locked(self) -> float:
        fill = self._fill_ratio_locked()
        if fill < self.config.random_early_drop_min_fill:
            return 0.0
        if fill >= 1.0:
            return 1.0
        span = 1.0 - self.config.random_early_drop_min_fill
        if span <= 0:
            return self.config.random_early_drop_max_probability
        progress = (fill - self.config.random_early_drop_min_fill) / span
        progress = max(0.0, min(1.0, progress))
        return progress * self.config.random_early_drop_max_probability

    def _fill_ratio_locked(self) -> float:
        if self.config.capacity <= 0:
            return 1.0
        return len(self._messages) / float(self.config.capacity)

    def _would_exceed_per_key_limit(self, key: str) -> bool:
        if self.config.max_per_key_inflight <= 0:
            return False
        return self._inflight_by_key[key] >= self.config.max_per_key_inflight

    def drain(self, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = self.config.max_dequeue_batch if max_items is None else max(1, int(max_items))
        outputs: List[Dict[str, Any]] = []
        while len(outputs) < limit:
            batch = self.dequeue(1)
            if not batch:
                break
            outputs.extend(batch)
        return outputs

    def extend(self, messages: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        decisions: List[Dict[str, Any]] = []
        for message in messages:
            payload = message.get("payload")
            decisions.append(
                self.enqueue(
                    payload=payload,
                    channel=str(message.get("channel", "unknown")),
                    protocol=str(message.get("protocol", "unknown")),
                    fairness_key=message.get("fairness_key"),
                    priority=int(message.get("priority", 0)),
                    ttl_seconds=message.get("ttl_seconds"),
                    metadata=message.get("metadata"),
                    message_id=message.get("message_id"),
                )
            )
        return decisions

