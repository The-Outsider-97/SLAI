"""Transport abstraction layer for radio and remote communication adapters.

This module provides a consistent API to connect, send, receive, retry failed
transmissions, and inspect channel state across diverse communication mediums:
LoRa, serial links, mesh peers, LTE modems, and SATCOM terminals.
"""

from __future__ import annotations

import copy
import random
import threading
import time

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Deque, Dict, Iterable, List, Mapping, Optional, Type

from .utils.config_loader import get_config_section
from .utils.functions_error import TransportChannelError, TransportError, TransportRetryExhausted
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Transport Service")
printer = PrettyPrinter

_UNSET = object()
_DEFAULT_MAX_RECEIVE_BUFFER = 1024
_DEFAULT_RECEIVE_POLL_SECONDS = 0.1
_DEFAULT_MAX_BACKOFF_SECONDS = 30.0


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _require_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_optional_endpoint(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    normalized = value.strip()
    return normalized or None


def _coerce_positive_int(value: Any, field_name: str, *, minimum: int = 1) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc

    if converted < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return converted


class TransportType(Enum):
    """Supported transport adapter types."""

    LORA = "lora"
    SERIAL = "serial"
    MESH = "mesh"
    LTE = "lte"
    SATCOM = "satcom"

    @classmethod
    def from_value(cls, value: str) -> "TransportType":
        """Parse a transport type from a config or API value."""
        normalized = _require_non_empty_string(value, "transport type").lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            valid = ", ".join(item.value for item in cls)
            raise TransportError(
                f"Unsupported transport type '{normalized}'. Expected one of: {valid}"
            ) from exc


class ChannelStatus(Enum):
    """Current adapter connectivity/health status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass(slots=True)
class ChannelState:
    """Snapshot of channel state and signal characteristics."""

    status: ChannelStatus = ChannelStatus.DISCONNECTED
    signal_strength_dbm: Optional[float] = None
    quality: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.quality = _clamp_ratio(self.quality)
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class TransportPacket:
    """Portable transport payload representation."""

    payload: bytes
    destination: Optional[str] = None
    source: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.payload = bytes(self.payload)
        except (TypeError, ValueError) as exc:
            raise TypeError("payload must be bytes-like") from exc

        self.destination = _normalize_optional_endpoint(self.destination, "destination")
        self.source = _normalize_optional_endpoint(self.source, "source")
        self.metadata = dict(self.metadata)


class TransportAdapter(ABC):
    """Base transport adapter interface."""

    def __init__(
        self,
        name: str,
        reliability: float = 0.97,
        base_latency_ms: int = 120,
        max_receive_buffer: int = _DEFAULT_MAX_RECEIVE_BUFFER,
        receive_poll_interval_seconds: float = _DEFAULT_RECEIVE_POLL_SECONDS,
        random_generator: Optional[random.Random] = None,
    ):
        self.name = _require_non_empty_string(name, "name")
        self.reliability = _clamp_ratio(reliability)
        self.base_latency_ms = _coerce_positive_int(base_latency_ms, "base_latency_ms")
        self.max_receive_buffer = _coerce_positive_int(
            max_receive_buffer,
            "max_receive_buffer",
        )
        self.receive_poll_interval_seconds = max(
            0.01,
            float(receive_poll_interval_seconds),
        )
        self._connected = False
        self._state = ChannelState()
        self._rx_queue: Deque[TransportPacket] = deque(maxlen=self.max_receive_buffer)
        self._lock = threading.RLock()
        self._rx_condition = threading.Condition(self._lock)
        self._rng = random_generator or random.Random()

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def state(self) -> ChannelState:
        with self._lock:
            return ChannelState(
                status=self._state.status,
                signal_strength_dbm=self._state.signal_strength_dbm,
                quality=self._state.quality,
                metadata=copy.deepcopy(self._state.metadata),
                updated_at=self._state.updated_at,
            )

    def _set_state(
        self,
        status: ChannelStatus,
        signal_strength_dbm: Any = _UNSET,
        quality: Any = _UNSET,
        metadata: Optional[Mapping[str, Any]] = None,
        clear_metadata_keys: Optional[Iterable[str]] = None,
    ) -> None:
        with self._lock:
            self._state.status = status
            if signal_strength_dbm is not _UNSET:
                self._state.signal_strength_dbm = signal_strength_dbm
            if quality is not _UNSET:
                self._state.quality = _clamp_ratio(quality)
            if clear_metadata_keys:
                for key in clear_metadata_keys:
                    self._state.metadata.pop(key, None)
            if metadata:
                self._state.metadata.update(copy.deepcopy(dict(metadata)))
            self._state.updated_at = time.time()

    def connect(self) -> None:
        """Connect adapter and move the channel to READY."""
        with self._lock:
            if self._connected:
                return
            self._set_state(
                ChannelStatus.CONNECTING,
                clear_metadata_keys=("reason", "last_error", "disconnect_error"),
            )

        try:
            self._connect_impl()
        except TransportError as exc:
            self._set_state(
                ChannelStatus.ERROR,
                signal_strength_dbm=None,
                quality=0.0,
                metadata={"reason": str(exc)},
            )
            raise TransportChannelError(
                f"Adapter '{self.name}' failed to connect: {exc}"
            ) from exc
        except Exception as exc:
            self._set_state(
                ChannelStatus.ERROR,
                signal_strength_dbm=None,
                quality=0.0,
                metadata={"reason": str(exc)},
            )
            raise TransportChannelError(
                f"Adapter '{self.name}' failed to connect: {exc}"
            ) from exc

        with self._lock:
            self._connected = True
            self._set_state(
                ChannelStatus.READY,
                signal_strength_dbm=self._sample_signal_dbm(),
                quality=self.reliability,
                clear_metadata_keys=("reason", "last_error", "disconnect_error"),
            )
        logger.info(f"Connected transport adapter '{self.name}'")

    def disconnect(self) -> None:
        """Disconnect adapter and clear transient state."""
        with self._lock:
            was_connected = self._connected
            self._connected = False

        if not was_connected:
            with self._rx_condition:
                self._rx_queue.clear()
                self._set_state(
                    ChannelStatus.DISCONNECTED,
                    signal_strength_dbm=None,
                    quality=0.0,
                    clear_metadata_keys=("reason", "last_error", "disconnect_error"),
                )
                self._rx_condition.notify_all()
            return

        disconnect_error: Optional[Exception] = None
        try:
            self._disconnect_impl()
        except Exception as exc:
            disconnect_error = exc
        finally:
            with self._rx_condition:
                self._rx_queue.clear()
                self._set_state(
                    ChannelStatus.DISCONNECTED,
                    signal_strength_dbm=None,
                    quality=0.0,
                    metadata={"disconnect_error": str(disconnect_error)}
                    if disconnect_error
                    else None,
                    clear_metadata_keys=("reason", "last_error"),
                )
                self._rx_condition.notify_all()

        if disconnect_error is not None:
            raise TransportChannelError(
                f"Adapter '{self.name}' failed to disconnect cleanly: {disconnect_error}"
            ) from disconnect_error

        logger.info(f"Disconnected transport adapter '{self.name}'")

    def send(self, packet: TransportPacket) -> str:
        """Transmit packet and return a transport-generated message ID."""
        packet = self._validate_packet(packet, require_payload=True)

        with self._lock:
            if not self._connected:
                raise TransportChannelError(f"Adapter '{self.name}' is not connected")
            self._set_state(
                ChannelStatus.READY,
                signal_strength_dbm=self._sample_signal_dbm(),
                clear_metadata_keys=("last_error",),
            )

        try:
            transmission_id = self._send_impl(packet)
        except TransportError as exc:
            self._mark_send_failure(str(exc))
            raise
        except Exception as exc:
            self._mark_send_failure(str(exc))
            raise TransportError(
                f"Adapter '{self.name}' failed to send packet: {exc}"
            ) from exc

        if not transmission_id:
            reason = "Transport adapter returned empty transmission ID"
            self._mark_send_failure(reason)
            raise TransportError(reason)
        return transmission_id

    def receive(self, timeout_seconds: float = 0.0) -> Optional[TransportPacket]:
        """Receive one packet from buffer or adapter source."""
        timeout = max(0.0, float(timeout_seconds))
        deadline = time.monotonic() + timeout

        while True:
            with self._rx_condition:
                if not self._connected:
                    raise TransportChannelError(f"Adapter '{self.name}' is not connected")
                if self._rx_queue:
                    return self._rx_queue.popleft()

            try:
                packet = self._receive_impl()
            except TransportError as exc:
                self._set_state(
                    ChannelStatus.DEGRADED,
                    quality=max(0.0, self.state.quality - 0.1),
                    metadata={"last_error": str(exc)},
                )
                raise
            except Exception as exc:
                self._set_state(
                    ChannelStatus.ERROR,
                    quality=0.0,
                    metadata={"reason": str(exc)},
                )
                raise TransportError(
                    f"Adapter '{self.name}' failed while receiving: {exc}"
                ) from exc

            if packet is not None:
                packet = self._validate_packet(packet)
                self._set_state(
                    ChannelStatus.READY,
                    signal_strength_dbm=self._sample_signal_dbm(),
                    clear_metadata_keys=("last_error", "reason"),
                )
                return packet

            if timeout <= 0.0:
                return None

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None

            with self._rx_condition:
                if not self._connected:
                    raise TransportChannelError(f"Adapter '{self.name}' is not connected")
                if self._rx_queue:
                    continue
                self._rx_condition.wait(
                    timeout=min(remaining, self.receive_poll_interval_seconds)
                )

    def send_with_retry(
        self,
        packet: TransportPacket,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        jitter_ratio: float = 0.1,
    ) -> str:
        """Send packet with bounded retry and jittered exponential backoff."""
        packet = self._validate_packet(packet, require_payload=True)
        attempts = _coerce_positive_int(max_attempts, "max_attempts")
        sleep_base = max(0.0, float(backoff_seconds))
        jitter = _clamp_ratio(jitter_ratio)
        errors: List[str] = []
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                tx_id = self.send(packet)
                self._set_state(
                    ChannelStatus.READY,
                    quality=self.reliability,
                    clear_metadata_keys=("last_error", "reason"),
                )
                return tx_id
            except TransportChannelError:
                raise
            except TransportError as exc:
                last_error = exc
                reason = str(exc)
                errors.append(reason)
                self._set_state(
                    ChannelStatus.DEGRADED,
                    quality=max(0.0, self.state.quality - 0.1),
                    metadata={"last_error": reason, "retry_attempt": attempt},
                )
                logger.warning(
                    f"Send attempt {attempt}/{attempts} failed on '{self.name}': {reason}"
                )

                if attempt >= attempts:
                    break
                pause = self._calculate_backoff(
                    attempt=attempt,
                    backoff_seconds=sleep_base,
                    jitter_ratio=jitter,
                )
                time.sleep(pause)

        summary = " | ".join(errors) if errors else "unknown transport error"
        raise TransportRetryExhausted(
            f"Adapter '{self.name}' failed after {attempts} attempts: {summary}"
        ) from last_error

    def inject_received(self, packet: TransportPacket) -> None:
        """Inject packet into receive queue (for testing or simulation)."""
        packet = self._validate_packet(packet)
        with self._rx_condition:
            self._rx_queue.append(packet)
            self._rx_condition.notify()

    def _mark_send_failure(self, reason: str) -> None:
        self._set_state(
            ChannelStatus.DEGRADED,
            quality=max(0.0, self.state.quality - 0.1),
            metadata={"last_error": reason},
        )

    def _calculate_backoff(
        self,
        *,
        attempt: int,
        backoff_seconds: float,
        jitter_ratio: float,
    ) -> float:
        pause = min(
            _DEFAULT_MAX_BACKOFF_SECONDS,
            backoff_seconds * (2 ** max(0, attempt - 1)),
        )
        if pause <= 0.0 or jitter_ratio <= 0.0:
            return pause
        pause += self._rng.uniform(0.0, pause * jitter_ratio)
        return min(pause, _DEFAULT_MAX_BACKOFF_SECONDS)

    def _validate_packet(
        self,
        packet: TransportPacket,
        *,
        require_payload: bool = False,
    ) -> TransportPacket:
        if not isinstance(packet, TransportPacket):
            raise TypeError("packet must be a TransportPacket instance")
        if require_payload and not packet.payload:
            raise TransportError("Cannot send empty payload")
        return packet

    def _sample_signal_dbm(self) -> float:
        """Provide a synthetic signal sample used for channel-state snapshots."""
        return round(self._rng.uniform(-122.0, -62.0), 2)

    @abstractmethod
    def _connect_impl(self) -> None:
        """Perform adapter-specific connection logic."""

    @abstractmethod
    def _disconnect_impl(self) -> None:
        """Perform adapter-specific disconnect logic."""

    @abstractmethod
    def _send_impl(self, packet: TransportPacket) -> str:
        """Perform adapter-specific send logic and return a transmission ID."""

    @abstractmethod
    def _receive_impl(self) -> Optional[TransportPacket]:
        """Perform adapter-specific receive polling."""


class _SimulatedAdapter(TransportAdapter):
    """Shared simulation behavior for transport adapters without hardware drivers."""

    transport_type: ClassVar[TransportType]
    default_name: ClassVar[str]
    default_reliability: ClassVar[float]
    default_latency_ms: ClassVar[int]
    signal_range_dbm: ClassVar[tuple[float, float]] = (-122.0, -62.0)

    def __init__(
        self,
        name: Optional[str] = None,
        reliability: Optional[float] = None,
        base_latency_ms: Optional[int] = None,
        max_receive_buffer: int = _DEFAULT_MAX_RECEIVE_BUFFER,
        **kwargs: Any,
    ):
        super().__init__(
            name=name or self.default_name,
            reliability=self.default_reliability if reliability is None else reliability,
            base_latency_ms=self.default_latency_ms if base_latency_ms is None else base_latency_ms,
            max_receive_buffer=max_receive_buffer,
            **kwargs,
        )
        self._sequence = 0

    def _connect_impl(self) -> None:
        time.sleep(self.base_latency_ms / 1000.0)

    def _disconnect_impl(self) -> None:
        time.sleep(min(0.05, self.base_latency_ms / 4000.0))

    def _send_impl(self, packet: TransportPacket) -> str:
        time.sleep(self.base_latency_ms / 1000.0)
        if self._rng.random() > self.reliability:
            raise TransportError(f"{self.transport_type.value} transmission link error")

        with self._lock:
            self._sequence += 1
            sequence = self._sequence

        tx_id = f"{self.transport_type.value}-{time.time_ns()}-{sequence}"
        ack_packet = TransportPacket(
            payload=b"ACK",
            destination=packet.source,
            source=packet.destination,
            metadata={
                "tx_id": tx_id,
                "transport": self.transport_type.value,
                "ack": True,
                "simulated": True,
            },
        )
        self.inject_received(ack_packet)
        return tx_id

    def _receive_impl(self) -> Optional[TransportPacket]:
        return None

    def _sample_signal_dbm(self) -> float:
        low, high = self.signal_range_dbm
        return round(self._rng.uniform(low, high), 2)


class LoRaAdapter(_SimulatedAdapter):
    """LoRa-specific adapter simulation with long-range, low-throughput defaults."""

    transport_type = TransportType.LORA
    default_name = "lora"
    default_reliability = 0.92
    default_latency_ms = 420
    signal_range_dbm = (-126.0, -95.0)


class SerialAdapter(_SimulatedAdapter):
    """Serial adapter simulation with low-latency local-link defaults."""

    transport_type = TransportType.SERIAL
    default_name = "serial"
    default_reliability = 0.995
    default_latency_ms = 20
    signal_range_dbm = (-45.0, -30.0)


class MeshAdapter(_SimulatedAdapter):
    """Mesh adapter simulation with multi-hop and moderate reliability defaults."""

    transport_type = TransportType.MESH
    default_name = "mesh"
    default_reliability = 0.90
    default_latency_ms = 260
    signal_range_dbm = (-100.0, -68.0)


class LTEAdapter(_SimulatedAdapter):
    """LTE adapter simulation with cellular latency and reliability defaults."""

    transport_type = TransportType.LTE
    default_name = "lte"
    default_reliability = 0.97
    default_latency_ms = 120
    signal_range_dbm = (-95.0, -65.0)


class SATCOMAdapter(_SimulatedAdapter):
    """SATCOM adapter simulation with higher latency and stronger retry needs."""

    transport_type = TransportType.SATCOM
    default_name = "satcom"
    default_reliability = 0.88
    default_latency_ms = 780
    signal_range_dbm = (-118.0, -90.0)


class TransportService:
    """Coordinator that manages multiple transport adapters through one interface."""

    _ADAPTERS: ClassVar[Dict[TransportType, Type[TransportAdapter]]] = {
        TransportType.LORA: LoRaAdapter,
        TransportType.SERIAL: SerialAdapter,
        TransportType.MESH: MeshAdapter,
        TransportType.LTE: LTEAdapter,
        TransportType.SATCOM: SATCOMAdapter,
    }

    def __init__(self, adapters: Optional[Mapping[str, TransportAdapter]] = None):
        self._adapters: Dict[str, TransportAdapter] = {}
        self._lock = threading.RLock()

        for alias, adapter in (adapters or {}).items():
            self.register_adapter(alias, adapter)

    @classmethod
    def from_config(cls) -> "TransportService":
        """Create a transport service using the functions config section 'transport'."""
        config = get_config_section("transport") or {}
        adapter_configs = config.get("adapters", [])
        if adapter_configs is None:
            adapter_configs = []
        if not isinstance(adapter_configs, list):
            raise TransportError("transport.adapters must be a list of adapter configurations")

        service = cls()
        for index, entry in enumerate(adapter_configs, start=1):
            if not isinstance(entry, Mapping):
                raise TransportError(
                    f"transport.adapters[{index}] must be a mapping of adapter settings"
                )

            transport_type = TransportType.from_value(str(entry.get("type", "")))
            adapter_cls = cls._ADAPTERS[transport_type]

            adapter_name = str(entry.get("name", transport_type.value)).strip() or transport_type.value
            reliability = entry.get("reliability", getattr(adapter_cls, "default_reliability", 0.97))
            base_latency_ms = entry.get(
                "base_latency_ms",
                getattr(adapter_cls, "default_latency_ms", 120),
            )
            max_receive_buffer = entry.get("max_receive_buffer", _DEFAULT_MAX_RECEIVE_BUFFER)

            adapter = adapter_cls(
                name=adapter_name,
                reliability=float(reliability),
                base_latency_ms=int(base_latency_ms),
                max_receive_buffer=int(max_receive_buffer),
            )
            service.register_adapter(adapter_name, adapter)

        return service

    def register_adapter(
        self,
        alias: str,
        adapter: TransportAdapter,
        *,
        replace: bool = False,
    ) -> None:
        alias = _require_non_empty_string(alias, "alias")
        if not isinstance(adapter, TransportAdapter):
            raise TypeError("adapter must be a TransportAdapter instance")

        with self._lock:
            if alias in self._adapters and not replace:
                raise TransportChannelError(f"Adapter alias already registered: {alias}")
            self._adapters[alias] = adapter

    def unregister_adapter(self, alias: str, *, disconnect: bool = False) -> TransportAdapter:
        alias = _require_non_empty_string(alias, "alias")
        with self._lock:
            adapter = self._adapters.pop(alias, None)
        if adapter is None:
            raise TransportChannelError(f"Unknown adapter alias: {alias}")
        if disconnect and adapter.is_connected:
            adapter.disconnect()
        return adapter

    def list_adapters(self) -> List[str]:
        with self._lock:
            return sorted(self._adapters.keys())

    def connect(self, alias: str) -> None:
        self._resolve(alias).connect()

    def connect_all(self) -> None:
        for alias in self.list_adapters():
            self.connect(alias)

    def disconnect(self, alias: str) -> None:
        self._resolve(alias).disconnect()

    def disconnect_all(self) -> None:
        for alias in self.list_adapters():
            self.disconnect(alias)

    def send(self, alias: str, packet: TransportPacket) -> str:
        return self._resolve(alias).send(packet)

    def receive(self, alias: str, timeout_seconds: float = 0.0) -> Optional[TransportPacket]:
        return self._resolve(alias).receive(timeout_seconds=timeout_seconds)

    def send_with_retry(
        self,
        alias: str,
        packet: TransportPacket,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        jitter_ratio: float = 0.1,
    ) -> str:
        return self._resolve(alias).send_with_retry(
            packet,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            jitter_ratio=jitter_ratio,
        )

    def channel_state(self, alias: str) -> ChannelState:
        return self._resolve(alias).state

    def channel_states(self) -> Dict[str, ChannelState]:
        return {alias: self.channel_state(alias) for alias in self.list_adapters()}

    def _resolve(self, alias: str) -> TransportAdapter:
        normalized_alias = _require_non_empty_string(alias, "alias")
        with self._lock:
            adapter = self._adapters.get(normalized_alias)
        if adapter is None:
            raise TransportChannelError(f"Unknown adapter alias: {normalized_alias}")
        return adapter


__all__ = [
    "TransportType",
    "ChannelStatus",
    "ChannelState",
    "TransportPacket",
    "TransportAdapter",
    "LoRaAdapter",
    "SerialAdapter",
    "MeshAdapter",
    "LTEAdapter",
    "SATCOMAdapter",
    "TransportService",
]


if __name__ == "__main__":
    print("\n=== Running Transport Service ===\n")
    printer.status("TEST", "Starting Transport Service tests", "info")

    transport = TransportService(
        {
            "field-lora": LoRaAdapter(name="field-lora"),
            "ops-lte": LTEAdapter(name="ops-lte"),
            "backup-sat": SATCOMAdapter(name="backup-sat"),
        }
    )

    packet = TransportPacket(
        payload=b"mission:heartbeat",
        destination="gateway-A",
        source="node-17",
        metadata={"priority": "normal"},
    )

    for alias in transport.list_adapters():
        transport.connect(alias)
        tx_id = transport.send_with_retry(alias, packet, max_attempts=4, backoff_seconds=0.1)
        ack = transport.receive(alias, timeout_seconds=1.0)
        print(alias, tx_id, ack.metadata if ack else None, transport.channel_state(alias).status.value)
        transport.disconnect(alias)

    print("\n=== Successfully ran Transport Service ===\n")
