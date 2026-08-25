"""Relay protocol codec and parser helpers.

This module standardizes an encode/decode pipeline for relay message frames,
including header validation, checksums, protocol versioning, and optional
fragmentation/reassembly support.
"""

from __future__ import annotations

import struct
import time
import zlib

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

MAGIC = b"RLYF"
PROTOCOL_VERSION = 1
_HEADER_FORMAT = "!4sBBBIHHHI"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)


class RelayCodecError(ValueError):
    """Raised when a frame cannot be encoded, decoded, or reassembled."""


@dataclass(frozen=True, slots=True)
class RelayFrameHeader:
    """Metadata carried with every encoded relay frame."""

    version: int
    flags: int
    message_type: int
    message_id: int
    fragment_index: int
    fragment_count: int
    payload_length: int
    header_checksum: int


@dataclass(frozen=True, slots=True)
class DecodedRelayFrame:
    """Decoded frame result with validated header and payload."""

    header: RelayFrameHeader
    payload: bytes
    payload_checksum: int


class RelayFrameCodec:
    """Encodes and decodes relay frames with checksum validation."""

    FLAG_FRAGMENTED = 0b0000_0001

    @staticmethod
    def _coerce_payload(payload: bytes) -> bytes:
        try:
            return bytes(payload)
        except (TypeError, ValueError) as exc:
            raise RelayCodecError("payload must be bytes-like") from exc

    @staticmethod
    def _header_checksum(
        version: int,
        flags: int,
        message_type: int,
        message_id: int,
        fragment_index: int,
        fragment_count: int,
        payload_length: int,
    ) -> int:
        header_without_checksum = struct.pack(
            "!4sBBBIHHH",
            MAGIC,
            version,
            flags,
            message_type,
            message_id,
            fragment_index,
            fragment_count,
            payload_length,
        )
        return zlib.crc32(header_without_checksum) & 0xFFFFFFFF

    @classmethod
    def encode(
        cls,
        payload: bytes,
        *,
        message_type: int = 0,
        message_id: Optional[int] = None,
        version: int = PROTOCOL_VERSION,
        max_fragment_payload: int = 1024,
    ) -> List[bytes]:
        """Encode payload bytes into one or more protocol frames."""
        payload_bytes = cls._coerce_payload(payload)

        if not (0 <= int(message_type) <= 255):
            raise RelayCodecError("message_type must be between 0 and 255")
        if not (1 <= int(version) <= 255):
            raise RelayCodecError("version must be between 1 and 255")

        max_fragment_payload = int(max_fragment_payload)
        if max_fragment_payload < 1:
            raise RelayCodecError("max_fragment_payload must be >= 1")

        resolved_message_id = (
            int(time.time_ns() & 0xFFFFFFFF) if message_id is None else int(message_id)
        )
        if not (0 <= resolved_message_id <= 0xFFFFFFFF):
            raise RelayCodecError("message_id must fit within uint32")

        chunks = [
            payload_bytes[i : i + max_fragment_payload]
            for i in range(0, len(payload_bytes), max_fragment_payload)
        ] or [b""]

        fragment_count = len(chunks)
        if fragment_count > 0xFFFF:
            raise RelayCodecError("fragment count exceeds uint16 maximum")

        flags = cls.FLAG_FRAGMENTED if fragment_count > 1 else 0
        encoded: List[bytes] = []

        for fragment_index, fragment_payload in enumerate(chunks):
            if len(fragment_payload) > 0xFFFF:
                raise RelayCodecError("fragment payload exceeds uint16 maximum")

            header_checksum = cls._header_checksum(
                version,
                flags,
                int(message_type),
                resolved_message_id,
                fragment_index,
                fragment_count,
                len(fragment_payload),
            )
            payload_checksum = zlib.crc32(fragment_payload) & 0xFFFFFFFF

            header = struct.pack(
                _HEADER_FORMAT,
                MAGIC,
                version,
                flags,
                int(message_type),
                resolved_message_id,
                fragment_index,
                fragment_count,
                len(fragment_payload),
                header_checksum,
            )
            encoded.append(header + struct.pack("!I", payload_checksum) + fragment_payload)

        return encoded

    @classmethod
    def decode_frame(cls, frame: bytes) -> DecodedRelayFrame:
        """Decode and validate a single relay frame."""
        frame_bytes = cls._coerce_payload(frame)

        if len(frame_bytes) < _HEADER_SIZE + 4:
            raise RelayCodecError("frame is too short to contain header and checksums")

        (
            magic,
            version,
            flags,
            message_type,
            message_id,
            fragment_index,
            fragment_count,
            payload_length,
            observed_header_checksum,
        ) = struct.unpack(_HEADER_FORMAT, frame_bytes[:_HEADER_SIZE])

        if magic != MAGIC:
            raise RelayCodecError("invalid frame magic")
        if version < 1:
            raise RelayCodecError("invalid protocol version")

        expected_header_checksum = cls._header_checksum(
            version,
            flags,
            message_type,
            message_id,
            fragment_index,
            fragment_count,
            payload_length,
        )
        if observed_header_checksum != expected_header_checksum:
            raise RelayCodecError("header checksum mismatch")

        payload_checksum_offset = _HEADER_SIZE
        payload_start = payload_checksum_offset + 4
        payload_end = payload_start + payload_length
        if len(frame_bytes) != payload_end:
            raise RelayCodecError("payload length does not match frame size")

        (observed_payload_checksum,) = struct.unpack(
            "!I", frame_bytes[payload_checksum_offset:payload_start]
        )
        payload = frame_bytes[payload_start:payload_end]

        expected_payload_checksum = zlib.crc32(payload) & 0xFFFFFFFF
        if observed_payload_checksum != expected_payload_checksum:
            raise RelayCodecError("payload checksum mismatch")

        header = RelayFrameHeader(
            version=version,
            flags=flags,
            message_type=message_type,
            message_id=message_id,
            fragment_index=fragment_index,
            fragment_count=fragment_count,
            payload_length=payload_length,
            header_checksum=observed_header_checksum,
        )
        return DecodedRelayFrame(
            header=header,
            payload=payload,
            payload_checksum=observed_payload_checksum,
        )

    @classmethod
    def decode_stream(cls, data: bytes) -> Tuple[List[DecodedRelayFrame], bytes]:
        """Decode as many complete frames as possible from a byte stream."""
        buffer = cls._coerce_payload(data)
        cursor = 0
        frames: List[DecodedRelayFrame] = []

        while len(buffer) - cursor >= _HEADER_SIZE + 4:
            if buffer[cursor : cursor + 4] != MAGIC:
                magic_index = buffer.find(MAGIC, cursor + 1)
                if magic_index == -1:
                    return frames, b""
                cursor = magic_index
                continue

            header_slice = buffer[cursor : cursor + _HEADER_SIZE]
            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                payload_length,
                _,
            ) = struct.unpack(_HEADER_FORMAT, header_slice)
            frame_length = _HEADER_SIZE + 4 + payload_length
            if len(buffer) - cursor < frame_length:
                break

            frame = buffer[cursor : cursor + frame_length]
            frames.append(cls.decode_frame(frame))
            cursor += frame_length

        return frames, buffer[cursor:]


class RelayReassembler:
    """Reassembles fragmented payloads keyed by message id."""

    def __init__(self, *, expiration_seconds: float = 120.0):
        self.expiration_seconds = float(expiration_seconds)
        self._buffers: Dict[int, Dict[str, object]] = {}

    def _expire(self) -> None:
        now = time.time()
        expired_ids = [
            message_id
            for message_id, state in self._buffers.items()
            if now - float(state["updated_at"]) > self.expiration_seconds
        ]
        for message_id in expired_ids:
            self._buffers.pop(message_id, None)

    def add(self, frame: DecodedRelayFrame) -> Optional[bytes]:
        """Add one decoded frame; return payload once all fragments are present."""
        self._expire()

        header = frame.header
        if header.fragment_count <= 1:
            return frame.payload
        if header.fragment_index >= header.fragment_count:
            raise RelayCodecError("fragment index is out of bounds")

        state = self._buffers.setdefault(
            header.message_id,
            {
                "fragment_count": header.fragment_count,
                "parts": {},
                "updated_at": time.time(),
            },
        )

        if int(state["fragment_count"]) != header.fragment_count:
            raise RelayCodecError("fragment count mismatch for message_id")

        parts: Dict[int, bytes] = state["parts"]  # type: ignore[assignment]
        parts[header.fragment_index] = frame.payload
        state["updated_at"] = time.time()

        if len(parts) < header.fragment_count:
            return None

        assembled = b"".join(parts[idx] for idx in range(header.fragment_count))
        self._buffers.pop(header.message_id, None)
        return assembled

    def add_many(self, frames: Iterable[DecodedRelayFrame]) -> List[bytes]:
        """Add many frames and return any fully reassembled payloads."""
        completed: List[bytes] = []
        for frame in frames:
            payload = self.add(frame)
            if payload is not None:
                completed.append(payload)
        return completed
