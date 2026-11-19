"""Read-only SLAI LanguageFrame-like -> LSTP v0.1 integration smoke test."""

from __future__ import annotations

import json
from types import SimpleNamespace

from lstp import canonical_dumps, canonical_loads

from src.integrations.lstp_adapter import packet_from_frame


def main() -> None:
    frame = SimpleNamespace(
        act_type="DIRECTIVE",
        intent="inspect",
        entities={"resource": "roof_connection_C4"},
        propositional_content="Inspect the connection without editing the model.",
        confidence=0.95,
    )
    packet = packet_from_frame(
        frame,
        source_text="Inspect connection C4. Do not edit anything.",
        packet_id="lstp_benchmark_smoke_001",
        thread_id="lstp_benchmark_smoke",
    )
    encoded = canonical_dumps(packet)
    decoded = canonical_loads(encoded, require_canonical_bytes=True)
    if not decoded.semantically_equals(packet):
        raise RuntimeError("SLAI -> LSTP semantic round-trip failed")
    if decoded.octad.permissions.mode is not None or decoded.octad.permissions.scope:
        raise RuntimeError("SLAI language frame gained forbidden execution authority")
    print(
        json.dumps(
            {
                "status": "passed",
                "integration": "src.integrations.lstp_adapter.packet_from_frame",
                "protocol_version": decoded.protocol_version,
                "canonical_bytes": len(encoded),
                "trusted_execution_authority": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
