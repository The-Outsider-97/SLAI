"""Reproducible local Phase 0 spatial-format performance baseline.

Run: python -m tests.spatial_formats.benchmark_modifier_phase0 [size ...]
This reports observations; it deliberately imposes no machine-dependent timing
thresholds.  Keep the JSON output alongside release evidence when benchmarking.
"""
from __future__ import annotations

import gc
import json
import platform
import sys
import tracemalloc
from statistics import median
from time import perf_counter

from src.utils.sgeo import SGEODocument
from src.utils.sdrw import SDRWDocument


def timed(fn, *, rounds=3):
    samples = []
    for _ in range(rounds):
        gc.collect()
        start = perf_counter()
        fn()
        samples.append(round((perf_counter() - start) * 1000, 3))
    return {"median_ms": round(median(samples), 3), "runs_ms": samples}


def measure(kind, size):
    doc = SGEODocument.new("benchmark") if kind == "sgeo" else SDRWDocument.new("benchmark")
    for i in range(size):
        if kind == "sgeo":
            oid = doc.add_box(width=1, depth=1, height=1, position=(i * 2, 0, 0))
        else:
            oid = doc.add_line((i * 2, 0), (i * 2 + 1, 0))
        if i == 0:
            target = oid
    raw = doc.dumps()
    measurements = {
        "entities": size, "bytes": len(raw), "format": kind,
        "strict_validation": timed(lambda: doc.validate("strict", raise_on_error=True)),
        "deserialize": timed(lambda: type(doc).from_bytes(raw)),
        "serialize": timed(doc.dumps),
    }
    tracemalloc.start()
    start = perf_counter()
    doc.add_modifier(target)
    measurements["modifier_transaction_ms"] = round((perf_counter() - start) * 1000, 3)
    measurements["modifier_evaluation_first"] = timed(doc.evaluate_modifiers, rounds=1)
    measurements["modifier_evaluation_cached"] = timed(doc.evaluate_modifiers)
    measurements["peak_tracemalloc_bytes"] = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return measurements


def main():
    sizes = [int(arg) for arg in sys.argv[1:]] or [100, 500, 1000]
    if any(n < 1 or n > 10_000 for n in sizes):
        raise SystemExit("sizes must be between 1 and 10000")
    output = {"python": sys.version.split()[0], "platform": platform.platform(),
              "measurements": [measure(kind, size) for kind in ("sgeo", "sdrw") for size in sizes]}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
