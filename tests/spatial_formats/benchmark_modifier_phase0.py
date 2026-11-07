"""Reproducible Phase 0/1 native spatial-format performance baselines.

Run: python -m tests.spatial_formats.benchmark_modifier_phase0 [size ...]
     python -m tests.spatial_formats.benchmark_modifier_phase0 --phase1 [size ...]
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


def measure_phase1(kind, size):
    """Measure real modifier evaluation and one dependency-aware invalidation."""
    if kind == "sgeo":
        doc = SGEODocument.new("CSG benchmark")
        target = doc.add_box(size=(2, 2, 2))
        operand = doc.add_box(size=(2, 2, 2), center=(1, 0, 0))
        for i in range(size):
            doc.add_box(size=(1, 1, 1), center=(10 + 2 * i, 0, 0))
        kwargs = {"kind": "sgeo.boolean", "parameters": {"operation": "intersect"},
                  "evaluation_policy": {"mode": "exact"}}
    else:
        doc = SDRWDocument.new("2D region benchmark")
        target = doc.add_polyline([(0, 0), (4, 0), (4, 4), (0, 4)], closed=True)
        operand = doc.add_polyline([(1, 0), (5, 0), (5, 4), (1, 4)], closed=True)
        for i in range(size):
            doc.add_line((10 + 2 * i, 0), (11 + 2 * i, 0))
        kwargs = {"kind": "sdrw.boolean", "parameters": {"operation": "intersect"},
                  "evaluation_policy": {"mode": "tolerance_controlled", "absolute": 1e-6}}
    tracer = {"format": kind, "unrelated_entities": size}
    tracemalloc.start()
    tracer["add_modifier"] = timed(lambda: doc.add_modifier(target, operands=[operand], **kwargs), rounds=1)
    tracer["evaluate_first"] = timed(doc.evaluate_modifiers, rounds=1)
    tracer["evaluate_cached"] = timed(doc.evaluate_modifiers)
    if kind == "sgeo":
        doc.set_geometry(operand, {"kind": "box", "center": [0.5, 0, 0], "size": [2, 2, 2]})
    else:
        doc.update(operand, geometry={"points": [[0.5, 0], [4.5, 0], [4.5, 4], [0.5, 4]],
                                      "closed": True})
    tracer["evaluate_invalidated"] = timed(doc.evaluate_modifiers, rounds=1)
    tracer["peak_tracemalloc_bytes"] = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return tracer


def main():
    args = sys.argv[1:]
    phase1 = bool(args and args[0] == "--phase1")
    if phase1:
        args = args[1:]
    sizes = [int(arg) for arg in args] or [100, 500, 1000]
    if any(n < 1 or n > 10_000 for n in sizes):
        raise SystemExit("sizes must be between 1 and 10000")
    output = {"python": sys.version.split()[0], "platform": platform.platform(),
              "measurements": [(measure_phase1 if phase1 else measure)(kind, size)
                               for kind in ("sgeo", "sdrw") for size in sizes]}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
