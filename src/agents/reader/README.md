# Reader Agent internals

`reader/` contains the internal handlers used by `ReaderAgent` for document parsing, recovery, conversion, merge, and state persistence.

## Goals
- Keep the orchestration **fast** and **deterministic** for batch runs.
- Prefer low-cost, low-latency operations first, then escalate only when needed.
- Preserve a strict recovery policy: recover only what is detectable, avoid fabrication.

## Module map
- `parser_engine.py`
  - Normalizes supported file types into an intermediate text representation.
  - Provides file metadata useful for downstream planning.

- `conversion_engine.py`
  - Converts intermediate text representation into target format artifacts.
  - Supports merged output generation for homogeneous/heterogeneous workflows.

- `recovery_engine.py`
  - Uses a **multi-pass** strategy:
    1. low-level cleanup (null-byte/newline normalization),
    2. quality scoring,
    3. semantic fallback only if quality stays low.
  - Uses in-memory recovery cache keyed by source+content hash for repeated runs.

- `semantic_recovery.py`
  - Conservative token salvage pipeline with chunked processing.
  - Tracks corruption ratio and token-level recovery metrics.
  - Returns `[CORRUPTED_DATA]` when confidence is too low.

- `reader_memory.py`
  - Handles checkpointing (`plan`, `convert`, `merge`) for replay/debug.
  - Provides hash-keyed cache storage for parsed outputs.

- `utils/reader_error.py`
  - Reader-specific typed exceptions for parse/recover/convert/merge flows.

- `utils/config_loader.py`
  - Local config loader for `reader/configs/reader_config.yaml`.

## Performance and efficiency strategy
1. **Asynchronous fan-out** in `ReaderAgent`
   - Parsing, recovery, and conversion run through `asyncio` + `to_thread` with bounded concurrency.
2. **Cache-first execution**
   - Parser outputs are cached to skip repeat work on unchanged files.
3. **Tiered recovery**
   - Low-level cleanup is attempted before semantic recovery to minimize expensive operations.
4. **Chunked semantic processing**
   - Large text recovery avoids loading unbounded string operations into memory.
5. **Checkpoint-first observability**
   - Each major stage persists a checkpoint to aid restart/replay and reduce recomputation.

## Config touchpoints
- Global agent config: `src/agents/base/configs/agents_config.yaml` (`reader_agent` section).
- Internal handler config: `src/agents/reader/configs/reader_config.yaml` (`reader` and `reader_memory` sections).

## Extension points
- Add specialized format loaders (e.g. `pypdf`, `python-docx`, OCR) in `parser_engine.py`.
- Add format-preserving conversion backends (e.g. LibreOffice) in `conversion_engine.py`.
- Add stronger corruption detection in `recovery_engine.py`.
- Add optional LLM-based semantic recovery in `semantic_recovery.py` behind explicit policy flags.
