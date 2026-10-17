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
  - Disk-backed checkpoint and cache store. **Not used by `ReaderAgent`**, which persists to `shared_memory` instead (see "Verified capabilities and limits").
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
   - Each major stage writes a checkpoint to `shared_memory` for observability. There is currently no restore/replay path, and `shared_memory` is in-process, so this state does not survive a restart.

## Config touchpoints
- Global agent config: `src/agents/base/configs/agents_config.yaml` (`reader_agent` section).
- Internal handler config: `src/agents/reader/configs/reader_config.yaml` (`reader` and `reader_memory` sections).

## Extension points
- Add specialized format loaders (e.g. `pypdf`, `python-docx`, OCR) in `parser_engine.py`.
- Add format-preserving conversion backends (e.g. LibreOffice) in `conversion_engine.py`.
- Add stronger corruption detection in `recovery_engine.py`.
- Add optional LLM-based semantic recovery in `semantic_recovery.py` behind explicit policy flags.

## Verified capabilities and limits

Verified by execution against revision `a854f17d` (audit of 2026-10-16). Where this section differs from older text above, this section reflects tested behavior.

| Extension | Read / extract | Notes |
|---|---|---|
| `.txt` `.md` | yes | UTF-8, UTF-8 BOM, UTF-16/32 (BOM required), Latin-1 fallback |
| `.csv` | yes | re-serialised; capped by `csv_max_rows` / `csv_max_columns` (warning when hit) |
| `.json` | yes | pretty-printed with sorted keys; invalid or over-nested JSON is returned raw with a warning |
| `.xml` `.html` | text only | attributes and structure are not preserved; `<script>`/`<style>` are dropped from HTML |
| `.pdf` | text via `pypdf` | encrypted PDFs rejected; no OCR; page markers are off by default (`pdf_page_break_template: ''`) |
| `.docx` | body text | tabs and line breaks kept; headers, footers, footnotes, comments, images and table structure are not extracted; `docx_max_xml_bytes` (default 32 MiB) caps decompressed size |
| `.doc` | no | listed as allowed but always rejected (`enable_doc: false`) |

Every other extension (for example `.rtf`, `.odt`, `.xlsx`, `.pptx`, `.epub`, `.eml`, images, archives) raises `UnsupportedFormatError`.

- **Output formats:** `txt`, `md`, `html`, `xml`, `json`, `csv`, all text renderings of the parsed content. There is no PDF/DOCX/XLSX writer, no format-preserving merge, and no compression or optimisation capability.
- **Recovery** works on already-parsed text (null-byte and newline cleanup, then conservative token salvage). It does not repair damaged containers: truncated PDF/DOCX/ZIP files are rejected. Reported confidence reflects character cleanliness, not readability.
- **Persistence:** `ReaderAgent` stores checkpoints, results and caches in `shared_memory` (in-process). Nothing reads checkpoints back and nothing persists shared memory automatically, so Reader state is lost on restart. Parse-cache keys use path, size and mtime, not content.
- **Timeouts:** the agent has no per-task timeout or cancellation.
- **Interpreters:** tests pass on Python 3.10 and 3.13. `pyproject.toml` still declares `<3.13`.
