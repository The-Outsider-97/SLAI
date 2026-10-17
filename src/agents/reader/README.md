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

- `detection_engine.py` (foundation)
  - Content-based format detection (signature, container members, text sniffing) with a format registry that records which operations each format supports.
  - Never trusts the extension alone; reports extension mismatch, damaged containers and encrypted Office files.

- `integrity_engine.py` (foundation)
  - The shared expansion guard (`ExpansionGuard`) used by every container reader, plus file and artifact validation (ZIP CRCs, package members, PDF trailer/page tree, JSON/XML/CSV/JSONL well-formedness, tar member paths).
  - `deep=False` reads headers and trailers only; checks that did not run are reported as skipped, never as passed.

- `job_engine.py` (foundation)
  - Durable, append-only job journal on top of `ReaderMemory` (separate store, `exact=True` records). Resume skips finished items whose artifacts still match their recorded size and SHA-256.

- `compression_engine.py` (capability)
  - `package` (zip, tar, tar.gz, tar.xz, tar.bz2), `compress_file` (gz, bz2, xz), `optimize` (lossless zip-package/PDF/PNG recompression; lossy JPEG only with `lossy: true`), `expand` (safe extraction).
  - Always writes a new file plus a report (`lossless`, `verified`, `verified_by`); an optimization that is not smaller returns `no_gain` and keeps the original.

- `salvage_engine.py` (capability)
  - Recovers damaged **containers at byte level** (truncated or corrupt ZIP packages, PDFs, gz/bz2/xz streams, tar, and truncated JSON/JSONL/CSV/XML) and writes a repaired copy that the normal `ParserEngine` can read.
  - Boundary with `RecoveryEngine`: salvage repairs bytes before parsing; recovery repairs text after parsing. They do not overlap.

- `utils/reader_error.py`
  - Reader-specific typed exceptions for parse/recover/convert/merge flows, plus `IntegrityError`, `CompressionError`, `SalvageError`, `JobError`, `JobNotFoundError` and `JobStateError`.

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
- Internal handler config: `src/agents/reader/configs/reader_config.yaml` (`reader` and `reader_memory` sections, plus `detection_engine`, `integrity_engine`, `job_engine`, `compression_engine` and `salvage_engine`). Each engine warns once about unknown keys in its section.
- Agent-level opt-in defaults for the new engines (`detect_files`, `verify_outputs`, `salvage_on_parse_failure`, `durable_jobs`) live in the `reader_agent` section and default to `false`.

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

- **Output formats:** `txt`, `md`, `html`, `xml`, `json`, `csv`, all text renderings of the parsed content. There is no PDF/DOCX/XLSX writer and no format-preserving merge. Compression and optimisation are separate operations (see below), not output formats.
- **Recovery** works on already-parsed text (null-byte and newline cleanup, then conservative token salvage). It does not repair damaged containers; that is `SalvageEngine`, which is opt-in (`salvage: true`). Reported confidence reflects character cleanliness, not readability.
- **Persistence:** `ReaderAgent` stores checkpoints, results and caches in `shared_memory` (in-process). Nothing reads checkpoints back and nothing persists shared memory automatically, so that state is lost on restart. The one durable store is the job journal (`JobEngine`, opt-in via `durable: true`). Parse-cache keys use path, size and mtime, not content.
- **Timeouts:** the agent has no per-task timeout or cancellation.
- **Interpreters:** tests pass on Python 3.10 and 3.13. `pyproject.toml` still declares `<3.13`.

## Foundations and capability engines

Everything here is opt-in: with default settings `ReaderAgent` behaves exactly as before, apart from two bug fixes (same-stem outputs no longer overwrite each other, and the module layout of the agent file).

### Agent modes and task flags

| Use | Task | Result |
|---|---|---|
| Identify formats | `mode: detect` | one detection report per file (format, family, confidence, mismatch, damaged, encrypted) |
| Check files | `mode: validate`, optional `deep: true` | one integrity report per file; an invalid file is a finding (`status: invalid`), the run is `partial` |
| Package / shrink | `mode: compress`, `compress_method` (`auto`, `package`, `optimize`, `gz`, `bz2`, `xz`), `package_format`, `lossy` | `auto` optimizes a single optimizable file, otherwise packages |
| Extract | `mode: decompress` | files extracted into a new directory; unsafe members, links and nested archives are skipped or listed, never followed |
| Recover a damaged file | `mode: salvage`, or `salvage: true` on a normal task | repaired copy plus a report; in a pipeline the document keeps its original `source`, carries `metadata.salvage` and a `salvaged:<level>` warning, and the run is `partial` |
| Survive interruption | `durable: true` on a convert or merge task; `mode: resume` with `job_id`; `mode: jobs` to list or inspect | finished artifacts are journaled; re-running the same task, or `resume`, redoes only what is missing, changed or failed |
| Verify / annotate | `verify: true`, `detect: true` | each written artifact is validated before it is reported; parsed documents get `metadata.detection` |

### Result levels and honesty rules
- **Salvage levels:** `full` (nothing lost), `partial` (format kept, some parts lost), `text_only` (only text recovered, layout and media gone), or `SalvageError` (nothing usable; the report is attached). When damage is detected but the lost parts cannot be listed, an `unverified_remainder` entry stops the level from claiming `full`. A repaired file is re-validated and discarded if it does not validate. Recovered documents are never written to the parse cache.
- **Compression:** every report states `lossless`, `verified` and `verified_by` (`round_trip_sha256`, `member_sha256`, `pixel_equality`, `page_count+page_text`, ...). PDF optimisation is verified by page count and page text, not by comparing interactive features.
- **Limits are enforced on bytes actually read**, not on sizes declared in an archive header (`integrity_engine.max_*`).
- **Outputs are published without overwriting** (`publish_file`). Where the filesystem has no hard links (FAT/exFAT, some network shares) the helper reserves the name exclusively and then replaces it; the path actually used is the one reported.

### Jobs
- The journal lives in `job_engine.job_dir` (default `tmp/reader/jobs`), separate from diagnostic checkpoints, so checkpoint pruning can never delete a job. Records are stored verbatim (no secret masking or string truncation, unlike diagnostic checkpoints).
- A job left `running` by a crashed process is resumable after `stale_after_seconds`, or immediately with `force`. Single writer per job is assumed.
- Merge jobs are one item; convert jobs have one item per input file.

### Tests
```
python -m unittest tests.test_reader_hardening tests.test_reader_engines -v
```
Each engine also has a self-test: `python -m src.agents.reader.<module>` (for example `python -m src.agents.reader.salvage_engine`).
