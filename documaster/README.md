# DocuMaster · SLAI integration

DocuMaster is a SLAIHub desktop module. It is opened from `main.py` through `documaster.documaster.DocumasterWindow` and should not be launched through SignalSentry, AutoPublisher, or any unrelated fallback app.

## What changed

- Added a strict `SLAIAdapter` boundary in `documaster/slai_adapter.py`.
- The SLAI Reader Agent is now the preferred document ingestion path for parsing, recovery, conversion-aware reading, and reader metadata.
- Local extraction remains only as a degraded DocuMaster fallback when SLAI/Reader Agent is unavailable.
- Added privacy, safety, quality, and observability gates where suitable.
- Kept persistent memory disabled by default.
- Improved the PDF merge workflow with an explicit page-level merge plan.
- Refactored the GUI into SLAIHub-style PyQt tabs:
  - AI Assistant
  - Reader / Analysis
  - Word Counter
  - Converter
  - PDF Merge / Page Organizer
  - PDF Editor / Future Tools
  - Runtime / SLAI Health
- Heavy learning agents are not loaded during normal document processing.

## Environment flags

```bash
# Enabled by default in this package. Set to 0 to force local degraded mode.
DOCMASTER_ENABLE_SLAI=1

# Privacy defaults: no persistent shared-memory document caching/checkpoints.
DOCMASTER_ENABLE_PERSISTENT_MEMORY=0

# Optional, off by default because these are not needed for normal document tasks.
DOCMASTER_ENABLE_KNOWLEDGE_AGENT=0
DOCMASTER_ENABLE_REASONING_AGENT=0
DOCMASTER_ENABLE_EVALUATION_AGENT=0

DOCMASTER_MAX_UPLOAD_BYTES=20971520
DOCMASTER_MAX_PAGES=250
DOCMASTER_MAX_TEXT_CHARS=120000
DOCMASTER_ALLOWED_FILE_TYPES=.pdf,.docx,.txt,.html,.htm,.xml,.odt
DOCMASTER_CLEANUP_TEMP_FILES=1

# Only needed if running DocuMaster outside the repository root.
SLAI_ROOT=/path/to/SLAI
```

## Install/runtime dependencies

Use the existing SLAIHub environment. DocuMaster expects:

```bash
pip install PyQt5 Flask werkzeug beautifulsoup4 python-docx pypdf lxml
```

The full SLAI runtime must be importable for real Reader Agent integration:

```bash
python main.py
```

Then click **Documaster** in SLAIHub.

## API preview

The PyQt desktop GUI is primary. Optional local/cloud preview routes are still available through `create_documaster_flask_app()`:

- `GET /api/ai/health`
- `POST /api/ai/runtime/initialize`
- `POST /api/ai/analyze`
- `POST /api/ai/summarize`
- `POST /api/ai/explain`
- `POST /api/ai/ask`
- `POST /api/ai/quality-check`
- `POST /api/ai/rewrite-suggestions`
- `POST /api/ai/key-points`

All upload routes use the same `DocumentAIService` as the desktop GUI and enforce explicit upload limits.

## Merge/page organizer workflow

1. Add at least two PDFs.
2. Click **Build / refresh page plan**.
3. Inspect file/page order and text preview metadata.
4. Move selected rows up/down.
5. Include or exclude selected pages.
6. Click **Merge selected plan**.

The final PDF is generated only from the validated structured plan. Original PDFs are never modified.

## Testing

Run syntax checks:

```bash
python -m py_compile \
  documaster/documaster.py \
  documaster/slai_adapter.py \
  documaster/services/document_ai_service.py \
  documaster/services/document_extractor.py \
  documaster/utils/documaster_utils.py \
  documaster/routes/ai_routes.py \
  documaster/styles/documaster_style.py
```

Run compact tests:

```bash
pytest documaster/tests/test_documaster_integration.py
```

## Limitations

- Image-only/scanned PDFs still need OCR support. This integration detects them and returns a clear readable error instead of fabricating text.
- The optional Evaluation Agent is not loaded by default because it imports heavy ML dependencies.
- Knowledge/Reasoning agents are intentionally off by default because DocuMaster should not persist uploaded document content unless explicitly configured.
- AI-powered PDF editing, rewriting, and comparison are prepared in the GUI but not implemented as destructive document modification tools.
