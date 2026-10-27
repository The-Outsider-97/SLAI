# SPlanner — SLAI v2.4

SPlanner is a local-first planning workspace in `applications/splanner`. This
directory currently contains the **functional Phase 1 foundation (v0.1.0)**,
not the complete multi-user and AI-assisted v1.0 product.

## Run

From the root of the **SLAI-v.2.4** checkout:

```bash
python -m pip install -r applications/splanner/requirements.txt
python -m applications.splanner
```

SPlanner opens `http://127.0.0.1:8769` with a randomly generated session access
token. The token is not persisted. The server refuses non-loopback connections.
Use `--no-browser` if you only want to start the server, or choose a port with
`--port 8770`. Other options: `--data-dir PATH` and `--config PATH`.

Default storage:

- Windows: `%LOCALAPPDATA%\SLAI\SPlanner`
- Linux: `$XDG_DATA_HOME/slai/splanner`, or `~/.local/share/slai/splanner`

Calendar records, tasks, reminders and sticky-note coordinates are stored in
`planner.sqlite3` (SQLite WAL, foreign keys, and transactional writes). Every
SPlanner rich note, including the contents of a sticky, is saved separately as
`notes/<opaque-ID>.sldc` using the **canonical** `src.utils.sldc` library.

## Implemented

- Weekly time-grid calendar, event create/edit/delete, colored blocks and date navigation.
- Task creation, priority, due date, completion, editing and deletion.
- A simple reminder inbox with durable due times, dismiss and snooze.
- Private SLDC v1.0 note creation, reopen and save using canonical package APIs.
- Basic text editing and autosave with version checks and explicit unsaved feedback.
- Sticky-note board with drag positioning; sticky content is SLDC-backed.
- Soft light/dark interface aligned with SciCal/Reader visual conventions.
- Loopback-only, random-token HTTP API, Origin checks on writes, capped requests.
- Structured YAML settings through SLAI `ConfigRepository`.

## Not yet implemented

The current app does **not** claim these capabilities:

- SLAI agent-driven scheduling or LLM assistance.
- Cross-app Reader/Writer handoff or reuse of Reader's ink/annotation editor.
- Collaborative accounts, permissions, sharing or synchronization.
- Recurring calendar events or recurring notifications.
- OS notifications while the app/browser is closed; reminders are polled only
  while the page is open. Native browser notifications are optional and require
  user permission.
- `.ics` import/export or external calendar providers.
- Handwritten journals, rich styling of notes, and simultaneous co-editing.
- Desktop package installers or production release signatures.

The basic note editor intentionally refuses advanced `.sldc` packages rather
than discarding spans, images, signing metadata, or annotations. Reader's own
annotation records are not automatically embedded in the source `.sldc` file;
that integration must be designed and tested before it is claimed.

## Security and data model boundaries

- The API binds to `127.0.0.1` and requires the launched session token for
  planner data. All document filenames are generated opaque IDs, not browser
  paths. This is a single-user local application, **not** a remote sharing API.
- `.sldc` is used for note content, not for calendar recurrence state or ACLs.
- Timestamps require explicit UTC offsets and are stored in UTC. Event time-zone
  identifiers must be valid IANA zones; Windows requires the `tzdata` dependency.
- The canonical `.sldc` writer stages files on the destination filesystem and
  performs atomic replacement when supported by the underlying OS/filesystem.
- Autosave asks before page unload when changes are still pending. A hard crash
  during an unsaved edit may lose those pending keystrokes; later phases should
  add recoverable drafts/journaling.
- SQLite WAL files are part of the database's durable state and should be
  included in coherent backups; do not copy only the main `.sqlite3` file
  while SPlanner is running.

## Tests

From the root checkout:

```bash
python -m unittest tests.test_splanner -v
node --check applications/splanner/web/app.js
python -m compileall -q applications/splanner
```

The tests use the actual SLDC module, SQLite and loopback HTTP handlers rather
than placeholder document adapters. HTTP tests require permission to bind a
loopback socket. All desktop UI paths should also undergo manual browser QA
before a release claim.

## Next milestones

1. **Phase 1 hardening:** calendar overlap layout, more complete CRUD and form
   feedback, controlled database migration/versioning, note draft recovery,
   structured reminder associations and native notification service feasibility.
2. **Phase 2:** negotiate a versioned Reader annotation/Writer handoff bridge,
   shared tokens/styles and richer note/journal modes, without duplicating
   canonical Reader document rendering.
3. **Phase 3:** deploy an authenticated collaboration service with explicit
   ACLs, encrypted transport, invitations and safe offline conflict resolution.
4. **Phase 4:** adapt actual SLAI v2.4 Planning/Optimization/Reasoning services
   using a permission-scoped, explainable, user-confirmed proposal interface.
5. **Phase 5:** iCalendar interoperability, provider synchronization with real
   OAuth scopes, end-to-end UI testing and installable desktop packaging.
