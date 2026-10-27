# SLAI application tools

One implementation of everything the SLAI applications share, so that every application behaves the same, looks the same, and
is fixed in one place. SPlanner is the first application built on it. Reader and SciCal still carry their own copies of some of
these pieces (see *Migrating Reader and SciCal* below); they have not been changed.

**Rules**

1. Nothing application-specific lives here: no event, document, formula or calculator concept.
2. An application never re-implements what is here. If an application needs something the tools lack, it is added here first,
   with a test, and then used.
3. The tools depend on nothing outside the standard library, PyYAML (through SLAI's own `ConfigRepository`) and TypeScript.

```
applications/tools/
  __init__.py  errors.py  validation.py  storage.py  database.py  config.py  slai.py
  server/            router.py  http.py  launch.py          the loopback HTTP server
  web/
    slai-ui.css                                              the one stylesheet
    src/ dom.ts util.ts api.ts icons.ts theme.ts ui.ts       the client kit (strict TypeScript)
  build/build-static.mjs                                     builds any application's client
  tests/  tests_ts/  typings/                                44 Python tests, 13 TypeScript tests
```

## Python

| Module | What it gives an application |
| --- | --- |
| `errors` | `AppError` and its subclasses (`BadRequest`, `Unauthorized`, `Forbidden`, `NotFound`, `Conflict`, `LimitExceeded`, `DependencyMissing`, `StorageError`). Each carries an HTTP status, a stable `code`, a plain-language `message` and an optional `hint`. `classify_oserror` and `classify_sqlite` turn "disk full", "permission denied", "database is locked" and the like into a `StorageError` that says what really happened. |
| `validation` | Field checks that name the field in the message: `text`, `choice`, `boolean`, `integer`, `number`, `instant` (an explicit UTC offset is required), `zone` (a real IANA zone), `record_id`, `new_id`, `utc_now`, `parse_instant`, `like_pattern`. |
| `storage` | `default_data_dir` (Windows `%LOCALAPPDATA%\SLAI\<App>`, otherwise `$XDG_DATA_HOME/slai/<app>`), `atomic_write*` (stage on the destination filesystem, flush, replace) and `Prefs`, schema-checked JSON preferences: unknown or invalid keys are refused with a message that names them, and an unreadable file starts over from the defaults and says so (`recovered`) instead of crashing. |
| `database` | `Database`: SQLite in WAL mode with foreign keys, a re-entrant `transaction()`, `PRAGMA user_version` migrations, a copy of the data before any migration that touches existing data (`migration_backup`, `upgraded`), and `backup()` through SQLite's online backup API. |
| `config` | `load_section(path, section, fields)`: typed settings from `configs/config.yaml`, read through SLAI's `ConfigRepository`, with a message that names the bad key. |
| `slai` | Finds the SLAI checkout (`--slai-root`, `SLAI_ROOT`, or by walking up), puts it on `sys.path`, and returns the canonical `src.utils.sldc` module and `applications/assets`. |
| `server` | `LocalServer`, `Handler`, `Router`, `serve`, `add_server_arguments`, `refuse_remote_host`. See below. |

### What the server does for every application

* Listens on loopback only, and refuses to start on any other address.
* Hands out access through a random token in the start-up URL. The token is traded once for an `HttpOnly`, `SameSite=Strict`
  cookie and the address bar is redirected to a clean URL. No cookie, no data.
* Checks the `Host` header (so DNS rebinding cannot reach it) and, on every write, an `Origin` that matches plus an
  `X-SLAI-<App>` marker header that a cross-site form cannot send.
* Sends a strict Content-Security-Policy (`default-src 'self'`, `script-src 'self'`, `style-src 'self'`, no framing), `nosniff`
  and `no-store`. Pages therefore contain no inline script or style; code sets styles through `style.setProperty`.
* Caps request size, answers every error as `{"ok": false, "error": {"code", "message", "hint?", "details?"}}` and every success
  as `{"ok": true, "data": ...}`, and never writes what a person plans or writes to its log.
* Serves the application's `dist/`, the shared stylesheet at `/tools/slai-ui.css`, and images and sounds at `/assets/...`
  (the application's own `assets/` first, then `applications/assets`).
* Shuts down cleanly on Ctrl+C, Ctrl+Break or SIGTERM.

An application writes only its routes:

```python
router = Router()

@router.route("POST", "/api/things")
def create(app, req):
    return created(app.store.create_thing(req.json()))
```

## Web

`web/slai-ui.css` holds the colour tokens (`light-dark()`, with `data-theme="light|dark|contrast"` and `prefers-color-scheme`
as the default) and every shared control: buttons, fields, switches, segmented buttons, chips, cards, notices, dialogs, toasts,
menus, swatches and the tone classes `.tone-blue|violet|green|pink|amber`. The `data-skin` attribute picks the surface style:
`raised` (the default, as in Reader and SciCal) or `soft` (flat, as in SPlanner). Reduced motion, forced colours and print are
handled once, here.

`web/src` is compiled together with the application (the application's `tsconfig.build.json` sets `rootDir` to `applications/`
so it can import the kit by relative path, with no bundler):

| Module | Contents |
| --- | --- |
| `dom` | `h()` and `s()` element builders (no `innerHTML` anywhere), `replace`, `clear`, `$`, `$$`, `labelled`, `announce` (screen-reader live region), `focusable`, `nextId`. |
| `util` | `clamp`, `debounce` (with `flush` and `cancel`), `uid`, `Emitter`, `compareText`, `formatBytes`. |
| `api` | `createClient({slug})` with `get/post/put/patch/del/download`, the marker header, the envelope, and `ApiError.friendly`. |
| `icons` | `glyph(name)`, an inline-SVG set that follows `currentColor`, `registerGlyph` for an application's own, `assetIcon(url)`. |
| `theme` | `applyAppearance`, `appearanceControls` (theme, text size 85–140 %, reduce motion) and `normaliseAppearance`. |
| `ui` | `toast`, `openDialog`, `confirmDialog`, `promptDialog`, `openMenu` (menus opened inside a dialog stay inside it), `copyText`, `saveBlob`. |

## Building and testing

```
node applications/tools/build/build-static.mjs applications/<app>     # or, from the app folder:  npm run build
python -m unittest discover -s applications/tools/tests -t .          # Python
cd applications/tools && tsc -p tsconfig.json                         # type-check the kit
cd applications/tools && tsc -p tsconfig.test.json && node --test .test-out/tests_ts/*.test.js
```

`build-static.mjs` removes `dist/js`, runs the application's own `typescript` (or a `tsc` on the path) with its
`tsconfig.build.json`, and copies `src/client/index.html` and `*.css` next to it. Icons and sounds are never copied.

## Adding a new application

1. Copy the layout of `applications/splanner`: `__init__.py`, `app.py`, `routes.py`, `store.py`, `run_<app>.py`,
   `configs/config.yaml`, `src/{core,client}`, `tsconfig*.json`, `package.json`, `tests*`.
2. In `app.py`, build a `LocalServer` with a `ServerOptions(slug=..., title=..., client_dir=...)` and your `Router`; call `serve`.
3. In `index.html`, link `/tools/slai-ui.css` before your own `styles.css`, and set `data-skin` if you want the soft surface.
4. Import the kit through one file (SPlanner's `src/client/kit.ts`) so that the relative path appears once.
5. Put your layout in `styles.css` using the kit's tokens; never a literal colour, so all three themes keep working.

## Migrating Reader and SciCal

Not done, deliberately: both applications work and are tested, and moving them is a separate change that should be reviewed on
its own. The duplicated pieces are `tools/build-static.mjs` (identical apart from names), `src/client/dom.ts` and `ui.ts`
(Reader), `src/client/api.ts` (SciCal), the token/cookie/Origin/CSP handling in each Python service, and the appearance
controls. Each can move to the kit one module at a time, with that application's existing tests as the safety net.

## Known limitations

* The TypeScript tests run with `tsx` in the applications' convention (`npm test`). Where `tsx` cannot be installed, compile with
  `tsc` and run `node --test` on the output (`npm run test:compiled`); both run the same files.
* `Database` is synchronous and uses one connection guarded by a lock. That is right for a single-user local service; it is not a
  connection pool.
* The kit has no component for rich text, charts or a PDF view. Those stay in the applications that need them.
