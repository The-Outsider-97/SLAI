# SLAI Secure Folder

Local-first encrypted file vault for the SLAI v2.4 application suite. This is a **security-focused 0.1 implementation**, not a certified production security product. Review [SECURITY.md](SECURITY.md) and [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) before storing irreplaceable sensitive material.

## Run

From `applications/secure_folder`:

```bash
npm ci
npm run build
```

Then, from the SLAI repository root (Python 3.10–3.12 and `cryptography` installed):

```bash
python -m applications.secure_folder
```

The launcher starts a loopback-only service and prints a one-time URL to open in the browser. On first use, create a strong passphrase (at least 12 characters); subsequent launches always require the passphrase again. A short numeric PIN is intentionally not supported: a software-only PIN would expose the offline vault to trivial guessing attacks.

Use `--no-browser` to avoid automatically opening a browser, or `--data-dir` to select a vault directory. **Do not place the live vault on an untrusted/shared filesystem.** By default data resides in the SLAI application-specific user data directory. Never place it in the source repository.

## Architecture

```text
applications/secure_folder/
  app.py              SLAI local server bootstrap and single-instance lock
  routes.py           protected endpoints, no file path parameters
  vault.py            encrypted index, file chunks, short-lived unlock sessions
  crypto.py           scrypt KDF, AES-256-GCM, authenticated envelopes
  src/client/         responsive TypeScript UI, SLAI shared UI kit
  tests/              vault security and HTTP authorization tests
```

Shared SLAI services used:

- `applications/tools/server`: loopback-only Host/Origin validation, random launch-token cookie, CSRF marker, CSP, no-store responses, request-size bounds and static-file sandbox.
- `applications/tools/storage`: user-specific data directory and atomic writes of already-encrypted blobs.
- `applications/tools/web`: canonical theme tokens, responsive controls, accessible dialogs, icons and appearance settings.
- `applications/assets`: canonical SLAI branding rather than copied images from reference applications.

The browser's launch cookie is **not** sufficient to open the vault. Every vault operation also requires a random in-memory `X-Vault-Session` token created by a passphrase unlock. Tokens are never written to local/session storage, cookies or URLs. Refreshing the page loses the token and requires reauthentication. The server revokes tokens on explicit lock and after idle/absolute expiry.

## Current capabilities

- First-use vault setup, passphrase unlock, manual lock, background lock and 5-minute inactivity timeout.
- Encrypted folders, arbitrary file import in bounded 512 KiB chunks, favorites, rename, move, search and categories.
- Private encrypted notes and a protected note editor.
- Encrypted Trash, restore, passphrase-confirmed permanent logical deletion.
- Browser previews for safe image types and plain text; limited, reauthenticated plaintext export.
- Passphrase changes that rewrap the same vault key and revoke all sessions.
- Responsive desktop/mobile browser layout; system/light/dark/high-contrast themes and reduced-motion handling.
- No cloud connection, no AI access, no analytics and no internet requirement after the client is built.

**Scope limits:** The Python loopback server must run on the same device as the browser. A responsive interface on a phone is not an Android/iOS native release or cross-device access to the PC's vault. Native mobile packaging and integration with OS-secure keystores remain separate platform projects. Export/preview currently buffers files up to 96 MiB in the browser; import streams in bounded chunks up to 2 GiB. The app is not yet equipped with encrypted backup/restore or comprehensive PDF/audio/video preview. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## Test

```bash
python -m unittest discover -s applications/secure_folder/tests -t . -v
python -m unittest discover -s applications/tools/tests -t .
cd applications/secure_folder
npm run typecheck
npm run build
```

The HTTP tests start a `127.0.0.1` server and therefore need a test environment that permits binding loopback sockets. Tests use a temporary vault and synthetic secrets, never personal data.

## Recovery warning

There is no recovery administrator, hidden backup key or master password. If the passphrase is lost and no separate valid backup exists, encrypted content may be irrecoverable. A supported encrypted backup/restore workflow is required before using the current version for irreplaceable data.
