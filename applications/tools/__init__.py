"""SLAI application tools: the one implementation of everything the SLAI applications share.

Python (this package)
    errors        typed errors with an HTTP status and a stable code; storage failures classified by real cause
    validation    request-field validation that names the field; UTC instants; opaque ids
    storage       data/output folders, atomic file writes, schema-checked JSON preferences
    database      SQLite with WAL, explicit transactions, versioned migrations and coherent backups
    config        typed ``config.yaml`` settings read through SLAI's ``ConfigRepository``
    slai          finding the SLAI checkout; the canonical SLDC module; the shared assets folder
    server        the loopback HTTP server (token handshake, Host/Origin checks, CSP), router and start-up scaffolding

Web (``tools/web``)
    slai-ui.css   one stylesheet: colour tokens for light, dark and high-contrast themes, and the shared controls
    src/*.ts      strict TypeScript: DOM builder, API client, dialogs and toasts, icons, appearance, utilities

Rules: nothing application-specific lives here, and an application never re-implements what is here. See ``README.md``.
"""

__version__ = "1.0.0"
