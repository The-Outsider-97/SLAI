# Reference-app audit and Secure Folder design decisions

Inspected the source implementations of [SciCal](https://github.com/The-Outsider-97/SciCal), [SPlanner](https://github.com/The-Outsider-97/splanner), and [SReader](https://github.com/The-Outsider-97/SReader) on their `main` branches, and the shared toolkit of [SLAI v2.4](https://github.com/The-Outsider-97/SLAI/tree/SLAI-v.2.4).

| Reference | Observed implementation | Secure Folder adaptation |
| --- | --- | --- |
| SciCal | TypeScript/Python local app, workspace tabs, a smooth soft-surface UI, compact header and context-specific tools. | Smooth transitions, concise headers, contextual file actions and focused workspaces. |
| SPlanner | TypeScript/Python local app, persistent sidebar, responsive topbar, `data-skin="soft"`, shared SLAI assets/tools and feature views. | Uses the exact shared SLAI kit; sidebar, main workspace, gold-accent soft cards, mobile tabs. |
| SReader | TypeScript/Python local app, document-oriented toolbar, library and reading state, keyboard-oriented tools and multiple accessibility themes. | File-oriented list, preview dialog, cautious document format boundaries; no direct, unsafe import of Reader's backend. |
| SLAI shared tools | Central `slai-ui.css`, icons, theme handling, DOM builders, dialogs, local HTTP security, validation and atomic storage. | No duplicated generic UI kit or generic local HTTP implementation. |

**Real repository layout:** SLAI's application directories are `applications/assets/` and `applications/tools/` at the repository root, not nested under a `slai/` directory. The external apps are independent repositories, so this change does not copy their source or require installing them.

## Design system

- Shared SLAI tokens (`--accent`, `--surface`, `--hairline`, etc.) govern both light and dark appearance.
- The soft skin, restrained panel shadows, rounded cards and high-contrast mode come from the canonical SLAI CSS.
- The authentication view is intentionally a focused single card; decrypted workspace controls appear only after unlock.
- Desktop: sidebar, search topbar, folder cards, file actions, security area.
- Mobile: single-column content, compact header, bottom tabs and collapsible menu.
- Dialogs, icons, theme controls, safe DOM text rendering and toast notifications use the shared kit.

The supplied Secure Folder images were treated as inspiration for privacy/dashboard concepts only, not copied assets. Deep integration with SReader's PDF/SLDC viewers and SciCal `.sgeo`/`.sdrw` requires explicit, reviewed adapters in later changes; temporary plaintext handoff into unrelated app directories is deliberately not implemented.

## Technical finding

`modules/security_manager.py` governs **agent permissions**, not password-derived disk encryption. It cannot serve as the vault's cryptographic boundary. Secure Folder deliberately reuses the shared SLAI HTTP and UI layer while using `cryptography` for file confidentiality. AI and other app access remain disabled by default.
