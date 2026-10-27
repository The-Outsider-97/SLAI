# SPlanner integration and reuse audit — SLAI-v.2.4

**Audit date:** 2026-10-27

**Source branch:** `SLAI-v.2.4`

**Scope:** Directly inspected source trees; implementation status for v0.1.

| Capability | Verified canonical implementation | SPlanner decision | Status |
| --- | --- | --- | --- |
| SLDC serialization | `src/utils/sldc.py` (`new_package`, `make_block`, `read_package`, `write_package`) | Import and call the source module directly; no copy of the format | **Integrated and round-trip tested** |
| SLDC package editing | `src/utils/sldc.py` supports multiple blocks, assets, signatures and comments | Limit initial in-app editor to a single unsigned paragraph; reject advanced editing instead of flattening | **Restricted, tested** |
| Reader annotation UI | Separate `SReader` repository: `src/client/annotate-ui.ts`, `src/client/ink.ts`, `src/core/strokes.ts` | Design a versioned shared component/bridge before importing or transplanting tools | **Not integrated** |
| Reader annotation persistence | Separate `SReader/slai_reader/annotations/{store,service,model}.py` | Reader stores annotations separately; cannot assume annotations are embedded into `.sldc` | **Observed limitation** |
| Reader handoff | `SReader/slai_reader/interop.py` implements `slai-handoff/1` | Reader currently permits `writer` and `documaster` targets. Define explicit `splanner` target and handshake in a later, tested cross-repository update | **Not implemented** |
| SLAI Writer integration | Writer source not present in this branch's `applications` directory | Require concrete Writer source/command contract; never invent a handoff target | **Not implemented** |
| SciCal appearance | `SciCal/src/client/styles.css` defines soft-neutral design tokens, high contrast and motion rules | Align palette, surface treatment and focus states in standalone UI; avoid copying entire UI architecture | **Visually adapted; no shared package yet** |
| Core configuration | `src/utils/configuration.py` (`ConfigRepository`) | Load `config.yaml` using canonical repository | **Integrated** |
| Planning intelligence | `src/agents/planning_agent.py`, factory in `src/agents/agent_factory.py` | Introduce read-only proposals only after schema/consent contract | **Deferred** |
| Optimization | `src/agents/optimization_agent.py` | Delay until feasibility constraints and objective semantics are agreed | **Deferred** |
| Privacy and validation | `src/agents/privacy_agent.py`, `src/agents/safety_agent.py`, `src/agents/verification_agent.py` | Required for future context retrieval and mutations, not current local CRUD | **Deferred** |
| Calendar persistence | No existing canonical planner service found in the target `applications` tree | Local SQLite with basic typed events/tasks/reminders; no claim of multiuser support | **Integrated locally** |
| Collaboration identity/sync | No suitable universal desktop app identity/sync service verified during this audit | Explicitly omit remote sharing until a real authenticated service and ACL contract exist | **Deferred** |

## Architectural constraints

The SLAI v2.4 `applications/` tree contains `__init__.py` and `assets/`, while
SReader and SciCal are separate GitHub repositories. Their documented runtime
ports are 8767 and 8768, respectively. SPlanner uses 8769 by default.

This checkout can reuse canonical in-repository SLDC and configuration services
immediately. Cross-application UI/annotation reuse requires independent package
versioning, a bridge, or explicit source inclusion: direct imports into SPlanner
would otherwise create a fragile dependency on other unchecked-out repositories.

## Evidence and remaining QA

- `python -m unittest tests.test_splanner -v`: **7 passed** with loopback socket
  capability granted.
- `node --check applications/splanner/web/app.js`: **passed**.
- `python -m compileall -q applications/splanner`: **passed**.
- Playwright Chromium download was attempted but unavailable; **visual browser
  automation and screenshot QA are not verified** in this environment.
- Windows-specific and macOS-specific runtime validation are **not performed**.
- Multiuser/security penetration testing is **not applicable to v0.1**, because
  remote serving and sharing are deliberately disabled.

## Integration acceptance gate

Do not mark Reader/Writer interoperability complete until actual bidirectional
open/handoff flows, revision mismatch detection, permissions, annotation
round-trip, and document ownership semantics pass integration tests. Do not
mark SLAI agent participation complete until actual agent construction,
configuration, consent, safe proposal validation, and offline fallback have
been exercised with the real v2.4 interfaces.
