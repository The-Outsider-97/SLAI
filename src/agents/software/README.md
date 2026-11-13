# SLAI v2.4 Software Subsystem

`src/agents/software/` is the program-analysis and source-transformation substrate behind `SoftwareAgent`. It understands, transforms and reasons about software artifacts. It produces **facts, reports and unapplied patch proposals**; it never executes, installs, writes or applies anything.

## Scope

1. **Syntax and symbols** – character-accurate AST facts, LEGB scope resolution, a symbol index with cross-module resolution (import chains, re-exports, star imports, `self`/`cls`/`super()` receivers, C3 method resolution) and exact vs. heuristic references.
2. **Graphs** – dependency graph (load-time / lazy / `TYPE_CHECKING` / conditional edges, Tarjan SCCs, layering) and call graph (exact, override and name-match edges, recursion groups, external calls).
3. **Semantics** – class hierarchy, code search (symbol queries and grep), public-API extraction and interface compatibility (breaking / additive / compatible, SemVer bump).
4. **Analysis** – rule-based static analysis (`SW1xx` correctness, `SW2xx` error handling, `SW3xx` security, `SW4xx` complexity, `SW5xx` style; `# noqa[: codes]` suppression), dead-code detection, change-impact analysis with test selection, and architecture analysis (layers, forbidden edges, cycles, Martin metrics).
5. **Transformation proposals** – rename, unused-import removal, dead-code removal, declarative API migration and externally supplied unified diffs. Each proposal carries a forward and reverse diff, syntax validation, a before/after regression check, risks, a confidence and its lineage.

It does **not** run shell commands (ExecutionAgent), read arbitrary documents (ReaderAgent), browse the web or GitHub (BrowserAgent), judge final performance (EvaluationAgent) or do generic NLP (LanguageAgent).

## Hand-off contract

```
Browser/Reader → SoftwareAgent (proposes) → Verification → Safety → Execution (applies) → Evaluation
```

`PatchProposal.applied` is always `False` and `requires_gates` is `("verification", "safety", "execution", "evaluation")`. The subsystem stops before OS and tool execution. A caller can chain proposals: the result of a patch is available as a repository state, and lineage is recorded.

## Layout

| Path | Role |
| --- | --- |
| `software_types.py` | Immutable contracts: enums, source facts, reports, patch types |
| `models/` | `SourceModel` (one file), `RepositorySnapshot` / `RepositoryModel` (content-addressed, incrementally rebuilt) |
| `graphs/` | `BaseGraph`, `DependencyGraph`, `CallGraph` |
| `semantics/` | `SymbolIndex`, `ClassHierarchy`, `CodeSearch`, `InterfaceCompatibility` |
| `modules/` | `parser`, `static_analysis`, `dead_code`, `change_impact`, `architecture`, `patch_builder`, `refactoring`, `api_migration` |
| `software_models.py` … `software_refactoring.py` | Five central handlers, one per layer (below) |
| `software_memory.py` | Private, bounded caches and the proposal ledger (subsystem internal) |
| `configs/software_config.yaml` | Subsystem thresholds, limits and cache sizes |
| `utils/` | `software_errors` (`SFT-xxxx`, inherits `BaseError`), `software_helpers`, `config_loader` |

The handlers are layered bottom-up and each only talks to the one below it:

```
SoftwareModels → SoftwareGraphs → SoftwareSemantics → SoftwareAnalysis → SoftwareRefactoring
```

## Configuration and memory

Subsystem settings live in `configs/software_config.yaml` and are read through `utils/config_loader.py`; every key is optional. Agent-level orchestration (SharedMemory publication, key prefixes, allowed operations) lives in `agents_config.yaml` under `software_agent`. `SoftwareAgent` uses only that section and SharedMemory; it does not import the subsystem's memory or config loader. The subsystem's caches are recomputable and process-local; the proposal ledger is how `base_patch_id` and `proposal` lookups work within one agent instance.

## Minimal usage

```python
from src.agents.software import (
    SoftwareAnalysis, SoftwareGraphs, SoftwareModels, SoftwareRefactoring, SoftwareSemantics,
)

models = SoftwareModels()
graphs = SoftwareGraphs(models)
semantics = SoftwareSemantics(graphs)
analysis = SoftwareAnalysis(semantics)
refactoring = SoftwareRefactoring(analysis)

repo = models.repository(models.snapshot(files={"app/core.py": "def user():\n    return 1\n"}))
report = analysis.static(repo)
result = refactoring.rename(repo, "app.core.user", "run_user")
print(result.proposal.diff, result.proposal.applied)   # ... False
```

Through the agent:

```python
agent.perform_task({
    "operation": "rename",
    "files": {"app/core.py": "def user():\n    return 1\n"},
    "parameters": {"symbol": "app.core.user", "new_name": "run_user"},
})
```

### Request shape

| Field | Meaning |
| --- | --- |
| `operation` | `summary`, `outline`, `symbol`, `search`, `references`, `dependency_graph`, `call_graph`, `class_hierarchy`, `static_analysis`, `dead_code`, `change_impact`, `compatibility`, `architecture`, `rename`, `remove_unused_imports`, `remove_dead_code`, `migrate_api`, `apply_diff`, `proposal` |
| `files` \| `snapshot` \| `base_patch_id` | Exactly one repository state: file contents, a remembered snapshot digest, or the result of an earlier patch |
| `changes` | Overlay of path → new text (`null` deletes) on that state |
| `baseline` | A second state of the same shape, for `change_impact` and `compatibility` (or pass `parameters.patch_id` to compare a recorded proposal with its base) |
| `source_roots` | e.g. `["src"]` for a src-layout |
| `parameters` / `options` | Operation arguments / per-request analyzer overrides (thresholds, `disabled_rules`, …) |
| `request_id`, `metadata`, `prior_request_ids` | Correlation and shared-memory references |

Outcomes always carry `execution_success`, `proposal_applied: false`, `requires_gates` and, for patches, `patch_id`.

## Design notes and limits

- **Stdlib only.** Parsing and analysis use `ast`, `difflib`, `hashlib` and `re`; torch is not required. No filesystem, network or subprocess access.
- **Content-addressed.** A snapshot's digest is order-independent; unchanged files are never re-parsed across snapshots.
- **Honest uncertainty.** References, call edges and impact edges that rest on name matching are flagged `exact=False` and carry lower confidence. Files that use `eval`/`exec`/`locals` suppress undefined/unused findings conservatively. Unresolvable constructs are counted, not guessed.
- **Python only today.** Other languages can be added by registering a `LanguageParser` with `SoftwareParser.register`.
- **No type inference.** Method calls on values of unknown type fall back to unique-name matching.
- **Not implemented:** extract/move refactorings; `rename_module` rewrites imports but does not move files; complexity metrics are approximations of cyclomatic/cognitive complexity.

## Academic design basis

Aho et al. (2006) *Compilers*; Tarjan (1972) strongly connected components; Grove et al. (1997) call-graph construction in object-oriented languages; Barrett et al. (1996) C3 linearization; McCabe (1976) cyclomatic complexity; Campbell (2018) cognitive complexity; Martin (2002) package metrics; Bohner & Arnold (1996) change-impact analysis; Fowler (2018) *Refactoring*.

## Testing

```bash
pytest tests/agents/test_software_subsystem.py tests/agents/test_software_agent.py
```
