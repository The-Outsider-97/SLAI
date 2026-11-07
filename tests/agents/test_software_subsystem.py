"""Software subsystem: models, graphs, semantics, analysis and patch proposals."""

import ast
import pathlib

import pytest

from src.agents import software
from src.agents.software import (
    HANDOFF_GATES,
    SoftwareAnalysis,
    SoftwareError,
    SoftwareGraphs,
    SoftwareModels,
    SoftwareNameConflictError,
    SoftwarePatchError,
    SoftwareRefactoring,
    SoftwareSemantics,
    SoftwareSnapshotError,
    SoftwareSymbolError,
    SoftwareValidationError,
)

SAMPLE = {
    "app/__init__.py": "from .core import compute\n__all__ = ['compute']\n",
    "app/core.py": (
        "import os, sys\n"
        "class Base:\n    def compute(self, x):\n        return x\n"
        "class Sub(Base):\n    def compute(self, x):\n        return super().compute(x) + 1\n"
        "def compute(a):\n    return Sub().compute(a)\n"
        "def user():\n    v = compute(2)\n    return v\n"
        "def _dead():\n    return 1\n"
    ),
    "app/other.py": "from app.core import compute as c\nx = c(1)\n",
    "tests/test_core.py": "from app.core import compute\n\n\ndef test_compute():\n    assert compute(1)\n",
}


class Stack:
    """The five central handlers wired bottom-up, as the agent wires them."""

    def __init__(self):
        self.models = SoftwareModels()
        self.graphs = SoftwareGraphs(self.models)
        self.semantics = SoftwareSemantics(self.graphs)
        self.analysis = SoftwareAnalysis(self.semantics)
        self.refactoring = SoftwareRefactoring(self.analysis)

    def repo(self, files=None, **kwargs):
        return self.models.repository(self.models.snapshot(files=SAMPLE if files is None else files, **kwargs))

    def repo_of_patch(self, patch_id):
        return self.models.repository(self.models.snapshot(patch_id=patch_id))


def codes(report):
    return sorted({finding.rule_id for finding in report.findings})


# --------------------------------------------------------------------------- package surface

def test_public_surface_has_no_private_subsystem_state():
    exported = set(software.__all__)
    assert len(exported) == len(software.__all__)
    assert {"SoftwareModels", "SoftwareGraphs", "SoftwareSemantics", "SoftwareAnalysis", "SoftwareRefactoring"} <= exported
    assert "SoftwareMemory" not in exported
    assert not hasattr(software, "get_config_section")


def test_subsystem_performs_no_io_or_process_execution():
    root = pathlib.Path(software.__file__).parent
    banned_modules = {"subprocess", "socket", "shutil", "urllib", "requests", "http", "ctypes", "multiprocessing", "pickle", "marshal", "tempfile", "os", "pathlib"}
    allowed_pathlib = {"utils/config_loader.py"}
    offences = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                names = []
            for name in names:
                if name.split(".")[0] in banned_modules and rel not in allowed_pathlib:
                    offences.append((rel, name))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"open", "eval", "exec", "__import__"}:
                offences.append((rel, node.func.id))
    assert offences == []


# --------------------------------------------------------------------------- snapshots and models

def test_snapshot_digest_is_deterministic_and_order_independent():
    stack = Stack()
    one = stack.models.snapshot(files={"a.py": "x = 1\n", "b.py": "y = 2\n"})
    two = stack.models.snapshot(files={"b.py": "y = 2\n", "a.py": "x = 1\n"})
    assert one.digest == two.digest
    assert stack.models.snapshot(files={"a.py": "x = 2\n", "b.py": "y = 2\n"}).digest != one.digest


def test_snapshot_changes_add_replace_and_delete():
    stack = Stack()
    snap = stack.models.snapshot(files={"a.py": "x = 1\n", "b.py": "y = 2\n"}, changes={"b.py": None, "c.py": "z = 3\n", "a.py": "x = 9\n"})
    assert sorted(snap.paths()) == ["a.py", "c.py"]
    assert snap.file("a.py").text == "x = 9\n"


def test_snapshot_needs_exactly_one_state_source():
    stack = Stack()
    with pytest.raises(SoftwareSnapshotError):
        stack.models.snapshot()
    with pytest.raises(SoftwareSnapshotError):
        stack.models.snapshot(files={"a.py": ""}, digest="abc")


def test_unknown_digest_and_patch_are_reported_not_guessed():
    stack = Stack()
    with pytest.raises(SoftwareError):
        stack.models.snapshot(digest="missing")
    with pytest.raises(SoftwareError):
        stack.refactoring.proposal("missing")


def test_source_roots_shape_module_names():
    stack = Stack()
    repo = stack.repo({"src/pkg/__init__.py": "", "src/pkg/mod.py": "x = 1\n"}, source_roots=["src"])
    assert set(repo.module_names()) == {"pkg", "pkg.mod"}


def test_syntax_errors_are_contained_to_their_file():
    stack = Stack()
    repo = stack.repo({"good.py": "import os\n", "bad.py": "def f(:\n"})
    assert [model.path for model in repo.failed()] == ["bad.py"]
    report = stack.analysis.static(repo)
    assert "SW000" in codes(report) and "SW102" in codes(report)


def test_models_are_reused_across_snapshots():
    stack = Stack()
    stack.repo()
    before = stack.refactoring.stats()["models"]
    changed = dict(SAMPLE, **{"app/other.py": "from app.core import compute as c\nx = c(2)\n"})
    stack.repo(changed)
    after = stack.refactoring.stats()["models"]
    assert after["misses"] - before["misses"] == 1


# --------------------------------------------------------------------------- symbols, graphs, search

def test_references_follow_aliases_and_reexports():
    stack = Stack()
    repo = stack.repo()
    refs = stack.semantics.references(repo, symbol="app.core.compute")
    paths = {item["path"] for item in refs["references"]}
    assert {"app/__init__.py", "app/other.py", "tests/test_core.py", "app/core.py"} <= paths
    assert refs["exact"] == refs["count"]


def test_symbol_resolution_accepts_unambiguous_short_names_only():
    stack = Stack()
    repo = stack.repo()
    assert stack.semantics.resolve_symbol(repo, "app.core.user") == "app.core.user"
    with pytest.raises(SoftwareSymbolError):
        stack.semantics.resolve_symbol(repo, "app.core.nothing")


def test_super_calls_bind_statically_to_the_next_class_in_the_mro():
    stack = Stack()
    repo = stack.repo()
    callees = stack.graphs.call_neighbours(repo, "app.core.Sub.compute")["symbols"]
    assert callees == ["app.core.Base.compute"]
    assert stack.graphs.call_graph(repo).recursive_groups() == ()
    assert stack.semantics.index(repo).overridden("app.core.Sub.compute") == ("app.core.Base.compute",)


def test_call_path_and_neighbours():
    stack = Stack()
    repo = stack.repo()
    path = stack.graphs.path(repo, "app.core.user", "app.core.Sub.compute", graph="call")["path"]
    assert path == ["app.core.user", "app.core.compute", "app.core.Sub.compute"]
    callers = stack.graphs.call_neighbours(repo, "app.core.compute", direction="callers")["symbols"]
    assert "app.core.user" in callers
    with pytest.raises(SoftwareError):
        stack.graphs.call_neighbours(repo, "app.core.compute", direction="sideways")


def test_dependency_cycles_distinguish_load_time_from_lazy_imports():
    stack = Stack()
    eager = stack.repo({"a.py": "import b\n", "b.py": "import a\n"})
    assert stack.graphs.cycles(eager)["cycles"] == [["a", "b"]]
    assert stack.graphs.cycles(eager, load_time_only=True)["count"] == 1
    lazy = stack.repo({"a.py": "import b\n", "b.py": "def f():\n    import a\n"})
    assert stack.graphs.cycles(lazy)["count"] == 1
    assert stack.graphs.cycles(lazy, load_time_only=True)["count"] == 0


def test_class_hierarchy_and_search():
    stack = Stack()
    repo = stack.repo()
    hierarchy = stack.semantics.hierarchy(repo).to_dict()
    assert {item["qualified_name"] for item in hierarchy["classes"]} >= {"app.core.Base", "app.core.Sub"}
    hits = stack.semantics.find(repo, text="comp", kinds=["function"], path_glob="app/*.py")
    assert [hit["qualified_name"] for hit in hits["hits"]] == ["app.core.compute"]
    grep = stack.semantics.find(repo, text="compute", grep=True, path_glob="app/other.py")
    assert grep["count"] == 1
    with pytest.raises(SoftwareError):
        stack.semantics.find(repo, text="x", kinds=["spaceship"])


# --------------------------------------------------------------------------- analyses

def test_static_rules_and_noqa_suppression():
    stack = Stack()
    cases = {
        "SW101": "def f():\n    return missing_name\n",
        "SW102": "import os\n",
        "SW103": "def f():\n    x = 1\n    return 2\n",
        "SW104": "def f():\n    pass\n\n\ndef f():\n    pass\n",
        "SW105": "def f(x=[]):\n    return x\n",
        "SW301": "def f(s):\n    return eval(s)\n",
    }
    for rule, source in cases.items():
        assert rule in codes(stack.analysis.static(stack.repo({"m.py": source}))), rule
    assert stack.analysis.static(stack.repo({"m.py": "import os  # noqa: SW102\n"})).findings == ()


def test_dead_code_spares_the_public_api_and_tests():
    stack = Stack()
    report = stack.analysis.dead_code(stack.repo())
    names = {item.qualified_name for item in report.items}
    assert names == {"app.core._dead"}


def test_change_impact_reaches_importers_and_selects_tests():
    stack = Stack()
    repo = stack.repo()
    impact = stack.analysis.impact(repo, symbols=["app.core.compute"])
    assert "app.other" in impact.impacted_modules
    assert "tests/test_core.py" in {item.path for item in impact.tests.selected}


def test_architecture_layers_forbidden_edges_and_cycles():
    stack = Stack()
    files = {"ui/__init__.py": "", "ui/v.py": "from core import x\n", "core/__init__.py": "", "core/x.py": "x = 1\n"}
    repo = stack.repo(files)
    layers = [{"name": "ui", "packages": ["ui*"]}, {"name": "core", "packages": ["core*"]}]
    assert stack.analysis.architecture(repo, layers=layers).layer_violations == ()
    inverted = stack.repo(dict(files, **{"core/x.py": "import ui.v\n"}))
    assert stack.analysis.architecture(inverted, layers=layers).layer_violations
    forbidden = stack.analysis.architecture(repo, forbidden=[{"from": "ui*", "to": "core*", "reason": "no"}])
    assert [item["rule"] for item in forbidden.to_dict()["layer_violations"]] == ["no"]


def test_compatibility_classifies_signature_changes():
    stack = Stack()
    before = stack.repo()
    changed = dict(SAMPLE, **{"app/core.py": SAMPLE["app/core.py"].replace("def compute(a):", "def compute(a, b):")})
    report = stack.semantics.compatibility(before, stack.repo(changed))
    assert report.recommended_bump == "major"
    assert [item.kind for item in report.changes if item.level == "breaking"] == ["required_parameter_added"]
    assert stack.semantics.compatibility(before, before).changes == ()


# --------------------------------------------------------------------------- patch proposals

def test_rename_rewrites_definition_imports_and_aliased_uses():
    stack = Stack()
    repo = stack.repo()
    result = stack.refactoring.rename(repo, "app.core.compute", "calculate")
    diff = result.proposal.diff
    assert "+from .core import calculate" in diff
    assert "+from app.core import calculate as c" in diff
    assert "+def calculate(a):" in diff
    assert "-def compute(self" not in diff
    assert "Sub.compute" not in diff


def test_rename_rejects_conflicts_keywords_and_unknown_symbols():
    stack = Stack()
    repo = stack.repo({"m.py": "def f():\n    pass\n\n\ndef g():\n    pass\n"})
    with pytest.raises(SoftwareNameConflictError):
        stack.refactoring.rename(repo, "m.f", "g")
    with pytest.raises(SoftwareValidationError):
        stack.refactoring.rename(repo, "m.f", "class")
    with pytest.raises(SoftwareSymbolError):
        stack.refactoring.rename(repo, "m.missing", "z")


def test_proposals_are_never_applied_and_name_the_gates():
    stack = Stack()
    result = stack.refactoring.rename(stack.repo(), "app.core.user", "run_user")
    proposal = result.proposal
    assert proposal.applied is False
    assert proposal.requires_gates == HANDOFF_GATES == ("verification", "safety", "execution", "evaluation")
    assert proposal.validation["syntax"]["ok"] is True
    assert proposal.base_snapshot_digest != proposal.result_snapshot_digest
    assert stack.models.snapshot(files=SAMPLE).digest == proposal.base_snapshot_digest


def test_patch_lineage_and_chained_snapshots():
    stack = Stack()
    first = stack.refactoring.rename(stack.repo(), "app.core.user", "run_user")
    second = stack.refactoring.remove_unused_imports(stack.repo_of_patch(first.proposal.patch_id), parent_patch_id=first.proposal.patch_id)
    third = stack.refactoring.remove_dead_code(stack.repo_of_patch(second.proposal.patch_id), parent_patch_id=second.proposal.patch_id)
    assert [entry.patch_id for entry in stack.refactoring.lineage(third.proposal.patch_id)] == [
        first.proposal.patch_id,
        second.proposal.patch_id,
        third.proposal.patch_id,
    ]
    final = stack.repo_of_patch(third.proposal.patch_id).snapshot.file("app/core.py").text
    assert "def run_user" in final and "_dead" not in final and "import os" not in final


def test_unused_import_removal_keeps_used_noqa_and_future_imports():
    stack = Stack()
    source = "from __future__ import annotations\nimport os\nimport sys  # noqa: SW102\nfrom typing import List\nimport json\nprint(json.dumps({}))\n"
    result = stack.refactoring.remove_unused_imports(stack.repo({"m.py": source}))
    removed = {line for line in result.proposal.diff.splitlines() if line.startswith("-") and not line.startswith("---")}
    assert removed == {"-import os", "-from typing import List"}


def test_dead_code_removal_honours_symbol_filter_and_keeps_syntax_valid():
    stack = Stack()
    source = "def _a():\n    return 1\n\n\ndef _b():\n    return 2\n\n\ndef main():\n    return 3\n"
    repo = stack.repo({"m.py": source})
    only_a = stack.refactoring.remove_dead_code(repo, symbols=["m._a"], min_confidence=0.0)
    text = stack.repo_of_patch(only_a.proposal.patch_id).snapshot.file("m.py").text
    assert "_a" not in text and "_b" in text and "main" in text
    ast.parse(text)


def test_api_migration_keyword_and_symbol_rules():
    stack = Stack()
    repo = stack.repo({"m.py": "from lib import f\nf(1, old=2)\nf(old=1)\n"})
    keyword = stack.refactoring.migrate(repo, [{"kind": "rename_keyword", "old": "lib.f", "keyword": "old", "new": "renamed"}])
    assert "+f(1, renamed=2)" in keyword.proposal.diff and "+f(renamed=1)" in keyword.proposal.diff
    with pytest.raises(SoftwareError):
        stack.refactoring.migrate(repo, [{"kind": "teleport", "old": "lib.f"}])


def test_external_diffs_replay_reverse_and_are_validated():
    stack = Stack()
    repo = stack.repo({"app/core.py": "def user():\n    return 1\n"})
    made = stack.refactoring.rename(repo, "app.core.user", "run").proposal
    replay = stack.refactoring.from_diff(repo, made.diff).proposal
    assert replay.result_snapshot_digest == made.result_snapshot_digest
    assert replay.confidence < made.confidence
    after = stack.repo_of_patch(made.patch_id)
    assert stack.refactoring.from_diff(after, made.reverse_diff).proposal.result_snapshot_digest == made.base_snapshot_digest

    broken = made.diff.replace("+def run():", "+def run(:")
    with pytest.raises(SoftwarePatchError):
        stack.refactoring.from_diff(repo, broken)
    stale = made.diff.replace("-def user():", "-def other():")
    with pytest.raises(SoftwarePatchError):
        stack.refactoring.from_diff(repo, stale)


def test_regression_check_flags_newly_introduced_correctness_findings():
    stack = Stack()
    repo = stack.repo({"m.py": "VALUE = 1\n\n\ndef f():\n    return VALUE\n"})
    diff = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1,5 +1,5 @@\n"
        "-VALUE = 1\n+OTHER = 1\n \n \n def f():\n     return VALUE\n"
    )
    proposal = stack.refactoring.from_diff(repo, diff).proposal
    assert proposal.validation["static_regressions"]["introduced_count"] >= 1
