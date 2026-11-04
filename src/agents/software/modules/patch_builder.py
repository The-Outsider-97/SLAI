"""Patch synthesis: turn edits into a verified, reversible, *unapplied* proposal.

The builder never writes to disk.  It applies :class:`TextEdit` objects (or
whole-file replacements) to an immutable :class:`RepositorySnapshot`, proves
that the touched Python files still parse, renders forward and reverse unified
diffs, and returns a :class:`PatchProposal` whose ``requires_gates`` lists the
pipeline stages (verification, safety, execution, evaluation) that must act
before anything changes in the real world.  It also ingests unified diffs
produced elsewhere (Browser/Reader) so they are validated the same way.
"""

from __future__ import annotations

__version__ = "2.4.0"

import re

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from ..models.repository_model import RepositorySnapshot
from ..software_types import (
    FileChange,
    FileOperation,
    ParseStatus,
    PatchProposal,
    RefactoringKind,
    SourceFile,
    TextEdit,
)
from ..utils.software_errors import SoftwareLimitError, SoftwarePatchError
from ..utils.software_helpers import (
    PYTHON_SUFFIXES,
    content_digest,
    count_diff_lines,
    module_name_for_path,
    normalize_repo_path,
    opt_bool,
    opt_int,
    split_lines_keepends,
    unified_diff,
)
from .parser import SoftwareParser
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Patch Builder")

Validator = Callable[[RepositorySnapshot, RepositorySnapshot, tuple[str, ...]], tuple[Mapping[str, Any], Sequence[str], Sequence[str]]]


@dataclass(frozen=True, slots=True)
class PatchLimits:
    max_files: int = 200
    max_changed_lines: int = 20_000
    max_diff_bytes: int = 4 * 1024 * 1024
    context_lines: int = 3
    reject_syntax_errors: bool = True

    @classmethod
    def from_config(cls, section: Optional[Mapping[str, Any]] = None) -> "PatchLimits":
        data = section or {}
        return cls(
            max_files=opt_int(data, "max_files", 200, minimum=1),
            max_changed_lines=opt_int(data, "max_changed_lines", 20_000, minimum=1),
            max_diff_bytes=opt_int(data, "max_diff_bytes", 4 * 1024 * 1024, minimum=1024),
            context_lines=opt_int(data, "context_lines", 3, minimum=0, maximum=20),
            reject_syntax_errors=opt_bool(data, "reject_syntax_errors", True),
        )


@dataclass(frozen=True, slots=True)
class PatchResult:
    proposal: PatchProposal
    snapshot: RepositorySnapshot


def apply_edits(text: str, edits: Sequence[TextEdit]) -> str:
    """Apply non-overlapping edits expressed against ``text`` (offsets are characters)."""

    ordered = sorted(enumerate(edits), key=lambda item: (item[1].start, item[1].end, item[0]))
    out: list[str] = []
    cursor = 0
    for _, edit in ordered:
        if not 0 <= edit.start <= edit.end <= len(text):
            raise SoftwarePatchError("edit range is outside the file", context={"path": edit.path, "start": edit.start, "end": edit.end, "length": len(text)})
        if edit.start < cursor:
            raise SoftwarePatchError("edits overlap", context={"path": edit.path, "start": edit.start, "previous_end": cursor})
        out.append(text[cursor:edit.start])
        out.append(edit.replacement)
        cursor = edit.end
    out.append(text[cursor:])
    return "".join(out)


def _label(prefix: str, path: str, missing: bool) -> str:
    return "/dev/null" if missing else f"{prefix}/{path}"


def file_diff(path: str, old: Optional[str], new: Optional[str], *, context: int = 3) -> str:
    body = unified_diff(old or "", new or "", _label("a", path, old is None), _label("b", path, new is None), context=context)
    if not body:
        return ""
    return f"diff --git a/{path} b/{path}\n" + body


class PatchBuilder:
    def __init__(self, parser: SoftwareParser, limits: Optional[PatchLimits] = None) -> None:
        self.parser = parser
        self.limits = limits or PatchLimits()

    # -- build -------------------------------------------------------------------------
    def build(
        self,
        snapshot: RepositorySnapshot,
        *,
        kind: RefactoringKind,
        intent: str,
        edits: Iterable[TextEdit] = (),
        replacements: Optional[Mapping[str, Optional[str]]] = None,
        risks: Iterable[str] = (),
        warnings: Iterable[str] = (),
        unresolved_sites: int = 0,
        confidence: float = 1.0,
        parent_patch_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        validator: Optional[Validator] = None,
    ) -> PatchResult:
        limits = self.limits
        by_path: dict[str, list[TextEdit]] = {}
        for edit in edits:
            path = normalize_repo_path(edit.path)
            if path not in snapshot:
                raise SoftwarePatchError("edit targets a file that is not in the snapshot", context={"path": path})
            by_path.setdefault(path, []).append(edit)
        replaced = {normalize_repo_path(k): v for k, v in (replacements or {}).items()}
        overlap = set(by_path) & set(replaced)
        if overlap:
            raise SoftwarePatchError("a file cannot be both edited and replaced", context={"paths": sorted(overlap)[:5]})
        if len(by_path) + len(replaced) > limits.max_files:
            raise SoftwareLimitError("patch touches too many files", context={"files": len(by_path) + len(replaced), "limit": limits.max_files})

        new_texts: dict[str, Optional[str]] = {}
        file_edits: dict[str, tuple[TextEdit, ...]] = {}
        for path, items in by_path.items():
            old_text = snapshot.file(path).text
            new_texts[path] = apply_edits(old_text, items)
            file_edits[path] = tuple(sorted(items, key=lambda e: (e.start, e.end)))
        for path, text in replaced.items():
            if text is None and path not in snapshot:
                raise SoftwarePatchError("cannot delete a file that is not in the snapshot", context={"path": path})
            new_texts[path] = text

        changes: list[FileChange] = []
        changed_changes: dict[str, Optional[str]] = {}
        diffs: list[str] = []
        reverse: list[str] = []
        for path in sorted(new_texts):
            old_file = snapshot.get(path)
            old_text = old_file.text if old_file else None
            new_text = new_texts[path]
            if old_text == new_text:
                continue
            operation = FileOperation.ADD if old_file is None else FileOperation.DELETE if new_text is None else FileOperation.MODIFY
            diff = file_diff(path, old_text, new_text, context=limits.context_lines)
            diffs.append(diff)
            reverse.append(file_diff(path, new_text, old_text, context=limits.context_lines))
            changed_changes[path] = new_text
            changes.append(
                FileChange(
                    path=path,
                    operation=operation,
                    base_digest=old_file.digest if old_file else None,
                    new_digest=content_digest(new_text) if new_text is not None else None,
                    edits=file_edits.get(path, ()),
                    new_text=new_text if operation is FileOperation.ADD else None,
                    diff=diff,
                )
            )
        diff_text = "".join(diffs)
        added, removed = count_diff_lines(diff_text)
        if added + removed > limits.max_changed_lines:
            raise SoftwareLimitError("patch changes too many lines", context={"lines": added + removed, "limit": limits.max_changed_lines})
        if len(diff_text.encode("utf-8", errors="replace")) > limits.max_diff_bytes:
            raise SoftwareLimitError("patch diff is too large", context={"limit": limits.max_diff_bytes})

        result_snapshot = snapshot.with_changes(changed_changes)
        changed_paths = tuple(sorted(changed_changes))
        validation: dict[str, Any] = self._validate_syntax(snapshot, result_snapshot, changed_paths)
        risk_list, warning_list = list(risks), list(warnings)
        if validator is not None and changed_paths:
            extra, more_risks, more_warnings = validator(snapshot, result_snapshot, changed_paths)
            validation.update(extra)
            risk_list.extend(more_risks)
            warning_list.extend(more_warnings)
        confidence *= float(validation.pop("confidence_factor", 1.0))
        if not changes:
            warning_list.append("the patch is empty: the edits do not change any file")
        validation["no_op"] = not changes

        patch_id = "patch-" + content_digest(f"{snapshot.digest}|{kind.value}|{diff_text}")[:20]
        proposal = PatchProposal(
            patch_id=patch_id,
            kind=kind,
            intent=intent,
            base_snapshot_digest=snapshot.digest,
            result_snapshot_digest=result_snapshot.digest,
            files=tuple(changes),
            diff=diff_text,
            reverse_diff="".join(reverse),
            stats={"files": len(changes), "additions": added, "deletions": removed, "edits": sum(len(c.edits) for c in changes)},
            validation=validation,
            risks=tuple(dict.fromkeys(risk_list)),
            warnings=tuple(dict.fromkeys(warning_list)),
            unresolved_sites=int(unresolved_sites),
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            parent_patch_id=parent_patch_id,
            metadata=dict(metadata or {}),
        )
        return PatchResult(proposal, result_snapshot)

    def _validate_syntax(self, before: RepositorySnapshot, after: RepositorySnapshot, paths: Sequence[str]) -> dict[str, Any]:
        checked = 0
        broken: list[dict[str, Any]] = []
        preexisting: list[str] = []
        for path in paths:
            new_file = after.get(path)
            if new_file is None or not self.parser.supports(new_file.language):
                continue
            checked += 1
            name, is_package = module_name_for_path(path, after.source_roots) if path.endswith(PYTHON_SUFFIXES) else ("", False)
            model = self.parser.parse(new_file, module_name=name, is_package=is_package)
            if model.status is ParseStatus.OK:
                continue
            old_file = before.get(path)
            was_broken = False
            if old_file is not None:
                old_model = self.parser.parse(old_file, module_name=name, is_package=is_package)
                was_broken = old_model.status is not ParseStatus.OK
            if was_broken:
                preexisting.append(path)
            else:
                broken.append({"path": path, "diagnostics": [d.to_dict() for d in model.diagnostics][:3]})
        if broken and self.limits.reject_syntax_errors:
            raise SoftwarePatchError("the patch would introduce syntax errors", context={"files": broken[:5]})
        return {"syntax": {"checked": checked, "introduced_errors": broken, "preexisting_errors": preexisting, "ok": not broken}}

    # -- ingest ---------------------------------------------------------------------------
    def from_unified_diff(
        self,
        snapshot: RepositorySnapshot,
        diff_text: str,
        *,
        intent: str = "apply externally supplied patch",
        validator: Optional[Validator] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        parent_patch_id: Optional[str] = None,
    ) -> PatchResult:
        file_patches = parse_unified_diff(diff_text)
        replacements: dict[str, Optional[str]] = {}
        warnings: list[str] = []
        for fp in file_patches:
            if fp.new_path is None:
                if fp.old_path not in snapshot:
                    raise SoftwarePatchError("diff deletes a file that does not exist", context={"path": fp.old_path})
                replacements[fp.old_path] = None  # type: ignore[index]
                continue
            old_text = snapshot.file(fp.new_path).text if fp.old_path is not None and fp.old_path in snapshot else None
            if fp.old_path is not None and fp.old_path not in snapshot:
                raise SoftwarePatchError("diff modifies a file that does not exist", context={"path": fp.old_path})
            if fp.old_path is not None and fp.old_path != fp.new_path:
                raise SoftwarePatchError("renames are not supported in externally supplied diffs", context={"from": fp.old_path, "to": fp.new_path})
            if fp.old_path is None and fp.new_path in snapshot:
                raise SoftwarePatchError("diff creates a file that already exists", context={"path": fp.new_path})
            text, notes = apply_file_patch(old_text or "", fp)
            warnings.extend(f"{fp.new_path}: {note}" for note in notes)
            replacements[fp.new_path] = text
        return self.build(
            snapshot,
            kind=RefactoringKind.EXTERNAL_DIFF,
            intent=intent,
            replacements=replacements,
            warnings=warnings,
            risks=("patch originates outside the Software subsystem and was not generated from verified facts",),
            confidence=0.5,
            metadata={"source": "unified_diff", **dict(metadata or {})},
            parent_patch_id=parent_patch_id,
            validator=validator,
        )


# ---------------------------------------------------------------------------
# Unified-diff ingestion
# ---------------------------------------------------------------------------

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(slots=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[tuple[str, str, bool]] = field(default_factory=list)  # (marker, text without eol, no_newline)


@dataclass(slots=True)
class FilePatch:
    old_path: Optional[str]
    new_path: Optional[str]
    hunks: list[Hunk] = field(default_factory=list)


def _clean_path(raw: str) -> Optional[str]:
    raw = raw.split("\t", 1)[0].strip()
    if raw == "/dev/null":
        return None
    if len(raw) > 1 and raw[0] == '"' and raw[-1] == '"':
        raw = raw[1:-1]
    return raw


def parse_unified_diff(text: str) -> list[FilePatch]:
    lines = text.replace("\r\n", "\n").split("\n")
    patches: list[FilePatch] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(("rename from", "rename to", "copy from", "copy to")):
            raise SoftwarePatchError("renames and copies are not supported in diffs")
        if line.startswith(("Binary files", "GIT binary patch")):
            raise SoftwarePatchError("binary patches are not supported")
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            old_raw, new_raw = _clean_path(line[4:]), _clean_path(lines[i + 1][4:])
            strip = (old_raw is None or old_raw.startswith("a/")) and (new_raw is None or new_raw.startswith("b/"))
            if strip:
                old_raw = old_raw[2:] if old_raw else None
                new_raw = new_raw[2:] if new_raw else None
            patch = FilePatch(normalize_repo_path(old_raw) if old_raw else None, normalize_repo_path(new_raw) if new_raw else None)
            i += 2
            while i < len(lines):
                header = _HUNK.match(lines[i])
                if header is None:
                    break
                hunk = Hunk(int(header.group(1)), int(header.group(2) or 1), int(header.group(3)), int(header.group(4) or 1))
                i += 1
                seen_old = seen_new = 0
                while i < len(lines) and (seen_old < hunk.old_count or seen_new < hunk.new_count or lines[i].startswith("\\")):
                    entry = lines[i]
                    if entry.startswith("\\"):
                        if hunk.lines:
                            marker, body, _ = hunk.lines[-1]
                            hunk.lines[-1] = (marker, body, True)
                        i += 1
                        continue
                    marker, body = (entry[:1] or " "), entry[1:]
                    if marker not in " +-":
                        raise SoftwarePatchError("malformed hunk line", context={"line": entry[:80]})
                    hunk.lines.append((marker, body, False))
                    seen_old += marker in " -"
                    seen_new += marker in " +"
                    i += 1
                if seen_old != hunk.old_count or seen_new != hunk.new_count:
                    raise SoftwarePatchError("hunk line counts do not match its header", context={"header": lines[i - 1][:80] if i else ""})
                patch.hunks.append(hunk)
            patches.append(patch)
            continue
        i += 1
    if not patches:
        raise SoftwarePatchError("no file patches found in diff")
    return patches


def apply_file_patch(old_text: str, patch: FilePatch, *, max_offset: int = 200) -> tuple[str, list[str]]:
    """Apply a parsed file patch strictly (context must match; small line drift is tolerated)."""

    eol = "\r\n" if "\r\n" in old_text else "\n"
    source = split_lines_keepends(old_text)
    notes: list[str] = []
    output: list[str] = []
    cursor = 0
    drift = 0

    def stripped(entry: str) -> str:
        return entry.rstrip("\r\n")

    for number, hunk in enumerate(patch.hunks, start=1):
        expected = [body for marker, body, _ in hunk.lines if marker in " -"]
        want = max(0, hunk.old_start - 1 + (0 if hunk.old_count else 1)) + drift if hunk.old_count else hunk.old_start + drift

        def matches(at: int) -> bool:
            if at < cursor or at + len(expected) > len(source):
                return False
            return all(stripped(source[at + k]) == expected[k] for k in range(len(expected)))

        position: Optional[int] = None
        for delta in range(0, max_offset + 1):
            for candidate in {want + delta, want - delta}:
                if matches(candidate):
                    position = candidate
                    break
            if position is not None:
                break
        if position is None:
            raise SoftwarePatchError("hunk context does not match the file", context={"hunk": number, "expected_line": hunk.old_start})
        if position != want:
            notes.append(f"hunk {number} applied with an offset of {position - want} line(s)")
        drift = position - (hunk.old_start - 1) if hunk.old_count else drift
        output.extend(source[cursor:position])
        at = position
        for marker, body, no_newline in hunk.lines:
            if marker == " ":
                output.append(source[at])
                at += 1
            elif marker == "-":
                at += 1
            else:
                output.append(body if no_newline else body + eol)
        cursor = at
    output.extend(source[cursor:])
    return "".join(output), notes


__all__ = [
    "FilePatch",
    "Hunk",
    "PatchBuilder",
    "PatchLimits",
    "PatchResult",
    "Validator",
    "apply_edits",
    "apply_file_patch",
    "file_diff",
    "parse_unified_diff",
]
