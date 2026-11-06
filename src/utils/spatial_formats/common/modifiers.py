"""Versioned, data-only modifier contracts shared by SGEO and SDRW.

This is the Phase 0 foundation, not a CAD Boolean/deformation kernel.  The
identity operation is deliberately the only executable modifier: it exercises
serialization, graph ordering, revisions and selective cache reuse without
claiming support for Phase 1 geometry algorithms.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from heapq import heappop, heappush

from .errors import ConflictError, ReferenceError, ResourceLimitError, UnsupportedFeatureError, ValidationError
from .helpers import MAX_DOCUMENT_NODES, canonical_json, digest, finite, json_safe
from .ids import require_id

MAX_MODIFIERS = 10_000
MAX_DEPENDENCIES = 64
MAX_GRAPH_DEPTH = 256
MAX_PARAMETERS_NODES = 1_000
MODIFIER_KINDS = frozenset({"sgeo.identity", "sdrw.identity"})
_REFERENCE_KEYS = frozenset({"document_id", "item_id", "revision", "subelement"})
_SUBELEMENT_KINDS = frozenset({"vertex", "edge", "face", "shell", "solid", "segment", "region", "anchor"})
_POLICY_MODES = frozenset({"exact", "tolerance_controlled", "bounded_approximation", "display_only"})
_POLICY_LIMITS = frozenset({"absolute", "relative", "angular", "max_error"})
_MODIFIER_KEYS = frozenset({"id", "kind", "version", "target", "operands", "parameters",
                            "enabled", "depends_on", "evaluation_policy"})


def validate_reference(reference, *, path="reference") -> dict:
    """Validate an inert reference; never resolve paths or external resources."""
    if not isinstance(reference, dict) or set(reference) - _REFERENCE_KEYS:
        raise ValidationError("invalid modifier reference", path=path)
    for key in ("document_id", "item_id"):
        require_id(reference.get(key))
    if "revision" in reference and (type(reference["revision"]) is not int or reference["revision"] < 1):
        raise ValidationError("reference revision must be positive", path=path)
    if "subelement" in reference:
        sub = reference["subelement"]
        if (not isinstance(sub, dict) or set(sub) != {"kind", "id"}
                or sub.get("kind") not in _SUBELEMENT_KINDS):
            raise ValidationError("invalid topological subelement", path=path)
        require_id(sub["id"])
    return reference


def reference_status(reference, *, document_id, revision, items) -> dict:
    """Report stable resolution states; subelement mapping is not implemented yet."""
    validate_reference(reference)
    if reference["document_id"] != document_id:
        return {"status": "external", "reason": "external references require an authorized resolver"}
    if reference["item_id"] not in items:
        return {"status": "missing", "reason": "referenced item does not exist"}
    if "revision" in reference and reference["revision"] != revision:
        return {"status": "stale", "reason": "pinned document revision differs"}
    if "subelement" in reference:
        return {"status": "unsupported", "reason": "topological lineage is not yet available"}
    return {"status": "resolved"}


def rebase_local_references(modifiers, *, previous_document_id, document_id, revision):
    """Keep local item identities when a document copy receives a new identity."""
    for modifier in modifiers:
        for ref in [modifier["target"], *modifier["operands"]]:
            if ref["document_id"] == previous_document_id:
                ref["document_id"] = document_id
                if "revision" in ref:
                    ref["revision"] = revision


def validate_evaluation_policy(policy, *, path="evaluation_policy") -> dict:
    if not isinstance(policy, dict) or set(policy) - ({"mode"} | _POLICY_LIMITS):
        raise ValidationError("invalid evaluation policy", path=path)
    if policy.get("mode") not in _POLICY_MODES:
        raise ValidationError("unsupported evaluation policy", path=path)
    for key in _POLICY_LIMITS & set(policy):
        if finite(policy[key], key, limit=1e9) < 0:
            raise ValidationError("negative evaluation tolerance", path=path)
    if policy["mode"] == "bounded_approximation" and "max_error" not in policy:
        raise ValidationError("bounded approximation requires max_error", path=path)
    return policy


def validate_modifier(value, *, format_id, path="modifier") -> dict:
    """Strict, bounded validation of the currently executable modifier vocabulary."""
    if not isinstance(value, dict) or set(value) != _MODIFIER_KEYS:
        raise ValidationError("modifier fields do not match version 1 contract", path=path)
    require_id(value.get("id"))
    kind = value.get("kind")
    if kind not in MODIFIER_KINDS or kind != format_id.split(".")[-1] + ".identity":
        raise UnsupportedFeatureError("modifier kind is not executable in Phase 0", path=path,
                                      context={"kind": kind if isinstance(kind, str) else None})
    if type(value.get("version")) is not int or value["version"] != 1:
        raise UnsupportedFeatureError("unsupported modifier contract version", path=path)
    if type(value.get("enabled")) is not bool:
        raise ValidationError("modifier enabled must be Boolean", path=path)
    validate_reference(value["target"], path=path + "/target")
    operands = value["operands"]
    if not isinstance(operands, list) or len(operands) > MAX_DEPENDENCIES:
        raise ResourceLimitError("too many modifier operands", path=path)
    for i, operand in enumerate(operands):
        validate_reference(operand, path=f"{path}/operands/{i}")
    deps = value["depends_on"]
    if (not isinstance(deps, list) or len(deps) > MAX_DEPENDENCIES or len(set(map(str, deps))) != len(deps)):
        raise ValidationError("invalid modifier dependencies", path=path)
    for dep in deps:
        require_id(dep)
    params = json_safe(value["parameters"], max_nodes=MAX_PARAMETERS_NODES)
    if not isinstance(params, dict) or params:
        raise ValidationError("identity modifier accepts no parameters", path=path)
    if operands:
        raise ValidationError("identity modifier accepts no operands", path=path)
    validate_evaluation_policy(value["evaluation_policy"], path=path + "/evaluation_policy")
    return value


@dataclass(frozen=True)
class ModifierGraph:
    """Validated dependency DAG, ordered deterministically by document order."""

    order: tuple[str, ...]
    dependencies: dict[str, frozenset[str]]
    dependents: dict[str, frozenset[str]]
    targets: dict[str, str]

    def affected(self, *, item_ids=(), modifier_ids=()) -> tuple[str, ...]:
        """Return only nodes transitively invalidated by changed sources."""
        affected = set(modifier_ids)
        changed = set(item_ids)
        affected.update(mid for mid, target in self.targets.items() if target in changed)
        queue = list(affected)
        for mid in queue:
            for child in self.dependents.get(mid, ()):
                if child not in affected:
                    affected.add(child)
                    queue.append(child)
        return tuple(mid for mid in self.order if mid in affected)


def apply_modifier_operation(modifiers, operation):
    """Mutate only a detached transaction candidate; final graph validation is atomic."""
    kind = operation.get("op")
    if kind == "add_modifier":
        if set(operation) != {"op", "value"}:
            raise ValidationError("invalid add_modifier operation")
        value = deepcopy(operation["value"])
        if not isinstance(value, dict):
            raise ValidationError("modifier must be an object")
        if any(m["id"] == value.get("id") for m in modifiers):
            raise ValidationError("duplicate modifier ID")
        if len(modifiers) >= MAX_MODIFIERS:
            raise ResourceLimitError("too many modifiers")
        modifiers.append(value)
        return value.get("id")
    if kind == "reorder_modifiers":
        if set(operation) != {"op", "ids"} or not isinstance(operation["ids"], list):
            raise ValidationError("invalid modifier reorder operation")
        ids = operation["ids"]
        if (len(ids) != len(modifiers) or not all(isinstance(mid, str) for mid in ids)
                or len(set(ids)) != len(ids) or set(ids) != {m["id"] for m in modifiers}):
            raise ValidationError("reorder must contain each modifier ID exactly once")
        mapping = {m["id"]: m for m in modifiers}
        modifiers[:] = [mapping[mid] for mid in ids]
        return list(ids)
    if kind in {"update_modifier", "remove_modifier"}:
        expected = {"op", "id", "value"} if kind == "update_modifier" else {"op", "id"}
        if set(operation) != expected:
            raise ValidationError("invalid modifier edit operation")
        mid = require_id(operation["id"])
        index = next((i for i, m in enumerate(modifiers) if m["id"] == mid), None)
        if index is None:
            raise ReferenceError("modifier does not exist")
        if kind == "remove_modifier":
            modifiers.pop(index)
        else:
            change = operation["value"]
            if not isinstance(change, dict) or not change or set(change) - (_MODIFIER_KEYS - {"id", "kind", "version"}):
                raise ValidationError("invalid modifier update fields")
            modifiers[index] = {**modifiers[index], **deepcopy(change)}
        return mid
    raise ValidationError("unknown modifier operation")


def validate_modifiers(modifiers, *, format_id, document_id, revision, items) -> ModifierGraph:
    """Validate references, contracts and graph cycles without executing geometry."""
    if not isinstance(modifiers, list) or len(modifiers) > MAX_MODIFIERS:
        raise ResourceLimitError("invalid modifier count")
    by_id = {}
    target_lists = defaultdict(list)
    for i, modifier in enumerate(modifiers):
        validate_modifier(modifier, format_id=format_id, path=f"/modifiers/{i}")
        mid = modifier["id"]
        if mid in by_id:
            raise ValidationError("duplicate modifier ID", path=f"/modifiers/{i}")
        by_id[mid] = modifier
        ref = modifier["target"]
        if ref["document_id"] != document_id or ref["item_id"] not in items:
            raise ReferenceError("modifier target must exist in this document", path=f"/modifiers/{i}/target")
        if "subelement" in ref:
            raise UnsupportedFeatureError("Phase 0 does not resolve topology targets", path=f"/modifiers/{i}/target")
        if ref.get("revision", revision) != revision:
            raise ConflictError("modifier target is pinned to a stale revision", path=f"/modifiers/{i}/target")
        target_lists[ref["item_id"]].append(mid)

    dependencies = {}
    previous_target = {}
    for i, modifier in enumerate(modifiers):
        mid = modifier["id"]
        deps = set(modifier["depends_on"])
        for dep in deps:
            if dep not in by_id:
                raise ReferenceError("unknown modifier dependency", path=f"/modifiers/{i}/depends_on")
        target = modifier["target"]["item_id"]
        if target in previous_target:
            deps.add(previous_target[target])
        previous_target[target] = mid
        for operand in modifier["operands"]:
            if operand["document_id"] != document_id or operand["item_id"] not in items:
                raise ReferenceError("unresolved modifier operand", path=f"/modifiers/{i}/operands")
            if operand["item_id"] == target:
                raise ValidationError("modifier cannot consume its own target as an operand", path=f"/modifiers/{i}")
            deps.update(target_lists[operand["item_id"]])
        if mid in deps:
            raise ValidationError("cyclic modifier dependency", path=f"/modifiers/{i}")
        dependencies[mid] = frozenset(deps)

    dependents = {mid: set() for mid in by_id}
    for mid, deps in dependencies.items():
        for dep in deps:
            dependents[dep].add(mid)
    index = {m["id"]: i for i, m in enumerate(modifiers)}
    indegree = {mid: len(deps) for mid, deps in dependencies.items()}
    ready = []
    for mid, degree in indegree.items():
        if degree == 0:
            heappush(ready, (index[mid], mid, 1))
    order = []
    levels = {}
    while ready:
        _, mid, depth = heappop(ready)
        if depth > MAX_GRAPH_DEPTH:
            raise ResourceLimitError("modifier dependency depth exceeds limit")
        order.append(mid)
        levels[mid] = depth
        for child in dependents[mid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                child_depth = 1 + max((levels[d] for d in dependencies[child]), default=0)
                heappush(ready, (index[child], child, child_depth))
    if len(order) != len(by_id):
        raise ValidationError("cyclic modifier dependency")
    return ModifierGraph(tuple(order), dependencies,
                         {mid: frozenset(ds) for mid, ds in dependents.items()},
                         {mid: mod["target"]["item_id"] for mid, mod in by_id.items()})


def evaluate_identity_modifiers(model, *, format_id, items) -> list[dict]:
    """Evaluate the Phase 0 identity operation; reuse unchanged results by fingerprint.

    Results and cache entries are detached from the authoritative source model.
    No derived geometry is serialized into SGEO/SDRW.
    """
    graph = validate_modifiers(model.modifiers, format_id=format_id, document_id=model.document_id,
                               revision=model.revision, items=items)
    cache = getattr(model, "_modifier_cache", None)
    if cache is None:
        cache = model._modifier_cache = {}
    modifiers = {m["id"]: m for m in model.modifiers}
    fingerprints = {}
    last_output = {}
    report = []
    for mid in graph.order:
        modifier = modifiers[mid]
        if not modifier["enabled"]:
            fingerprints[mid] = digest(canonical_json({"disabled": modifier}, max_nodes=MAX_DOCUMENT_NODES))
            report.append({"modifier_id": mid, "item_id": modifier["target"]["item_id"],
                           "status": "suppressed", "cache_hit": False, "fingerprint": fingerprints[mid],
                           "result": None})
            continue
        target = modifier["target"]
        state = reference_status(target, document_id=model.document_id, revision=model.revision, items=items)
        if state["status"] != "resolved":
            raise ReferenceError("modifier target could not be resolved", context={"status": state["status"]})
        if modifier["evaluation_policy"]["mode"] != "exact":
            raise UnsupportedFeatureError("identity evaluator supports exact mode only")
        key = target["item_id"]
        source = deepcopy(last_output.get(key, items[key]))
        fingerprint = digest(canonical_json({"modifier": modifier, "source": source,
                                            "dependencies": [fingerprints[d] for d in sorted(graph.dependencies[mid])]
                                            }, max_nodes=MAX_DOCUMENT_NODES))
        hit = mid in cache and cache[mid][0] == fingerprint
        value = deepcopy(cache[mid][1]) if hit else deepcopy(source)
        if not hit:
            cache[mid] = (fingerprint, deepcopy(value))
        fingerprints[mid] = fingerprint
        last_output[key] = value
        report.append({"modifier_id": mid, "item_id": key, "status": "ok", "cache_hit": hit,
                       "fingerprint": fingerprint, "result": deepcopy(value)})
    for stale in set(cache) - set(modifiers):
        cache.pop(stale, None)
    return report
