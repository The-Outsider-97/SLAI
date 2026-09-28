"""Deterministic hierarchical mapping merge for SLAI configuration data.

The behavior matches the ``deepmerge.Merger`` policy previously used by
``src.utils.config_loader``:

* mapping + mapping -> recursively merge;
* all other same-key values -> the override value wins;
* type conflicts -> the override value wins.

Unlike the former call site, this implementation never mutates either input.
That is important for SLAI because the configuration defaults are class-level
state and a shallow copy can otherwise leak one user's overrides into later
``ConfigLoader`` instances.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


class RecursiveMappingError(ValueError):
    """Raised when a configuration mapping contains a recursive cycle."""


def merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached recursive merge of ``base`` and ``override``.

    Existing mapping branches are merged recursively. For scalar values,
    sequences, sets, ``None``, or mapping/non-mapping type conflicts, the value
    from ``override`` replaces the value from ``base``. Keys present only in
    ``base`` are retained; keys present only in ``override`` are appended in
    override iteration order.

    Parameters
    ----------
    base:
        Baseline mapping, typically SLAI defaults.
    override:
        Mapping whose values take precedence, typically user configuration.

    Returns
    -------
    dict[str, Any]
        A deep-detached merged dictionary. Mutating the result cannot mutate
        either input mapping.

    Raises
    ------
    TypeError
        If either root argument is not a mapping.
    RecursiveMappingError
        If either mapping contains a recursive mapping cycle. Cyclic
        configuration graphs are rejected deterministically rather than being
        allowed to fail later with unbounded recursive merge behavior.
    """

    if not isinstance(base, Mapping):
        raise TypeError(f"base must be a mapping, got {type(base).__name__}")
    if not isinstance(override, Mapping):
        raise TypeError(f"override must be a mapping, got {type(override).__name__}")

    _assert_acyclic_mapping(base)
    _assert_acyclic_mapping(override)
    return _merge_mappings(base, override)


def _assert_acyclic_mapping(root: Mapping[str, Any]) -> None:
    active: set[int] = set()

    def visit(value: Any) -> None:
        if not isinstance(value, Mapping):
            return

        identity = id(value)
        if identity in active:
            raise RecursiveMappingError("recursive mapping cycle detected while merging configuration")

        active.add(identity)
        try:
            for child in value.values():
                visit(child)
        finally:
            active.remove(identity)

    visit(root)


def _merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))

    for key, override_value in override.items():
        if key in result:
            base_value = result[key]
            if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
                result[key] = _merge_mappings(base_value, override_value)
                continue

        result[key] = deepcopy(override_value)

    return result
