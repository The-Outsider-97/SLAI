"""
Read-only graph queries over persisted SLAI derivation provenance.

The implementation follows OPM/W3C PROV graph semantics and OPQL's graph-level
query perspective.  It intentionally uses lightweight Python adjacency indexes
instead of adding a graph/database dependency.  Traversals are iterative,
deterministically ordered, and cycle guarded.
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional, Set

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.provenance_errors import ProvenanceGraphError, ProvenanceNotFoundError
from ..utils.provenance_helpers import require_identifier
from ..provenance_store import ProvenanceStore
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Lineage Graph")
printer = PrettyPrinter()


class LineageGraph:
    """Deterministic read-only view of the artifact derivation DAG."""

    def __init__(self, store: Optional[ProvenanceStore] = None) -> None:
        self.config = load_global_config()
        self.lineage_graph_config = get_config_section( "lineage_graph", config=self.config, default={})
        query = self.lineage_graph_config.get("query", {})
        self.max_depth = int(query.get("max_depth", 256)) if isinstance(query, Mapping) else 256
        if self.max_depth < 1:
            self.max_depth = 256
        self.store = store or ProvenanceStore()

    def _adjacency(self) -> tuple[Dict[str, Set[str]], Dict[str, Set[str]], List[dict[str, Any]]]:
        parents: Dict[str, Set[str]] = defaultdict(set)
        children: Dict[str, Set[str]] = defaultdict(set)
        records = self.store.get_lineage_records()
        for record in records:
            child = str(record["artifact_id"])
            parents.setdefault(child, set())
            children.setdefault(child, set())
            for parent in record.get("parent_artifact_ids") or ():
                parent_id = str(parent)
                parents[child].add(parent_id)
                children[parent_id].add(child)
                parents.setdefault(parent_id, set())
                children.setdefault(parent_id, set())
        return parents, children, records

    def _assert_known(self, artifact_id: str, parents: Mapping[str, Set[str]], children: Mapping[str, Set[str]]) -> str:
        artifact = require_identifier(artifact_id, field_name="artifact_id")
        if artifact not in parents and artifact not in children and self.store.get_entity(artifact, strict=False) is None:
            raise ProvenanceNotFoundError(
                "artifact is not present in the provenance graph",
                context={"artifact_id": artifact},
            )
        return artifact

    @staticmethod
    def _validate_acyclic(parents: Mapping[str, Set[str]]) -> None:
        """Raise if externally corrupted/malformed persisted data contains a cycle."""
        state: Dict[str, int] = {}
        for start in sorted(parents):
            if state.get(start) == 2:
                continue
            stack: list[tuple[str, bool]] = [(start, False)]
            while stack:
                node, exiting = stack.pop()
                if exiting:
                    state[node] = 2
                    continue
                marker = state.get(node, 0)
                if marker == 1:
                    raise ProvenanceGraphError(
                        "cycle detected in persisted provenance graph",
                        context={"artifact_id": node},
                    )
                if marker == 2:
                    continue
                state[node] = 1
                stack.append((node, True))
                for parent in sorted(parents.get(node, ()), reverse=True):
                    if state.get(parent) == 1:
                        raise ProvenanceGraphError(
                            "cycle detected in persisted provenance graph",
                            context={"artifact_id": node, "parent_id": parent},
                        )
                    if state.get(parent) != 2:
                        stack.append((parent, False))

    def direct_parents(self, artifact_id: str) -> List[str]:
        parents, children, _ = self._adjacency()
        artifact = self._assert_known(artifact_id, parents, children)
        self._validate_acyclic(parents)
        return sorted(parents.get(artifact, set()))

    parents = direct_parents

    def children(self, artifact_id: str) -> List[str]:
        parents, children, _ = self._adjacency()
        artifact = self._assert_known(artifact_id, parents, children)
        self._validate_acyclic(parents)
        return sorted(children.get(artifact, set()))

    def _walk(self, start: str, adjacency: Mapping[str, Set[str]], *, max_depth: Optional[int] = None) -> List[str]:
        limit = self.max_depth if max_depth is None else max(0, min(int(max_depth), self.max_depth))
        seen: Set[str] = {start}
        found: Set[str] = set()
        queue = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= limit:
                continue
            for neighbor in sorted(adjacency.get(node, set())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                found.add(neighbor)
                queue.append((neighbor, depth + 1))
        return sorted(found)

    def ancestors(self, artifact_id: str, *, max_depth: Optional[int] = None) -> List[str]:
        parents, children, _ = self._adjacency()
        artifact = self._assert_known(artifact_id, parents, children)
        self._validate_acyclic(parents)
        return self._walk(artifact, parents, max_depth=max_depth)

    def descendants(self, artifact_id: str, *, max_depth: Optional[int] = None) -> List[str]:
        parents, children, _ = self._adjacency()
        artifact = self._assert_known(artifact_id, parents, children)
        self._validate_acyclic(parents)
        return self._walk(artifact, children, max_depth=max_depth)

    def derivation_path(self, source_id: str, output_id: str) -> List[str]:
        """Return one deterministic shortest source→output derivation path."""
        parents, children, _ = self._adjacency()
        source = self._assert_known(source_id, parents, children)
        output = self._assert_known(output_id, parents, children)
        self._validate_acyclic(parents)
        if source == output:
            return [source]
        queue = deque([source])
        previous: Dict[str, Optional[str]] = {source: None}
        while queue:
            node = queue.popleft()
            for child in sorted(children.get(node, set())):
                if child in previous:
                    continue
                previous[child] = node
                if child == output:
                    path = [output]
                    cursor: Optional[str] = output
                    while cursor is not None and cursor != source:
                        cursor = previous[cursor]
                        if cursor is not None:
                            path.append(cursor)
                    return list(reversed(path))
                queue.append(child)
        return []

    def subgraph(
        self,
        artifact_id: str,
        *,
        depth: Optional[int] = None,
        include_descendants: bool = False,
    ) -> Dict[str, Any]:
        parents, children, records = self._adjacency()
        artifact = self._assert_known(artifact_id, parents, children)
        self._validate_acyclic(parents)
        nodes = {artifact, *self._walk(artifact, parents, max_depth=depth)}
        if include_descendants:
            nodes.update(self._walk(artifact, children, max_depth=depth))
        edges = []
        for child in sorted(nodes):
            for parent in sorted(parents.get(child, set())):
                if parent in nodes:
                    edges.append({"parent_id": parent, "artifact_id": child})
        relevant_records = [
            record for record in records
            if str(record.get("artifact_id")) in nodes
            and all(str(parent) in nodes for parent in record.get("parent_artifact_ids") or ())
        ]
        relevant_records.sort(key=lambda item: (item.get("timestamp", ""), item.get("record_id", "")))
        return {"root": artifact, "nodes": sorted(nodes), "edges": edges, "records": relevant_records}

    def _records_for_closure(self, artifact_id: str, recursive: bool) -> List[dict[str, Any]]:
        parents, children, records = self._adjacency()
        artifact = self._assert_known(artifact_id, parents, children)
        nodes = {artifact}
        if recursive:
            nodes.update(self._walk(artifact, parents))
        selected = [record for record in records if str(record.get("artifact_id")) in nodes]
        selected.sort(key=lambda item: (item.get("timestamp", ""), item.get("record_id", "")))
        return selected

    def activities(self, artifact_id: str, *, recursive: bool = True) -> List[dict[str, Any]]:
        ids = sorted({str(r["transformation_id"]) for r in self._records_for_closure(artifact_id, recursive) if r.get("transformation_id")})
        snapshot = self.store.snapshot().get("activities", {})
        transforms = self.store.snapshot().get("transformations", {})
        result = []
        for activity_id in ids:
            if activity_id in snapshot:
                result.append(dict(snapshot[activity_id]))
            elif activity_id in transforms:
                result.append(dict(transforms[activity_id]))
            else:
                result.append({"activity_id": activity_id})
        return result

    def sources(self, artifact_id: str, *, recursive: bool = True) -> List[dict[str, Any]]:
        ids = sorted({str(source) for r in self._records_for_closure(artifact_id, recursive) for source in (r.get("source_ids") or ())})
        return [source for source_id in ids if (source := self.store.get_source(source_id, strict=False)) is not None]

    def agents_involved(self, artifact_id: str, *, recursive: bool = True) -> List[dict[str, Any]]:
        ids = sorted({str(r["agent_id"]) for r in self._records_for_closure(artifact_id, recursive) if r.get("agent_id")})
        agents = self.store.snapshot().get("agents", {})
        return [dict(agents[agent_id]) if agent_id in agents else {"agent_id": agent_id} for agent_id in ids]

    def get_lineage_graph(self, artifact_id: str) -> Dict[str, Any]:
        graph = self.subgraph(artifact_id)
        graph["parents"] = self.direct_parents(artifact_id)
        graph["children"] = self.children(artifact_id)
        graph["ancestors"] = self.ancestors(artifact_id)
        graph["activities"] = self.activities(artifact_id)
        graph["sources"] = self.sources(artifact_id)
        graph["agents"] = self.agents_involved(artifact_id)
        return graph


__all__ = ["LineageGraph"]

if __name__ == "__main__":
    configure_logging()
    printer.status("SMOKE", "LineageGraph module loaded", "success")
