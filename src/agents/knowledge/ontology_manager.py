import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import quote, unquote

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

from src.agents.knowledge.utils.knowledge_errors import OntologyError
from src.agents.knowledge.utils.config_loader import get_config_section, load_global_config
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Ontology Manager")
printer = PrettyPrinter


class OntologyManager:
    """Manage ontology triples across SQLite storage and an RDF graph."""

    TYPE_PREDICATES = {"is_a", "type", "class"}
    SUBCLASS_PREDICATES = {"subclassof", "subclass_of", "subclass", "rdfs:subclassof"}
    LABEL_PREDICATES = {"label", "name", "rdfs:label"}

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        namespace: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        version_dir: Optional[Union[str, Path]] = None,
        graph: Optional[Graph] = None,
    ):
        self.config = load_global_config()
        self.enabled = bool(self.config.get("enabled", True))
        self.manager_config = get_config_section("ontology_manager") or {}
        self.namespace = namespace or self.config.get("namespace") or "http://slaiknowledge.org/ontology#"
        self.knowledge_ontology_path = str(
            db_path or self.manager_config.get("knowledge_ontology_path") or "knowledge_ontology.db"
        )
        self.use_ontology_expansion = bool(
            self.manager_config.get("use_ontology_expansion", True)
        )
        self.output_path = str(
            output_path or self.manager_config.get("output_path") or "ontology.ttl"
        )
        self.version_dir = str(
            version_dir or self.manager_config.get("version_dir") or Path(self.output_path).parent
        )

        self.graph = graph or Graph()
        self.ns = Namespace(self.namespace)
        self.graph.bind("SLAI", self.ns)

        self.db_path = str(self.knowledge_ontology_path)
        self._db_lock = threading.RLock()
        self._init_db()
        self._load_ontology_from_db()

        logger.info("Ontology Manager initialized with db=%s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        try:
            db_path = Path(self.db_path)
            if db_path.parent and str(db_path.parent) not in {"", "."}:
                db_path.parent.mkdir(parents=True, exist_ok=True)

            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ontology (
                        id INTEGER PRIMARY KEY,
                        subject TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        object TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'internal',
                        last_updated REAL NOT NULL DEFAULT (strftime('%s', 'now')),
                        version INTEGER NOT NULL DEFAULT 1,
                        metadata TEXT,
                        UNIQUE(subject, predicate, object, source)
                    )
                    """
                )
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_ontology_spos ON ontology(subject, predicate, object, source)"
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_subject ON ontology(subject)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_predicate ON ontology(predicate)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_object ON ontology(object)")
        except sqlite3.Error as exc:
            self._raise_ontology_error(
                operation="init_db",
                subject=self.db_path,
                details=f"Database initialization failed: {exc}",
                exc=exc,
            )

    def _load_ontology_from_db(self) -> None:
        """Load stored ontology triples from SQLite into the RDF graph."""
        try:
            self.graph.remove((None, None, None))
            with self._connect() as conn:
                cursor = conn.execute("SELECT subject, predicate, object FROM ontology")
                for subject, predicate, obj in cursor.fetchall():
                    self.graph.add(self._build_graph_triple(subject, predicate, obj))
        except sqlite3.Error as exc:
            self._raise_ontology_error(
                operation="load_ontology",
                subject=self.db_path,
                details=f"Failed to load ontology from DB: {exc}",
                exc=exc,
            )

    def _normalize_term(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise OntologyError(
                operation="normalize_term",
                subject=field_name,
                details=f"Expected string for {field_name}, received {type(value).__name__}",
            )
        normalized = value.strip()
        if not normalized:
            raise OntologyError(
                operation="normalize_term",
                subject=field_name,
                details=f"{field_name} must be non-empty",
            )
        return normalized

    def _safe_uri(self, value: str) -> str:
        return quote(self._normalize_term(value, "uri_component"), safe="")

    def _term_ref(self, value: str) -> URIRef:
        return URIRef(self.ns[self._safe_uri(value)])

    def _predicate_key(self, predicate: str) -> str:
        return self._normalize_term(predicate, "predicate").lower()

    def _predicate_ref(self, predicate: str) -> URIRef:
        predicate_key = self._predicate_key(predicate)
        if predicate_key in self.TYPE_PREDICATES:
            return RDF.type
        if predicate_key in self.SUBCLASS_PREDICATES:
            return RDFS.subClassOf
        if predicate_key in self.LABEL_PREDICATES:
            return RDFS.label
        return self._term_ref(predicate)

    def _build_graph_triple(self, subject: str, predicate: str, obj: str) -> Tuple[URIRef, URIRef, Union[URIRef, Literal]]:
        predicate_key = self._predicate_key(predicate)
        subj_ref = self._term_ref(subject)
        pred_ref = self._predicate_ref(predicate)

        if predicate_key in self.TYPE_PREDICATES or predicate_key in self.SUBCLASS_PREDICATES:
            obj_ref: Union[URIRef, Literal] = self._term_ref(obj)
        else:
            obj_ref = Literal(self._normalize_term(obj, "object"))

        return subj_ref, pred_ref, obj_ref

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: str = "internal",
        metadata: Optional[dict] = None,
    ) -> None:
        """Upsert a triple into SQLite and add its canonical representation to the RDF graph."""
        try:
            subject = self._normalize_term(subject, "subject")
            predicate = self._normalize_term(predicate, "predicate")
            obj = self._normalize_term(obj, "object")
            source = self._normalize_term(source, "source")
            triple = self._build_graph_triple(subject, predicate, obj)
            now = time.time()
            metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)

            with self._db_lock:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO ontology (subject, predicate, object, source, last_updated, version, metadata)
                        VALUES (?, ?, ?, ?, ?, 1, ?)
                        ON CONFLICT(subject, predicate, object, source)
                        DO UPDATE SET
                            last_updated=excluded.last_updated,
                            metadata=excluded.metadata,
                            version=ontology.version + 1
                        """,
                        (subject, predicate, obj, source, now, metadata_json),
                    )
                self.graph.add(triple)
        except OntologyError:
            raise
        except sqlite3.Error as exc:
            self._raise_ontology_error(
                operation="add_triple",
                subject=subject,
                details=f"Failed to persist triple ({subject}, {predicate}, {obj}): {exc}",
                exc=exc,
            )
        except Exception as exc:
            self._raise_ontology_error(
                operation="add_triple",
                subject=str(subject),
                details=f"Unexpected ontology error for ({subject}, {predicate}, {obj}): {exc}",
                exc=exc,
            )

    def expand_query(self, terms: Union[str, Iterable[str]]) -> set[str]:
        """Expand query terms using ontology relationships."""
        if isinstance(terms, str):
            terms = [terms]

        expanded: set[str] = set()
        for term in terms:
            try:
                normalized_term = self._normalize_term(term, "term")
                expanded.add(normalized_term)
                term_ref = self._term_ref(normalized_term)

                for superclass in self.graph.objects(term_ref, RDFS.subClassOf):
                    if isinstance(superclass, URIRef):
                        expanded.add(self._uri_to_term(superclass))

                for _, _, obj in self.graph.triples((term_ref, None, None)):
                    if isinstance(obj, URIRef):
                        expanded.add(self._uri_to_term(obj))
            except OntologyError as exc:
                logger.warning("Query expansion failed for term '%s': %s", term, exc)
        return expanded

    def _uri_to_term(self, uri: URIRef) -> str:
        uri_text = str(uri)
        if "#" in uri_text:
            uri_text = uri_text.split("#", 1)[1]
        else:
            uri_text = uri_text.rsplit("/", 1)[-1]
        return unquote(uri_text)

    def get_relations(self, subject: str) -> List[Tuple[str, str]]:
        try:
            subject = self._normalize_term(subject, "subject")
            with self._connect() as conn:
                cursor = conn.execute(
                    "SELECT predicate, object FROM ontology WHERE subject = ? ORDER BY predicate, object",
                    (subject,),
                )
                return cursor.fetchall()
        except OntologyError:
            raise
        except sqlite3.Error as exc:
            logger.error("Failed to get relations for %s: %s", subject, exc)
            return []

    def get_types(self, subject: str) -> List[str]:
        try:
            subject = self._normalize_term(subject, "subject")
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    SELECT object
                    FROM ontology
                    WHERE subject = ?
                      AND lower(predicate) IN ('type', 'is_a', 'class')
                    ORDER BY object
                    """,
                    (subject,),
                )
                return [row[0] for row in cursor.fetchall()]
        except OntologyError:
            raise
        except sqlite3.Error as exc:
            logger.error("Failed to get types for %s: %s", subject, exc)
            return []

    def export_to_rdf(self) -> None:
        try:
            export_path = Path(self.output_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            self.graph.serialize(destination=str(export_path), format="turtle")
            logger.info("Ontology exported to %s", export_path)
        except Exception as exc:
            self._raise_ontology_error(
                operation="export_to_rdf",
                subject=self.output_path,
                details=f"RDF export failed: {exc}",
                exc=exc,
            )

    def version_ontology(self, version_notes: str) -> str:
        try:
            version_dir = Path(self.version_dir)
            version_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            safe_notes = re.sub(r"[^\w-]", "_", version_notes or "snapshot")[:50] or "snapshot"
            output_path = version_dir / f"ontology_{timestamp}_{safe_notes}.ttl"
            self.graph.serialize(destination=str(output_path), format="turtle")
            return str(output_path)
        except Exception as exc:
            self._raise_ontology_error(
                operation="version_ontology",
                subject=version_notes,
                details=f"Versioning failed: {exc}",
                exc=exc,
            )

    def _raise_ontology_error(
        self,
        operation: str,
        subject: str,
        details: str,
        exc: Optional[BaseException] = None,
    ) -> None:
        error = OntologyError(operation=operation, subject=str(subject), details=details)
        try:
            error.report()
        except Exception:
            pass
        logger.error(details)
        if exc is not None:
            raise error from exc
        raise error


if __name__ == "__main__":  # pragma: no cover
    print("\n=== Ontology Manager Test ===")
    manager = OntologyManager()
    printer.status("Initialized:", manager)

    manager.add_triple("AI", "is_a", "Technology")
    manager.add_triple("Machine Learning", "subClassOf", "AI")
    printer.pretty("Relations for AI:", manager.get_relations("AI"))
    printer.status("Expanded query:", manager.expand_query(["Machine Learning"]))

    version_path = manager.version_ontology("pre-release")
    printer.status("Ontology version created:", version_path)

    print("\n=== Ontology Manager Test Completed ===\n")
