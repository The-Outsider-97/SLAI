
import time
import json
import re
import sqlite3

from pathlib import Path
from urllib.parse import quote
from typing import Dict, List, Optional, Tuple, Union
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS

from src.agents.knowledge.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Ontology Manager")
printer = PrettyPrinter

class OntologyManager:
    def __init__(self):
        self.config = load_global_config()
        self.enabled = self.config.get('enabled')
        self.namespace = self.config.get('namespace')

        self.manager_config = get_config_section('ontology_manager')
        self.knowledge_ontology_path = self.manager_config.get('knowledge_ontology_path')
        self.use_ontology_expansion = self.manager_config.get('use_ontology_expansion')
        self.output_path = self.manager_config.get('output_path')
        self.version_dir = self.manager_config.get('version_dir')

        self.graph = Graph()
        self.ns = Namespace(self.namespace)

        self.graph.bind("SLAI", self.ns)

        self.db_path = self.knowledge_ontology_path
        self._init_db()
        self._load_ontology_from_db()

        logger.info(f"Ontology Manager initialized with: {self.knowledge_ontology_path}")

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ontology (
                        id INTEGER PRIMARY KEY,
                        subject TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        object TEXT NOT NULL,
                        source TEXT DEFAULT 'internal',
                        last_updated REAL DEFAULT (strftime('%s', 'now')),
                        version INTEGER DEFAULT 1,
                        metadata JSON
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_subject ON ontology(subject)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_predicate ON ontology(predicate)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_object ON ontology(object)")
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise

    def _load_ontology_from_db(self) -> None:
        """Load existing ontology triples from database into RDF graph"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT subject, predicate, object FROM ontology")
                for subject, predicate, obj in cursor.fetchall():
                    subj_ref = URIRef(self.ns[subject])
                    pred_ref = URIRef(self.ns[predicate])
                    
                    # Handle class/type differently
                    if predicate in {'is_a', 'type', 'class'}:
                        obj_ref = URIRef(self.ns[obj])
                        self.graph.add((subj_ref, RDF.type, obj_ref))
                    else:
                        self.graph.add((subj_ref, pred_ref, Literal(obj)))
        except sqlite3.Error as e:
            logger.error(f"Failed to load ontology from DB: {str(e)}")

    def _safe_uri(self, s: str) -> str:
        """URL-encode string for safe use in URIs"""
        return quote(s, safe='')  # Encode special characters

    def add_triple(self, 
                  subject: str, 
                  predicate: str, 
                  obj: str, 
                  source: str = "internal", 
                  metadata: Optional[dict] = None) -> None:
        """Add triple to ontology with atomic DB+RDF sync"""
        try:
            # Validate inputs
            if not all([subject, predicate, obj]):
                raise ValueError("Subject, predicate, and object must be non-empty")

            # URL-encode all components
            safe_subject = self._safe_uri(subject)
            safe_predicate = self._safe_uri(predicate)

            # Prepare triple for RDF
            subj_ref = URIRef(self.ns[subject])
            pred_ref = URIRef(self.ns[predicate])
    
            if predicate in {'is_a', 'type', 'class'}:
                safe_obj = self._safe_uri(obj)
                obj_ref = URIRef(self.ns[safe_obj])
                self.graph.add((subj_ref, RDF.type, obj_ref))
            else:
                # Add original expression as literal
                self.graph.add((subj_ref, pred_ref, Literal(obj)))
                # Also store as a separate property if needed
                self.graph.add((subj_ref, self.ns["hasExpression"], Literal(obj)))
    
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO ontology (subject, predicate, object, source, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (subject, predicate, obj, source, json.dumps(metadata or {})))
            
            # Update RDF graph
            if predicate in {'is_a', 'type', 'class'}:
                obj_ref = URIRef(self.ns[obj])
                self.graph.add((subj_ref, RDF.type, obj_ref))
            else:
                self.graph.add((subj_ref, pred_ref, Literal(obj)))
                
        except (sqlite3.Error, ValueError) as e:
            logger.error(f"Failed to add triple ({subject}, {predicate}, {obj}): {str(e)}")
            raise

    def expand_query(self, terms: Union[str, List[str]]) -> set:
        """Expand query terms using ontology relationships"""
        if isinstance(terms, str):
            terms = [terms]
            
        expanded = set(terms)
        for term in terms:
            try:
                term_ref = URIRef(self.ns[term])
                
                # Get superclasses
                for superclass in self.graph.objects(term_ref, RDFS.subClassOf):
                    expanded.add(self._uri_to_term(superclass))
                
                # Get related entities
                for s, p, o in self.graph.triples((term_ref, None, None)):
                    if isinstance(o, URIRef):
                        expanded.add(self._uri_to_term(o))
            except Exception as e:
                logger.warning(f"Query expansion failed for term '{term}': {str(e)}")
                
        return expanded

    def _uri_to_term(self, uri: URIRef) -> str:
        """Convert URIRef to term name"""
        return str(uri).split("#")[-1]

    def get_relations(self, subject: str) -> List[Tuple[str, str]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT predicate, object 
                    FROM ontology 
                    WHERE subject = ?
                """, (subject,))
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Failed to get relations for {subject}: {str(e)}")
            return []

    def get_types(self, subject: str) -> List[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT object 
                    FROM ontology 
                    WHERE subject = ? 
                    AND predicate IN ('type', 'is_a', 'class')
                """, (subject,))
                return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Failed to get types for {subject}: {str(e)}")
            return []

    def export_to_rdf(self) -> None:
        """Export ontology to RDF/Turtle format"""
        try:
            export_path = Path(self.output_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            self.graph.serialize(destination=self.output_path, format='turtle')
            logger.info(f"Ontology exported to {self.output_path}")
        except Exception as e:
            logger.error(f"RDF export failed: {str(e)}")
            raise

    def version_ontology(self, version_notes: str) -> str:
        """Create versioned snapshot of ontology"""
        try:
            # Convert version_dir to Path and ensure it exists
            version_dir = Path(self.version_dir)
            version_dir.mkdir(parents=True, exist_ok=True)
    
            # Generate safe filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            safe_notes = re.sub(r'[^\w-]', '_', version_notes)[:50]
            filename = f"ontology_{timestamp}_{safe_notes}.ttl"
            output_path = version_dir / filename
    
            # Export directly to the versioned file
            self.graph.serialize(destination=output_path, format='turtle')
            return str(output_path)
    
        except Exception as e:
            logger.error(f"Versioning failed: {str(e)}")
            raise

if __name__ == "__main__":
    print("\n=== Ontology Manager Test ===")
    manager = OntologyManager()
    printer.status("Initialized:", manager)
    
    # Test adding and querying
    manager.add_triple("AI", "is_a", "Technology")
    manager.add_triple("MachineLearning", "subClassOf", "AI")
    printer.pretty("Relations for AI:", manager.get_relations("AI"))
    printer.status("Expanded query:", manager.expand_query(["MachineLearning"]))
    
    # Test versioning
    version_path = manager.version_ontology("pre-release")
    printer.status("Ontology version created:", version_path)
    
    print("\n=== Ontology Manager Test Completed ===\n")
