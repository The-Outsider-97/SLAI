"""
Production-ready documentation, audit-trail, and versioning support.

This module provides three main capabilities:
1. Schema-aware document validation driven by configuration
2. Immutable audit chains for evidence traceability
3. Retained version history for generated reports and artifacts
"""

from __future__ import annotations

import hashlib
import json
import yaml

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from jsonschema import ValidationError, validate

from ..utils.evaluation_errors import (DocumentationConfigurationError, UnsupportedExportFormatError,
                                DocumentationError, SchemaLoadError, InvalidDocumentError,
                                AuditIntegrityError)
from ..utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Documentation")
printer = PrettyPrinter


@dataclass(slots=True)
class AuditBlock:
    """Single immutable block in the audit chain."""

    index: int
    timestamp: str
    data: Dict[str, Any]
    previous_hash: str
    hash_algorithm: str = "sha256"
    nonce: int = 0
    block_hash: str = ""

    def __post_init__(self) -> None:
        self.timestamp = _coerce_timestamp(self.timestamp).isoformat()

        if not isinstance(self.data, dict):
            raise InvalidDocumentError("Audit block data must be a dictionary.")
        if not isinstance(self.previous_hash, str) or not self.previous_hash:
            raise InvalidDocumentError("Audit block previous_hash must be a non-empty string.")

        _get_hash_constructor(self.hash_algorithm)

        if not self.block_hash:
            self.block_hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate a deterministic hash of the block contents."""
        payload = {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }
        serialized = _canonical_json(payload)
        hash_constructor = _get_hash_constructor(self.hash_algorithm)
        return hash_constructor(serialized.encode("utf-8")).hexdigest()

    def mine_block(self, difficulty: int) -> str:
        """Perform proof-of-work style mining for demonstration or tamper resistance."""
        if not isinstance(difficulty, int) or difficulty < 0:
            raise ValueError("difficulty must be a non-negative integer.")

        target = "0" * difficulty
        while not self.block_hash.startswith(target):
            self.nonce += 1
            self.block_hash = self.calculate_hash()
        return self.block_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash_algorithm": self.hash_algorithm,
            "hash": self.block_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuditBlock":
        if not isinstance(payload, Mapping):
            raise InvalidDocumentError("Audit block payload must be a mapping.")

        return cls(
            index=int(payload["index"]),
            timestamp=str(payload["timestamp"]),
            data=dict(payload["data"]),
            previous_hash=str(payload["previous_hash"]),
            nonce=int(payload.get("nonce", 0)),
            hash_algorithm=str(payload.get("hash_algorithm", "sha256")),
            block_hash=str(payload.get("hash", payload.get("block_hash", ""))),
        )


@dataclass(slots=True)
class VersionRecord:
    """Versioned document snapshot with deterministic fingerprinting."""

    timestamp: str
    content: Dict[str, Any]
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    version_id: str = ""

    def __post_init__(self) -> None:
        self.timestamp = _coerce_timestamp(self.timestamp).isoformat()
        if not isinstance(self.content, dict):
            raise InvalidDocumentError("Versioned content must be a dictionary.")
        if not isinstance(self.metadata, dict):
            raise InvalidDocumentError("Version metadata must be a dictionary.")
        if not isinstance(self.content_hash, str) or len(self.content_hash) < 16:
            raise InvalidDocumentError("Version content hash is invalid.")
        if not self.version_id:
            seed = f"{self.timestamp}:{self.content_hash}"
            self.version_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "content": self.content,
            "hash": self.content_hash,
            "metadata": self.metadata,
            "version_id": self.version_id,
        }


class Documentation:
    """Shared configuration and validation services for documentation workflows."""

    def __init__(self) -> None:
        self.config = load_global_config()
        self.doc_config = get_config_section("documentation")
        self.audit_config = self.doc_config.get("audit_trail", {})
        self.export_config = self.doc_config.get("export", {})
        self.validation_config = self.doc_config.get("validation", {})

        self.schema = self._load_validation_schema()
        logger.info("Documentation successfully initialized")

    def _resolve_config_path(self, configured_path: str) -> Path:
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise DocumentationConfigurationError("Configured path must be a non-empty string.")

        candidate = Path(configured_path)
        if candidate.is_absolute():
            return candidate

        config_file = self.config.get("__config_path__")
        if not config_file:
            return candidate

        return Path(config_file).resolve().parent / candidate

    def _load_validation_schema(self) -> Optional[Dict[str, Any]]:
        schema_path = self.validation_config.get("schema_path")
        if not schema_path:
            logger.info("No validation schema configured; schema validation is disabled.")
            return None

        resolved_path = self._resolve_config_path(schema_path)
        try:
            with open(resolved_path, "r", encoding="utf-8") as handle:
                schema = json.load(handle)
        except FileNotFoundError as exc:
            raise SchemaLoadError(f"Validation schema not found: {resolved_path}") from exc
        except json.JSONDecodeError as exc:
            raise SchemaLoadError(f"Validation schema is not valid JSON: {resolved_path}") from exc
        except OSError as exc:
            raise SchemaLoadError(f"Unable to read validation schema: {resolved_path}") from exc

        if not isinstance(schema, dict):
            raise SchemaLoadError("Validation schema must deserialize to a JSON object.")
        return schema

    def validate_document(self, document: Mapping[str, Any]) -> bool:
        """Validate a document against the configured schema, if present."""
        if not isinstance(document, Mapping):
            raise InvalidDocumentError("Document must be a mapping.")

        if self.schema is None:
            return True

        try:
            validate(instance=dict(document), schema=self.schema)
            return True
        except ValidationError as exc:
            logger.error("Document validation failed: %s", exc)
            raise InvalidDocumentError(f"Document validation failed: {exc.message}") from exc


class AuditTrail(Documentation):
    """Immutable validation evidence ledger with export and integrity checks."""

    def __init__(self) -> None:
        super().__init__()
        self.hash_algorithm_name = self.audit_config.get("hash_algorithm", "sha256")
        _get_hash_constructor(self.hash_algorithm_name)

        difficulty = self.audit_config.get("difficulty", 4)
        if not isinstance(difficulty, int) or difficulty < 0:
            raise DocumentationConfigurationError(
                "documentation.audit_trail.difficulty must be a non-negative integer."
            )
        self.difficulty = difficulty

        formats = self.export_config.get("formats", ["json"])
        if not isinstance(formats, list) or not formats:
            raise DocumentationConfigurationError(
                "documentation.export.formats must be a non-empty list."
            )
        self.supported_formats = [str(item).strip().lower() for item in formats]
        self.default_format = str(self.export_config.get("default_format", "json")).strip().lower()
        if self.default_format not in self.supported_formats:
            raise DocumentationConfigurationError(
                "documentation.export.default_format must be included in documentation.export.formats."
            )

        self.chain: List[AuditBlock] = [self._create_genesis_block()]
        logger.info("Audit Trail successfully initialized")

    def _create_genesis_block(self) -> AuditBlock:
        """Create the initial chain block with bootstrap configuration metadata."""
        return AuditBlock(
            index=0,
            timestamp=_utcnow().isoformat(),
            data={
                "system": self.doc_config.get("system", "SLAI Core"),
                "message": "GENESIS BLOCK",
                "initial_parameters": {
                    "hash_algorithm": self.hash_algorithm_name,
                    "difficulty": self.difficulty,
                },
            },
            previous_hash="0" * 64,
            hash_algorithm=self.hash_algorithm_name,
        )

    def add_entry(
        self,
        document: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
        mine: bool = False,
        validate_evidence: bool = False,
    ) -> AuditBlock:
        """Validate and append a new block to the chain."""
        payload = dict(document)
        if validate_evidence:
            # Optional validation against a dedicated evidence schema
            self.validate_document(payload)
        else:
            logger.debug("Skipping evidence validation – no evidence schema configured")
    
        block_data = {
            "document": payload,
            "document_hash": hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
            "metadata": dict(metadata or {}),
        }
    
        block = AuditBlock(
            index=len(self.chain),
            timestamp=_utcnow().isoformat(),
            data=block_data,
            previous_hash=self.chain[-1].block_hash,
            hash_algorithm=self.hash_algorithm_name,
        )
        if mine:
            block.mine_block(self.difficulty)
    
        self.chain.append(block)
        logger.info("Audit block appended: index=%d hash=%s", block.index, block.block_hash)
        return block

    def verify_chain(self) -> bool:
        """Verify block hashes, links, and proof-of-work constraints."""
        if not self.chain:
            raise AuditIntegrityError("Audit chain is empty.")

        prefix = "0" * self.difficulty if self.difficulty > 0 else ""
        for index, block in enumerate(self.chain):
            recalculated = block.calculate_hash()
            if recalculated != block.block_hash:
                raise AuditIntegrityError(
                    f"Block hash mismatch detected at index {block.index}."
                )

            if index == 0:
                if block.previous_hash != "0" * 64:
                    raise AuditIntegrityError("Genesis block previous_hash is invalid.")
            else:
                previous = self.chain[index - 1]
                if block.previous_hash != previous.block_hash:
                    raise AuditIntegrityError(
                        f"Broken hash chain detected at block index {block.index}."
                    )

            if prefix and block.index > 0 and not block.block_hash.startswith(prefix):
                raise AuditIntegrityError(
                    f"Block {block.index} does not satisfy configured mining difficulty."
                )

        return True

    def export_chain(self, format: Optional[str] = None) -> str:
        """Export the audit trail in JSON or YAML format."""
        export_format = str(format or self.default_format).strip().lower()
        if export_format not in self.supported_formats:
            raise UnsupportedExportFormatError(
                f"Unsupported export format: {export_format}. Supported formats: {self.supported_formats}"
            )

        chain_data = [block.to_dict() for block in self.chain]
        if export_format == "json":
            return json.dumps(chain_data, indent=2, sort_keys=False)
        if export_format == "yaml":
            return yaml.safe_dump(chain_data, default_flow_style=False, sort_keys=False)

        raise UnsupportedExportFormatError(f"Unsupported export format: {export_format}")

    def load_chain(self, chain_payload: List[Mapping[str, Any]]) -> List[AuditBlock]:
        """Replace the current chain with a deserialized one and verify integrity."""
        if not isinstance(chain_payload, list) or not chain_payload:
            raise AuditIntegrityError("Serialized chain payload must be a non-empty list.")

        loaded_chain = [AuditBlock.from_dict(item) for item in chain_payload]
        self.chain = loaded_chain
        self.verify_chain()
        logger.info("Audit chain loaded and verified: length=%d", len(self.chain))
        return list(self.chain)


class DocumentVersioner(Documentation):
    """Manage document versions using bounded retention and deterministic hashes."""

    def __init__(self) -> None:
        super().__init__()
        versioning_config = self.doc_config.get("versioning", {})
        max_versions = versioning_config.get("max_versions", 7)
        if not isinstance(max_versions, int) or max_versions <= 0:
            raise DocumentationConfigurationError(
                "documentation.versioning.max_versions must be a positive integer."
            )

        self.max_versions = max_versions
        self.versions: List[VersionRecord] = []
        logger.info("Document Versioner successfully initialized")

    def add_version(
        self,
        document: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
        validate_schema: bool = True,
    ) -> VersionRecord:
        """Add a new document version and prune old versions according to retention."""
        payload = dict(document)
        if validate_schema:
            self.validate_document(payload)

        content_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        version = VersionRecord(
            timestamp=_utcnow().isoformat(),
            content=payload,
            content_hash=content_hash,
            metadata=dict(metadata or {}),
        )

        if len(self.versions) >= self.max_versions:
            removed = self.versions.pop(0)
            logger.info("Pruned old document version: %s", removed.version_id)

        self.versions.append(version)
        logger.info("Document version added: %s", version.version_id)
        return version

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return the latest document version as a serializable payload."""
        return self.versions[-1].to_dict() if self.versions else None

    def get_version_history(self) -> List[Dict[str, Any]]:
        """Return all retained versions in chronological order."""
        return [version.to_dict() for version in self.versions]

    def get_version_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Return a retained version matching the supplied content hash."""
        for version in self.versions:
            if version.content_hash == content_hash:
                return version.to_dict()
        return None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)



def _coerce_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise InvalidDocumentError("Timestamp must be a non-empty ISO-8601 string.")

    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise InvalidDocumentError(f"Invalid ISO-8601 timestamp: {value}") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)



def _canonical_json(payload: Any) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError) as exc:
        raise InvalidDocumentError(
            f"Document payload cannot be serialized deterministically: {exc}"
        ) from exc



def _get_hash_constructor(name: str):
    if not isinstance(name, str) or not name.strip():
        raise DocumentationConfigurationError("Hash algorithm name must be a non-empty string.")

    algorithm = name.strip().lower()
    try:
        return getattr(hashlib, algorithm)
    except AttributeError as exc:
        raise DocumentationConfigurationError(
            f"Unsupported hash algorithm configured for audit trail: {algorithm}"
        ) from exc


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Running Documentation ===\n")

    try:
        docs = Documentation()
        logger.info("%s", docs)

        print("\n* * * * * Phase 2 * * * * *\n")
        trail = AuditTrail()

        document = {
            "report_hash": "4db74ef020d228ea339a60eaeb1e19bbc1f5445c799717cffb1d2cc16fd83821",
            "metrics_snapshot": {
                "success_rate": 0.85,
                "current_risk": 0.02,
                "operational_time": 152.0,
            },
            "timestamp": _utcnow().isoformat(),
        }

        versioner = DocumentVersioner()
        version = versioner.add_version(document, metadata={"source": "evaluation_run"}, validate_schema=False)
        block = trail.add_entry(document, metadata={"pipeline": "nightly"}, mine=True)

        logger.info("Version stored: %s", version.version_id)
        logger.info("Block appended: %s", block.block_hash)
        logger.info("Audit chain verified: %s", trail.verify_chain())

        print("\n* * * * * Phase 3 * * * * *\n")
        printer.pretty("Latest Version:", versioner.get_latest(), "success")
        printer.pretty("Audit Chain JSON:", trail.export_chain("json"), "success")
        print("\n=== Successfully Ran Documentation ===\n")

    except DocumentationError as exc:
        logger.error("Documentation workflow failed: %s", exc)
        raise
