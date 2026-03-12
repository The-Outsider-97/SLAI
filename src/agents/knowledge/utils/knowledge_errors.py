import os
import json
import time
import hashlib
from enum import Enum
from typing import Dict, Any, Optional

class KnowledgeErrorType(Enum):
    RETRIEVAL_FAILURE = "Knowledge Retrieval Failure"
    ONTOLOGY_ERROR = "Ontology Management Error"
    BIAS_DETECTION_FAILURE = "Bias Detection Failure"
    GOVERNANCE_VIOLATION = "Governance Policy Violation"
    CACHE_ERROR = "Knowledge Cache Error"
    MEMORY_UPDATE_ERROR = "Memory Update Failure"
    INVALID_DOCUMENT = "Invalid Document Format"
    EMBEDDING_GENERATION_ERROR = "Embedding Generation Failure"
    ONTOLOGY_EXPANSION_FAILURE = "Ontology Query Expansion Failure"
    KNOWLEDGE_BROADCAST_FAILURE = "Knowledge Broadcast Failure"

class KnowledgeError(Exception):
    """Base exception for knowledge agent errors with forensic capabilities"""
    def __init__(
        self,
        error_type: KnowledgeErrorType,
        message: str,
        severity: str = "medium",
        context: Optional[Dict[str, Any]] = None,
        agent_state: Optional[Dict] = None,
        remediation: Optional[str] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.severity = severity
        self.context = context or {}
        self.agent_state = agent_state or {}
        self.remediation = remediation
        self.timestamp = time.time()
        self.error_id = self._generate_error_id()
        self.forensic_hash = self._generate_forensic_hash()

    def _generate_error_id(self) -> str:
        """Generate unique error ID using context and timestamp"""
        unique_str = f"{self.timestamp}{self.error_type.value}{json.dumps(self.context)}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:12]

    def _generate_forensic_hash(self) -> str:
        """Create verifiable hash of error context"""
        data = {
            "timestamp": self.timestamp,
            "error_id": self.error_id,
            "context": self.context,
            "agent_state": self.agent_state
        }
        return hashlib.sha3_256(json.dumps(data).encode()).hexdigest()

    def to_audit_dict(self) -> Dict[str, Any]:
        """Structured representation for logging and auditing"""
        return {
            "error_id": self.error_id,
            "type": self.error_type.value,
            "severity": self.severity,
            "message": str(self),
            "timestamp": self.timestamp,
            "forensic_hash": self.forensic_hash,
            "context": self.context,
            "agent_state_snapshot": self.agent_state,
            "remediation": self.remediation
        }

# Specific Error Classes
class RetrievalError(KnowledgeError):
    """Failure in knowledge retrieval process"""
    def __init__(self, query: str, reason: str, retrieval_mode: str):
        super().__init__(
            KnowledgeErrorType.RETRIEVAL_FAILURE,
            f"Retrieval failed for query: '{query}' ({reason})",
            severity="high",
            context={
                "failed_query": query,
                "retrieval_mode": retrieval_mode
            },
            remediation="Check retrieval configuration, embedding model availability, and document corpus"
        )

class OntologyError(KnowledgeError):
    """Failure in ontology operations"""
    def __init__(self, operation: str, subject: str, details: str):
        super().__init__(
            KnowledgeErrorType.ONTOLOGY_ERROR,
            f"Ontology {operation} failed for '{subject}': {details}",
            context={
                "operation": operation,
                "subject": subject,
                "details": details
            },
            remediation="Verify ontology relationships and ensure proper initialization"
        )

class BiasDetectionError(KnowledgeError):
    """Failure in bias detection subsystem"""
    def __init__(self, text_sample: str, error_details: str):
        super().__init__(
            KnowledgeErrorType.BIAS_DETECTION_FAILURE,
            f"Bias detection failed for text sample: {error_details}",
            severity="medium",
            context={
                "text_sample": text_sample[:200] + "..." if len(text_sample) > 200 else text_sample,
                "error": error_details
            },
            remediation="Check bias detection model and configuration thresholds"
        )

class GovernanceViolation(KnowledgeError):
    """Governance policy violation during knowledge operation"""
    def __init__(self, policy_name: str, violation_details: Dict, query: str):
        super().__init__(
            KnowledgeErrorType.GOVERNANCE_VIOLATION,
            f"Governance violation: {policy_name}",
            severity="critical",
            context={
                "policy": policy_name,
                "violation_details": violation_details,
                "offending_query": query
            },
            remediation="Review knowledge corpus and governance rules alignment"
        )

class CacheError(KnowledgeError):
    """Failure in knowledge cache operations"""
    def __init__(self, operation: str, key: str, error_details: str):
        super().__init__(
            KnowledgeErrorType.CACHE_ERROR,
            f"Cache {operation} failed for key '{key}': {error_details}",
            severity="medium",
            context={
                "cache_operation": operation,
                "cache_key": key,
                "error": error_details
            },
            remediation="Verify cache storage and serialization mechanisms"
        )

class MemoryUpdateError(KnowledgeError):
    """Failure in memory update operations"""
    def __init__(self, key: str, value: Any, error_details: str):
        super().__init__(
            KnowledgeErrorType.MEMORY_UPDATE_ERROR,
            f"Memory update failed for key '{key}'",
            severity="high",
            context={
                "memory_key": key,
                "attempted_value": str(value)[:100] + "..." if len(str(value)) > 100 else str(value),
                "error": error_details
            },
            remediation="Check shared memory connection and serialization formats"
        )

class InvalidDocumentError(KnowledgeError):
    """Invalid document format or content"""
    def __init__(self, document: Any, reason: str):
        super().__init__(
            KnowledgeErrorType.INVALID_DOCUMENT,
            f"Invalid document: {reason}",
            context={
                "document_type": type(document).__name__,
                "document_preview": str(document)[:200] + "..." if len(str(document)) > 200 else str(document),
                "rejection_reason": reason
            },
            remediation="Validate documents before ingestion: min_length=3, must be string"
        )

class EmbeddingError(KnowledgeError):
    """Failure in embedding generation"""
    def __init__(self, doc_id: str, model_name: str, error_details: str):
        super().__init__(
            KnowledgeErrorType.EMBEDDING_GENERATION_ERROR,
            f"Embedding failed for doc {doc_id} using {model_name}",
            severity="high",
            context={
                "doc_id": doc_id,
                "model": model_name,
                "error": error_details
            },
            remediation="Verify embedding model availability and input data format"
        )
