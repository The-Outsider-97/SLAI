"""
Implements immutable audit trails and documentation management
using blockchain-inspired techniques from:
- Nakamoto (2008) Bitcoin Whitepaper
- Gipp et al. (2015) Cryptocurrency-based document timestamping
"""

import hashlib
import yaml, json

from pathlib import Path
from typing import Dict, List, Optional
from jsonschema import ValidationError, validate
from datetime import datetime

from src.agents.evaluators.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Documentation")
printer = PrettyPrinter

class Documentation:
    def __init__(self):
        self.config = load_global_config()
        self.doc_config = get_config_section('documentation')
        self.audit_config = self.doc_config.get('audit_trail', {})

        self.schema = self._load_validation_schema()

        logger.info(f"Documentation succesfully initialized")
 
    def _load_validation_schema(self) -> Optional[Dict]:
        schema_path = self.doc_config.get('validation', {}).get('schema_path')
        if not schema_path:
            return None
            
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("Validation schema not found or invalid")
            return None


class AuditBlock(Documentation):
    """Single unit in the audit chain"""
    
    def __init__(self, data: Dict, previous_hash: str):
        super().__init__()
        self.timestamp = datetime.now()
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

        logger.info(f"Audit Block succesfully initialized")

    def calculate_hash(self) -> str:
        """SHA-256 hash of block contents"""
        content = f"{self.timestamp}{self.data}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def mine_block(self, difficulty: int):
        """Proof-of-work mining simulation"""
        while self.hash[:difficulty] != "0"*difficulty:
            self.nonce += 1
            self.hash = self.calculate_hash()

class AuditTrail(Documentation):
    """Immutable validation evidence ledger"""
    
    def __init__(self):
        super().__init__()
        self.hash_algorithm_name = self.audit_config.get('hash_algorithm', 'sha256')
        self.hash_algo = getattr(hashlib, self.hash_algorithm_name)
        self.difficulty = self.audit_config.get('difficulty', 4)
        self.chain = [self._create_genesis_block()]

        # Load export configuration
        self.export_config = self.doc_config.get('export', {})
        self.supported_formats = self.export_config.get('formats', ['json'])
        self.default_format = self.export_config.get('default_format', 'json')

        logger.info(f"Audit Trail succesfully initialized")

    def _create_genesis_block(self) -> AuditBlock:
        """Create initial block with system bootstrap parameters"""
        return AuditBlock(
            data={
                "system": "SLAI Core",
                "message": "GENESIS BLOCK",
                "initial_parameters": {
                    "hash_algorithm": self.hash_algorithm_name,
                    "difficulty": self.difficulty
                }
            },
            previous_hash="0"*64  # Standard genesis hash pattern
        )

    # Modified hash calculation to use configured algorithm
    def calculate_hash(self) -> str:
        content = f"{self.timestamp}{self.data}{self.previous_hash}{self.nonce}"
        return self.hash_algo(content.encode()).hexdigest()

    def validate_document(self, document: Dict) -> bool:
        """Validate document against schema if configured"""
        if not self.schema:
            return True
            
        try:
            validate(instance=document, schema=self.schema)
            return True
        except ValidationError as e:
            logger.error(f"Document validation failed: {str(e)}")
            return False

    def export_chain(self, format: str = None) -> str:
        """Export audit trail in specified format"""
        format = format or self.default_format
        
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported export format: {format}")

        chain_data = [block.__dict__ for block in self.chain]
        
        if format == "json":
            return json.dumps(chain_data, indent=2, default=str)
        if format == "yaml":
            return yaml.safe_dump(chain_data, default_flow_style=False)
        return ""

class DocumentVersioner(Documentation):
    """Manage document versions with retention policy"""
    
    def __init__(self):
        super().__init__()
        self.versions = []
        self.max_versions = self.config.get('versioning', {}).get('max_versions', 7)

    def add_version(self, document: Dict):
        """Add new document version with automatic pruning"""
        if len(self.versions) >= self.max_versions:
            self.versions.pop(0)
            
        self.versions.append({
            "timestamp": datetime.now(),
            "content": document,
            "hash": hashlib.sha256(json.dumps(document).encode()).hexdigest()
        })

    def get_latest(self) -> Optional[Dict]:
        """Get most recent document version"""
        return self.versions[-1] if self.versions else None

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Documentation ===\n")
    docs = Documentation()
    logger.info(docs)
    print(f"\n* * * * * Phase 2 * * * * *\n")
    data = None
    previous_hash = None

    trail = AuditTrail()
    block = AuditBlock(data, previous_hash)

    logger.info(f"{block}")
    logger.info(f"{trail}")
    print(f"\n* * * * * Phase 3 * * * * *\n")
    document = {
        "report_hash": "4db74ef020d228ea339a60eaeb1e19bbc1f5445c799717cffb1d2cc16fd83821",
        "metrics_snapshot": {
            "success_rate": 0.85,
            "current_risk": 0.02,
            "operational_time": 152.0
        },
        "timestamp": datetime.now().isoformat(),
        "previous_hash": "0"*64,
        "nonce": 42,
        "hash": "7f6a79bca7c94c71ee2d25340a1f2aa71979066a5e4729d68f43e8a59889aabc"
    }

    version = DocumentVersioner()
    version.add_version(document)
    block.calculate_hash()
    trail.validate_document(document)

    logger.info(f"01. {version.add_version(document)}")
    logger.info(f"02. {block.calculate_hash()}")
    logger.info(f"03. {trail.validate_document(document)}")
    print("\n=== Successfully Ran Documentation ===\n")
