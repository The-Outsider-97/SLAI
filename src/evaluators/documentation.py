"""
Implements immutable audit trails and documentation management
using blockchain-inspired techniques from:
- Nakamoto (2008) Bitcoin Whitepaper
- Gipp et al. (2015) Cryptocurrency-based document timestamping
"""

import hashlib
from datetime import datetime
from typing import List, Dict

class AuditBlock:
    """Single unit in the audit chain"""
    
    def __init__(self, data: Dict, previous_hash: str):
        self.timestamp = datetime.now()
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()
        
    def calculate_hash(self) -> str:
        """SHA-256 hash of block contents"""
        content = f"{self.timestamp}{self.data}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def mine_block(self, difficulty: int):
        """Proof-of-work mining simulation"""
        while self.hash[:difficulty] != "0"*difficulty:
            self.nonce += 1
            self.hash = self.calculate_hash()

class AuditTrail:
    """Immutable validation evidence ledger"""
    
    def __init__(self, difficulty: int = 4):
        self.chain = [self._create_genesis_block()]
        self.difficulty = difficulty
        
    def _create_genesis_block(self) -> AuditBlock:
        """Initial block in the chain"""
        return AuditBlock({"message": "GENESIS BLOCK"}, "0")
    
    def add_document(self, document: Dict) -> bool:
        """Add validated document to the chain"""
        new_block = AuditBlock(
            data=document,
            previous_hash=self.chain[-1].hash
        )
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        return self._validate_chain()
    
    def _validate_chain(self) -> bool:
        """Verify chain integrity"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            if current.hash != current.calculate_hash():
                return False
            if current.previous_hash != previous.hash:
                return False
        return True
    
    def generate_verification_report(self) -> Dict:
        """ISO 9001-compliant audit report"""
        return {
            "chain_length": len(self.chain),
            "first_block": self.chain[0].timestamp,
            "last_block": self.chain[-1].timestamp,
            "tamper_status": "CLEAN" if self._validate_chain() else "COMPROMISED",
            "documents": [
                {
                    "timestamp": block.timestamp,
                    "content_type": list(block.data.keys())[0],
                    "block_hash": block.hash
                } 
                for block in self.chain[1:]  # Skip genesis
            ]
        }
