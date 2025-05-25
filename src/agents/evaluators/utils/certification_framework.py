"""
Implements multi-level certification process based on:
- UL 4600 Standard for Safety for Autonomous Vehicles
- ISO 26262 Functional Safety
- EU AI Act risk classification
"""

import json, yaml
import hashlib

from datetime import datetime
from pathlib import Path
from enum import Enum, auto
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from logs.logger import get_logger

logger = get_logger("Certification Framework")

CONFIG_PATH = "src/agents/evaluators/configs/evaluator_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class CertificationLevel(Enum):
    """SAE J3016-inspired levels"""
    DEVELOPMENT = auto()
    PILOT = auto()
    DEPLOYMENT = auto()
    CRITICAL = auto()

class SafetyCase:
    """
    Safety case structure following UL 4600 format:
    - goals
    - argument
    - evidence
    """
    def __init__(self):
        self.case = {
            "goals": [],
            "arguments": [],
            "evidence": []
        }

    def add_goal(self, text):
        self.case["goals"].append(text)

    def add_argument(self, claim):
        self.case["arguments"].append(claim)

    def add_evidence(self, doc):
        self.case["evidence"].append(doc)

    def export(self):
        return self.case

@dataclass
class CertificationRequirement:
    """Individual certification criterion"""
    description: str
    test_method: str
    passing_condition: str
    evidence_required: List[str]

@dataclass
class CertificationStatus:
    quality_characteristics: Dict[str, str] = field(default_factory=dict)
    iso25010_pass: bool = False
    iso26262_asil: str = "ASIL-A"
    ul4600_valid: bool = False
    nist_rmf_risks: List[str] = field(default_factory=list)

class CertificationManager:
    """End-to-end certification lifecycle handler"""
    
    def __init__(self, config, domain: str = "automotive"):
        config = load_config() or {}
        self.config = config.get('certification_manager', {})
        self.domain = domain
        self.requirements = self._load_domain_requirements()
        self.current_level = CertificationLevel.DEVELOPMENT
        self.evidence_registry = []

        logger.info(f"Certification Manager succesfully initialized")
 
    def _load_domain_requirements(self) -> Dict[CertificationLevel, List[CertificationRequirement]]:
        """Load requirements from JSON template file"""
        template_path = Path(__file__).parent.parent.parent / self.config.get('template_path')
        print(f"Looking for templates at: {template_path}")

        try:
            with open(template_path, 'r') as f:
                templates = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load certification templates: {str(e)}")

        domain_templates = templates.get(self.domain, {})
        
        # Convert JSON structure to CertificationRequirement objects
        requirements = {}
        for level_name, req_list in domain_templates.items():
            try:
                level = CertificationLevel[level_name]
                requirements[level] = [
                    CertificationRequirement(
                        description=req["description"],
                        test_method=req["test_method"],
                        passing_condition=req["passing_condition"],
                        evidence_required=req["evidence_required"]
                    ) for req in req_list
                ]
            except KeyError:
                continue  # Skip invalid certification levels
                
        return requirements
    
    def submit_evidence(self, evidence: Dict):
        """Add validation evidence to certification package"""
        self.evidence_registry.append({
            "timestamp": evidence["timestamp"],
            "type": evidence["type"],
            "content_hash": hashlib.sha256(json.dumps(evidence).encode()).hexdigest()
        })
    
    def evaluate_certification(self) -> tuple[bool, List[str]]:
        """Check requirements for the current certification level"""
        unmet = []
        # Get requirements for the current level
        level_requirements = self.requirements.get(self.current_level, [])
        for req in level_requirements:
            if not any(self._matches_requirement(ev, req) for ev in self.evidence_registry):
                unmet.append(req.description)
        return (len(unmet) == 0, unmet)
    
    def _matches_requirement(self, evidence: Dict, requirement: CertificationRequirement) -> bool:
        """Check if evidence satisfies a requirement"""
        return all(
            doc_type in evidence["type"] 
            for doc_type in requirement.evidence_required
        )
    
    def generate_certificate(self) -> Dict:
        """ISO-compliant certification document"""
        passed, _ = self.evaluate_certification()
        return {
            "system": "AI Agent",
            "level": self.current_level.name,
            "status": "PASSED" if passed else "FAILED",
            "valid_until": "1 year from issuance",
            "requirements": [req.description for req in self.requirements[self.current_level]]
        }

class CertificationAuditor:
    """
    Verifies alignment with:
    - ISO 25010
    - ISO 26262 (ASIL-D target)
    - UL 4600 (safety case)
    - NIST RMF (metrics + risks)
    """
    def __init__(self):
        self.safety_case = SafetyCase()
        self.status = CertificationStatus()

        logger.info(f"Certification Auditor succesfully initialized")
 
    def assess_iso25010(self, metrics: Dict[str, float]):
        checks = {
            "reliability": metrics.get("mtbf", 0) > 1000,
            "performance": metrics.get("response_time", 1e6) < 1.0,
            "maintainability": metrics.get("tech_debt", 1.0) < 0.2,
            "security": metrics.get("vuln_count", 10) < 5
        }
        for k, passed in checks.items():
            self.status.quality_characteristics[k] = "Pass" if passed else "Fail"
        self.status.iso25010_pass = all(checks.values())
        return checks

    def evaluate_asil(self, coverage: float, test_count: int):
        if coverage >= 0.95 and test_count >= 10000:
            self.status.iso26262_asil = "ASIL-D"
        elif coverage >= 0.9:
            self.status.iso26262_asil = "ASIL-C"
        elif coverage >= 0.8:
            self.status.iso26262_asil = "ASIL-B"
        else:
            self.status.iso26262_asil = "ASIL-A"
        return self.status.iso26262_asil

    def finalize_ul4600(self, logs: List[str]):
        self.safety_case.add_goal("Prevent catastrophic failures in SLAI agents.")
        self.safety_case.add_argument("All critical paths are tested and analyzed.")
        for log in logs:
            self.safety_case.add_evidence(log)
        self.status.ul4600_valid = True
        return self.safety_case.export()

    def integrate_nist_rmf(self, metrics: Dict[str, float]):
        risks = []
        if metrics.get("distribution_shift", 0) > 0.2:
            risks.append("Concept drift detected")
        if metrics.get("fairness_score", 1.0) < 0.8:
            risks.append("Potential bias risk")
        self.status.nist_rmf_risks = risks
        return risks

    def generate_certificate_report(self):
        return {
            "timestamp": datetime.now().isoformat(),
            "quality_check": self.status.quality_characteristics,
            "ISO25010": self.status.iso25010_pass,
            "ASIL_Level": self.status.iso26262_asil,
            "UL4600_Safety_Case": self.safety_case.export(),
            "NIST_RMF_Risks": self.status.nist_rmf_risks
        }


# === Example usage ===
if __name__ == "__main__":
    print("\n=== Running Certification Framework ===\n")
    auditor = CertificationAuditor()

    # Simulated inputs
    iso_metrics = {
        "mtbf": 1200,
        "response_time": 0.8,
        "tech_debt": 0.15,
        "vuln_count": 2
    }
    print("ISO 25010 checks:", auditor.assess_iso25010(iso_metrics))
    print("ASIL Level:", auditor.evaluate_asil(0.96, 12000))
    print("UL 4600:", auditor.finalize_ul4600(["Unit test log", "Simulation results"]))
    print("NIST RMF:", auditor.integrate_nist_rmf({"distribution_shift": 0.25, "fairness_score": 0.75}))
    print("Final Report:", auditor.generate_certificate_report())

    print(f"\n* * * * * Phase 2 - Evidence Submission * * * * *\n")
    config = load_config()
    domain = "automotive"
    valid_evidence = {
        "timestamp": datetime.now().isoformat(),
        "type": ["safety_report", "test_logs"],
        "content": "Simulation passed 10k scenario tests"
    }

    manager = CertificationManager(config, domain=domain)
    manager.submit_evidence(valid_evidence)

    print(f"\n* * * * * Phase 3 - Certification Check * * * * *\n")
    passed, failures = manager.evaluate_certification()

    logger.info(f"{manager}")
    print(f"Certification Status: {'PASSED' if passed else 'FAILED'}")
    print(f"Unmet Requirements: {failures}")
    print("\n=== Successfully Ran Certification Framework ===\n")
