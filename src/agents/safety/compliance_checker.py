
import re, sys
import hashlib
import yaml, json
import numpy as np

from pathlib import Path
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication
from datetime import timedelta, datetime
from typing import Dict, List, Callable, Union, Any

from src.agents.safety.utils.config_loader import load_global_config, get_config_section
from src.agents.safety.secure_memory import SecureMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Security Compliance Checker")
printer = PrettyPrinter

class ComplianceChecker:
    def __init__(self):
        self.config = load_global_config()
        self.complience_config = get_config_section('compliance_checker')
        self.compliance_file_path =  self.complience_config.get('compliance_file_path')
        self.phishing_model_path =  self.complience_config.get('phishing_model_path')
        self.enable_memory_bootstrap =  self.complience_config.get('enable_memory_bootstrap')
        self.report_thresholds =  self.complience_config.get('report_thresholds', {
            'critical', 'warning'
        })
        self.weights =  self.complience_config.get('weights', {
            'data_security', 'model_security', 'app_security', 'operational_security'
        })

        self.memory = SecureMemory()
        self.memory.bootstrap_if_empty()

        self.compliance_framework = self._load_compliance_framework()
        
        logger.info("Compliance Checker initialized with framework version %s", 
                   self.compliance_framework.get('documentInfo', {}).get('version', 'unknown'))
    
    def _load_compliance_framework(self) -> Dict:
        """Load the compliance framework from secure storage or file"""
        # First try to recall from secure memory
        framework_entries = self.memory.recall(tag="compliance_framework", top_k=1)
        if framework_entries:
            return framework_entries[0]['data']
        
        # Fallback to file system
        path = self.compliance_file_path
        try:
            with open(path, 'r', encoding='utf-8') as f:
                framework = json.load(f)
                
            # Store in secure memory
            self.memory.add(
                framework,
                tags=["compliance_framework", "security"],
                sensitivity=0.8
            )
            return framework
        except Exception as e:
            logger.error(f"Failed to load compliance framework: {e}")
            return {}
    
    def evaluate_compliance(self) -> Dict[str, Any]:
        """Comprehensive compliance evaluation against all controls"""
        printer.status("CHECK", "Evaluating compliance", "info")

        if not self.compliance_framework:
            return {"status": "error", "message": "Compliance framework not loaded"}
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "sections": {},
            "overall_score": 0.0,
            "status": "pending"
        }
        
        total_controls = 0
        passed_controls = 0
        
        # Evaluate each section
        for section in self.compliance_framework.get('sections', []):
            section_id = section['sectionId']
            section_results = []
            
            # Handle controls in section
            if 'controls' in section:
                for control in section['controls']:
                    control_id = control['controlId']
                    status = self._evaluate_control(control)
                    section_results.append({
                        "controlId": control_id,
                        "objective": control['objective'],
                        "status": status,
                        "owner": control.get('owner', 'Unknown')
                    })
                    if status == "pass":
                        passed_controls += 1
                    total_controls += 1
            
            # Handle subsections
            if 'subsections' in section:
                for subsection in section['subsections']:
                    if 'controls' in subsection:
                        for control in subsection['controls']:
                            control_id = control['controlId']
                            status = self._evaluate_control(control)
                            section_results.append({
                                "controlId": control_id,
                                "objective": control['objective'],
                                "status": status,
                                "owner": control.get('owner', 'Unknown')
                            })
                            if status == "pass":
                                passed_controls += 1
                            total_controls += 1
            
            results['sections'][section_id] = {
                "title": section['title'],
                "results": section_results
            }
        
        # Calculate overall score
        if total_controls > 0:
            results['overall_score'] = passed_controls / total_controls
            results['status'] = self._get_compliance_status(results['overall_score'])
        
        # Store evaluation in secure memory
        self.memory.add(
            results,
            tags=["compliance_evaluation", "security"],
            sensitivity=0.7
        )
        
        return results
    
    def _evaluate_control(self, control: Dict) -> str:
        """Evaluate a single control using control-specific logic"""
        printer.status("CHECK", "Evaluating control", "info")

        control_id = control['controlId']
        
        # Map control IDs to evaluation methods
        control_mapping = {
            "DP-001": self._check_data_classification,
            "DP-002": self.check_gdpr,
            "DP-003": self._check_data_minimization,
            "MS-001": self._check_model_integrity,
            # Add more mappings as needed
        }
        
        evaluator = control_mapping.get(control_id, self._generic_control_check)
        return evaluator(control)

    def _generic_control_check(self, control: Dict) -> str:
        """Perform a generic compliance control check based on required memory tags and fields"""
        printer.status("CHECK", "Control check", "info")

        try:
            required_tags = control.get("required_tags", [])
            required_fields = control.get("required_fields", [])
    
            for tag in required_tags:
                entries = self.memory.recall(tag=tag, top_k=1)
                if not entries:
                    logger.warning(f"Missing required tag: {tag}")
                    return "fail"
    
                entry = entries[0]['data']
                for field in required_fields:
                    if field not in entry or entry[field] in [None, "", [], {}]:
                        logger.warning(f"Missing or invalid field '{field}' in tag '{tag}'")
                        return "fail"
    
            return "pass"
        except Exception as e:
            logger.error(f"Generic control check failed: {e}")
            return "fail"
    
    def _get_compliance_status(self, score: float) -> str:
        """Determine compliance status based on score"""
        printer.status("CHECK", "Status compliance", "info")

        thresholds = self.complience_config.get('report_thresholds', {})
        critical = thresholds.get('critical', 0.7)
        warning = thresholds.get('warning', 0.9)
        
        if score < critical:
            return "critical"
        elif score < warning:
            return "warning"
        return "compliant"
    
    # Control-specific evaluation methods
    def check_gdpr(self, control: Dict) -> str:
        """Validate GDPR compliance across key principles"""
        printer.status("CHECK", "Checking GDPR compliance", "info")

        try:
            # 1. Lawful basis for data processing
            basis = self.memory.recall(tag="consent_records", top_k=1)
            if not basis or not basis[0]['data'].get("consent_granted", False):
                logger.warning("Lawful basis (consent) missing or false")
                return "fail"
    
            # 2. Purpose limitation
            purpose_log = self.memory.recall(tag="data_usage_purpose", top_k=1)
            if not purpose_log or not purpose_log[0]['data'].get("declared_purpose"):
                logger.warning("Declared purpose for data processing not found")
                return "fail"
    
            # 3. Data minimization (reuse existing check)
            minimization_result = self._check_data_minimization({})
            if minimization_result != "pass":
                logger.warning("Data minimization principle violated")
                return "fail"
    
            # 4. Subject rights (access, deletion, correction)
            rights_log = self.memory.recall(tag="subject_requests", top_k=1)
            if not rights_log:
                logger.warning("No subject access requests found")
                return "fail"
    
            rights = rights_log[0]['data']
            if not all(k in rights for k in ["accessed", "corrected", "deleted"]):
                logger.warning("Incomplete handling of subject rights")
                return "fail"
    
            # 5. Retention policy
            retention = self.memory.recall(tag="retention_policy", top_k=1)
            if not retention or not retention[0]['data'].get("expiration_days"):
                logger.warning("Data retention policy not configured")
                return "fail"
    
            return "pass"
    
        except Exception as e:
            logger.error(f"GDPR check failed: {e}")
            return "fail"
    
    def check_hipaa(self, data: Dict) -> str:
        """Check HIPAA compliance"""
        printer.status("CHECK", "Checking HIPAA compliance", "info")

        return "pass" if 'PHI' not in data or data.get('encrypted', False) else "fail"
    
    def _check_data_classification(self, control: Dict) -> str:
        """Check if data classification is implemented and enforced"""
    
        try:
            classification_map = self.memory.recall(tag="data_classification", top_k=1)
            if not classification_map:
                logger.warning("No data classification map found in secure memory")
                return "fail"
    
            classifications = classification_map[0]['data']  # Should be a dict like {'email.body': 'Confidential'}
            required_fields = ["email.body", "training_features_url", "log_entries"]
    
            for field in required_fields:
                label = classifications.get(field)
                if not label or label not in ["Confidential", "Restricted", "Internal"]:
                    logger.warning(f"Missing or incorrect classification for: {field}")
                    return "fail"
    
            return "pass"
    
        except Exception as e:
            logger.error(f"Error during data classification check: {e}")
            return "fail"
    
    def _check_data_minimization(self, control: Dict) -> str:
        """Check whether data minimization principles are followed"""
        printer.status("CHECK", "Checking data minimization", "info")
    
        try:
            input_limit = self.config.get("adaptive_security", {}).get("input_size_limit", 2024)

            logs = self.memory.recall(tag="feature_extraction", top_k=1)
            if not logs:
                logger.warning("No feature extraction logs found")
                return "fail"
    
            features_used = logs[0]['data'].get("features", [])
            if not features_used or len(features_used) > 30:  # Heuristic threshold
                logger.warning("Excessive number of features used")
                return "fail"
    
            if logs[0]['data'].get("input_size") > input_limit:
                logger.warning("Feature extraction input size exceeds configured limit")
                return "fail"
    
            return "pass"
    
        except Exception as e:
            logger.error(f"Error during data minimization check: {e}")
            return "fail"

    def _check_model_integrity(self, control: Dict) -> str:
        """Verify model file integrity using cryptographic hashes"""
        printer.status("CHECK", "Verifying model file integrity using cryptographic hashes", "info")

        try:
            trusted = self.memory.recall(tag="trusted_hashes", top_k=1)
            if not trusted:
                logger.warning("Trusted hash store not found")
                return "fail"
    
            trusted_hashes = trusted[0]['data']  # Ex: {'phishing_model.json': 'abc123...'}
            model_dir = Path(self.phishing_model_path).parent
            
            for model_file, expected_hash in trusted_hashes.items():
                path = model_dir / model_file
                if not path.exists():
                    logger.warning(f"Model file missing: {model_file}")
                    return "fail"
            
                with open(path, "rb") as f:
                    content = f.read()
                    actual_hash = hashlib.sha256(content).hexdigest()
            
                if actual_hash != expected_hash:
                    logger.error(f"Hash mismatch for {model_file}")
                    return "fail"
    
            return "pass"
    
        except Exception as e:
            logger.error(f"Model integrity check failed: {e}")
            return "fail"
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive compliance report"""
        printer.status("CHECK", "Generating report", "info")

        report = [
            "# Security Compliance Report",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Overall Compliance Status**: {results.get('status', 'unknown')}",
            f"**Overall Score**: {results.get('overall_score', 0):.1%}",
            "---"
        ]
        
        # Section details
        for section_id, section_data in results.get('sections', {}).items():
            report.append(f"## {section_data['title']}")
            
            pass_count = sum(1 for r in section_data['results'] if r['status'] == 'pass')
            total_count = len(section_data['results'])
            section_score = pass_count / total_count if total_count > 0 else 0
            
            report.append(f"**Status**: {self._get_compliance_status(section_score)}")
            report.append(f"**Score**: {section_score:.1%} ({pass_count}/{total_count} controls passed)")
            report.append("### Control Status")
            
            for control in section_data['results']:
                status_icon = "✅" if control['status'] == "pass" else "❌"
                report.append(f"- {status_icon} **{control['controlId']}**: {control['objective']} (Owner: {control['owner']})")
            
            report.append("---")
        
        # Add summary and footer
        report.append("## Recommendations")
        if results.get('status') == "critical":
            report.append("- Immediate action required to address critical compliance gaps")
            report.append("- Conduct security audit within 48 hours")
        elif results.get('status') == "warning":
            report.append("- Address compliance gaps within 14 days")
            report.append("- Update security documentation and controls")
        else:
            report.append("- Maintain current security practices")
            report.append("- Schedule quarterly compliance review")
        
        report.append(f"\n---\n*Report generated by {self.__class__.__name__}*")
        
        return "\n".join(report)

# Usage Example
if __name__ == "__main__":
    print("\n=== Running Security Compliance Checker ===\n")
    
    checker = ComplianceChecker()
    results = checker.evaluate_compliance()
    report = checker.generate_report(results)
    
    print(report)
    print("\n=== Compliance Check Complete ===\n")
