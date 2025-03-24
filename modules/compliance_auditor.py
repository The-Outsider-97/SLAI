# modules/compliance_auditor.py

import logging
import os
import json
import re
from datetime import datetime

logger = logging.getLogger('SafeAI.ComplianceAuditor')
logger.setLevel(logging.INFO)

class ComplianceAuditor:
    """
    Performs automated compliance audits for GDPR/CCPA:
    - Scans for sensitive logging
    - Validates data anonymization settings
    - Checks for user consent mechanisms
    """

    def __init__(self, config_path="config.yaml", logs_path="logs/", output_dir="audits/"):
        self.config_path = config_path
        self.logs_path = logs_path
        self.output_dir = output_dir
        self.violations = []
        self.report = {}

        os.makedirs(self.output_dir, exist_ok=True)

    def run_audit(self):
        logger.info("Starting full compliance audit...")
        try:
            self.violations.clear()
            self.report = {}

            self._check_data_policies()
            self._scan_logs()
            self._generate_report()

            logger.info("Compliance audit complete. Report generated.")
        except Exception as e:
            logger.error(f"Compliance audit failed: {e}")
            raise

    def _check_data_policies(self):
        logger.info("Checking anonymization and consent policies...")

        try:
            with open(self.config_path, "r") as f:
                content = f.read()

                if "anonymize: false" in content:
                    self.violations.append("Data is not anonymized in config.")
                if "consent_required: false" in content:
                    self.violations.append("User consent not enforced.")

            self.report["data_policies"] = "OK" if not self.violations else "Non-Compliant"
        except FileNotFoundError:
            self.violations.append("Config file not found.")
            self.report["data_policies"] = "Missing"

    def _scan_logs(self):
        logger.info("Scanning logs for personal data leaks...")
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",  # email
            r"\b(?:\d[ -]*?){13,16}\b",  # credit cards
        ]

        flagged_entries = []

        for root, _, files in os.walk(self.logs_path):
            for file in files:
                if file.endswith(".log"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        for line in f:
                            for pattern in pii_patterns:
                                if re.search(pattern, line, re.IGNORECASE):
                                    flagged_entries.append(line.strip())

        if flagged_entries:
            self.violations.append("PII patterns detected in logs.")
            self.report["log_scan"] = {
                "status": "Non-Compliant",
                "entries": flagged_entries[:5]  # limit report size
            }
        else:
            self.report["log_scan"] = {"status": "OK"}

    def _generate_report(self):
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = os.path.join(self.output_dir, f"compliance_report_{timestamp}.json")

        self.report["violations"] = self.violations
        self.report["compliant"] = not bool(self.violations)

        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=4)

        logger.info(f"Compliance report written to {report_path}")

    def get_report(self):
        return self.report

    def is_compliant(self):
        return not bool(self.violations)
