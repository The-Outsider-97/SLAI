
import os
import json, yaml
import re

from datetime import datetime

from logs.logger import get_logger

logger = get_logger('SafeAI.ComplianceAuditor')

CONFIG_PATH = "config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class ComplianceAuditor:
    """
    Performs automated compliance audits for GDPR/CCPA:
    - Scans for sensitive logging
    - Validates data anonymization settings
    - Checks for user consent mechanisms
    """

    def __init__(self, config, logs_path="logs/", output_dir="audits/"):
        config = load_config()
        self.config = config
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
            config_yaml_text = yaml.dump(self.config)

            if "anonymize: false" in config_yaml_text:
                self.violations.append("Data is not anonymized in config.")
            if "consent_required: false" in config_yaml_text:
                self.violations.append("User consent not enforced.")

            self.report["data_policies"] = "OK" if not self.violations else "Non-Compliant"
        except Exception as e:
            self.violations.append(f"Error checking data policies: {e}")
            self.report["data_policies"] = "Error"

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
                    try:
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            lines = f.readlines()
                    except UnicodeDecodeError:
                        with open(os.path.join(root, file), "r", encoding="latin-1") as f:
                            lines = f.readlines()
                            for line in lines:
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
        """
        Returns the current compliance audit report. If the report has not been generated yet,
        it compiles the current state into a structured dictionary.
    
        Ensures consistency by deferring compliance evaluation to is_compliant().
        """
        if not self.report:
            self.report = {
                "data_policies": "Unknown",
                "log_scan": "Unknown",
                "violations": self.violations,
                "compliant": self.is_compliant()
            }
        return self.report
    
    def is_compliant(self):
        """
        Returns True if no compliance violations are recorded, False otherwise.
    
        Acts as the single point of compliance status logic.
        """
        return len(self.violations) == 0

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running SLAI Compliance Auditor ===\n")
    config = load_config()
    logs_path="logs/"
    output_dir="audits/"

    auditor = ComplianceAuditor(config, logs_path=logs_path, output_dir=output_dir)
    logger.info(f"{auditor}")
    auditor.run_audit()
    print(f"\n* * * * * Phase 2 * * * * *\n")
    report = auditor.get_report()

    # Pretty-print the report
    logger.info("Audit Report:\n" + json.dumps(report, indent=4))
    print("\nAudit Report:")
    print(json.dumps(report, indent=4))
    print(f"\n* * * * * Phase 3 * * * * *\n")
    print("\n=== Successfully Ran SLAI Compliance Auditor ===\n")
