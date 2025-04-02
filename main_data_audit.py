import os
import sys
import json
import logging
import yaml
from typing import Tuple
import pandas as pd
from logs.logger import get_logger, get_log_queue
from modules.data_handler import DataHandler
from modules.compliance_auditor import ComplianceAuditor
from modules.monitoring import Monitoring
from modules.security_manager import SecurityManager
from collaborative.shared_memory import SharedMemory
from agents.safe_ai_agent import SafeAI_Agent

# Logging Setup
log_queue = get_log_queue()
logger = get_logger("main_data_audit")

# Load configuration
def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        logger.error(f"Missing configuration file: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)

def validate_paths(*paths):
    for path in paths:
        if not os.path.exists(path):
            logger.error(f"Missing required path: {path}")
            sys.exit(1)

def run_monitoring(shared_memory: SharedMemory, alert_config: dict) -> Monitoring:
    monitor = Monitoring(shared_memory=shared_memory, alert_config=alert_config)

    # Weighted scoring functions
    def calculate_accuracy_score(raw_accuracy: float) -> float:
        return min(max(raw_accuracy, 0.0), 1.0)

    def calculate_f1_score_score(raw_f1: float) -> float:
        return min(max(raw_f1, 0.0), 1.0)

    def calculate_risk_score(raw_risk: float) -> float:
        return 1.0 - min(max(raw_risk, 0.0), 1.0)  # Lower risk = higher score

    # Example values
    raw_accuracy = 0.82
    raw_f1_score = 0.79
    raw_risk_score = 0.31

    # Compute individual scores
    accuracy_score = calculate_accuracy_score(raw_accuracy)
    f1_score_score = calculate_f1_score_score(raw_f1_score)
    risk_score = calculate_risk_score(raw_risk_score)

    # Apply weights (e.g. accuracy=0.5, f1_score=0.3, risk_score=0.2)
    weights = {
        "accuracy": 0.5,
        "f1_score": 0.3,
        "risk_score": 0.2
    }

    weighted_composite_score = round(
        accuracy_score * weights["accuracy"] +
        f1_score_score * weights["f1_score"] +
        risk_score * weights["risk_score"],
        3
    )

    model_metrics = {
        "accuracy": raw_accuracy,
        "f1_score": raw_f1_score,
        "composite_score_weighted": weighted_composite_score
    }

    safe_ai_metrics = {
        "risk_score": raw_risk_score,
        "risk_level": round(risk_score, 3),
        "composite_score_weighted": weighted_composite_score
    }

    record_metrics(monitor, "model_trainer", model_metrics)
    record_metrics(monitor, "safe_ai", safe_ai_metrics)

    monitor.print_summary()
    return monitor

def run_compliance_audit(config_path: str, logs_path: str, output_dir: str):
    auditor = ComplianceAuditor(config_path=config_path, logs_path=logs_path, output_dir=output_dir)
    auditor.run_audit()
    if not auditor.is_compliant():
        logger.critical("Critical compliance violations detected. Aborting.")
        for v in auditor.get_report().get("violations", []):
            logger.critical(f"  - {v}")
            suggest_compliance_fix(v)  # Suggest fix for each violation
        sys.exit(1)
    logger.info("Compliance check passed.")

def load_and_validate_data(shared_memory: SharedMemory, dataset_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        data_handler = DataHandler(shared_memory=shared_memory)
        data = data_handler.load_data(dataset_path)
    except FileNotFoundError:
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}", exc_info=True)
        sys.exit(1)

    required_columns = ["gender", "age", "income", "label"]
    if not data_handler.validate_schema(data, required_columns):
        logger.error("Missing required columns in dataset.")
        sys.exit(1)

    logger.info("Dataset schema validated.")

    fairness_report = data_handler.check_data_fairness(data)
    logger.info("Fairness Report Summary:")
    for feature, dist in fairness_report.items():
        logger.info(f"  • {feature}: {dist}")

    features, labels = data_handler.preprocess_data(data)
    data_handler.export_data(features, labels, output_path=config["processed_output_path"])
    logger.info(f"Preprocessed dataset saved to: {config['processed_output_path']}")
    return features, labels

def perform_safeai_assessment(shared_memory: SharedMemory, risk_threshold: float):
    safe_ai = SafeAI_Agent(shared_memory=shared_memory, risk_threshold=risk_threshold)
    result = safe_ai.execute({
        "policy_risk_score": 0.27,
        "task_type": "reinforcement_learning"
    })

    logger.info("SafeAI Risk Assessment:")
    logger.info(json.dumps(result, indent=2))

    safe_ai.train()
    eval_summary = safe_ai.evaluate()
    logger.info("SafeAI Evaluation:")
    logger.info(json.dumps(eval_summary, indent=2))

    return result, eval_summary

def perform_security_checks(shared_memory: SharedMemory, policy: dict):
    sec_mgr = SecurityManager(shared_memory=shared_memory, policy_config=policy)
    if not sec_mgr.is_action_allowed("safe_ai", "can_export"):
        logger.info("[SECURITY] SafeAI is restricted from exporting data.")
    if sec_mgr.is_action_allowed("model_trainer", "can_access_data"):
        logger.info("[SECURITY] ModelTrainer has access to training data.")
    sec_mgr.print_report()

def finalize_monitoring(monitor: Monitoring, eval_summary: dict, result: dict):
    record_metrics(monitor, "model_trainer", {
        "accuracy": eval_summary.get("accuracy", 0.0),
        "f1_score": eval_summary.get("f1_score", 0.0)
    })
    record_metrics(monitor, "safe_ai", {
        "risk_score": result.get("risk_score", 0.0)
    })
    monitor.print_summary()

def record_metrics(monitor: Monitoring, component: str, metrics: dict):
    monitor.record(component, metrics)
    logger.info(f"Recorded {component} metrics: {metrics}")

def main():
    logger.info("=== SLAI Data Preflight Audit ===")
    global config
    config = load_config()
    validate_paths(config["dataset_path"], config["logs_path"], config["config_path"])

    shared_memory = SharedMemory()
    monitor = run_monitoring(shared_memory, alert_config=config.get("alert_thresholds", {}))

    run_compliance_audit(
        config_path=config["config_path"],
        logs_path=config["logs_path"],
        output_dir=config["audit_output_dir"]
    )

    features, labels = load_and_validate_data(shared_memory, config["dataset_path"])
    result, eval_summary = perform_safeai_assessment(shared_memory, config["safe_ai"]["risk_threshold"])
    perform_security_checks(shared_memory, config["security_policy"])
    finalize_monitoring(monitor, eval_summary, result)

    logger.info("=== SLAI Data Audit Complete ===")

def suggest_compliance_fix(violation: str):
    if "PII patterns detected" in violation:
        logger.warning("➡️  Suggestion: Scrub sensitive info (e.g. emails, SSNs, credit cards) from log files.")
        logger.warning("   Use regex filters or a log sanitization tool before writing logs.")
    elif "Data is not anonymized" in violation:
        logger.warning("➡️  Suggestion: Set `anonymize: true` in your config.yaml.")
    elif "User consent not enforced" in violation:
        logger.warning("➡️  Suggestion: Set `consent_required: true` in your config.yaml and implement a consent prompt.")
    elif "Config file not found" in violation:
        logger.warning("➡️  Suggestion: Ensure `config.yaml` exists and is properly formatted.")
    else:
        logger.warning("➡️  Suggestion: Please review the violation and ensure your data handling and privacy policies comply with GDPR/CCPA.")

if __name__ == "__main__":
    main()
