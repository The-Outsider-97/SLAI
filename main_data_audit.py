import os, sys
import logging
import json
import queue
from logs.logger import get_logger, get_log_queue
from modules.data_handler import DataHandler
from modules.compliance_auditor import ComplianceAuditor
from modules.monitoring import Monitoring
from modules.security_manager import SecurityManager
from collaborative.shared_memory import SharedMemory
from agents.safe_ai_agent import SafeAI_Agent

log_queue = get_log_queue()

def main():
    print("\n=== SLAI Data Preflight Audit ===")

    # Shared memory and monitoring setup
    shared_memory = SharedMemory()
    monitor = Monitoring(shared_memory=shared_memory, alert_config={"accuracy": 0.75, "risk_score": 0.25})

    # Step 1: Log initial metrics (placeholder)
    monitor.record("model_trainer", {"accuracy": 0.82, "f1_score": 0.79})
    monitor.record("safe_ai", {"risk_score": 0.31})
    monitor.print_summary()

    # Step 2: Compliance check
    auditor = ComplianceAuditor(config_path="config.yaml", logs_path="logs/", output_dir="audits/")
    auditor.run_audit()

    if not auditor.is_compliant():
        print("[!] ⚠️  Compliance violations detected:")
        for v in auditor.get_report().get("violations", []):
            print(f"  - {v}")
    else:
        print("[✓] Compliance check passed.")

    # Step 3: Load dataset
    dataset_path = "datasets/dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"[X] Dataset not found at path: {dataset_path}")
        return

    data_handler = DataHandler(shared_memory=shared_memory)
    data = data_handler.load_data(dataset_path)

    # Step 4: Validate schema
    required_columns = ["gender", "age", "income", "label"]
    schema_valid = data_handler.validate_schema(data, required_columns)
    if not schema_valid:
        print("[X] Missing required columns. Aborting audit.")
        return
    print("[✓] Dataset schema validated.")

    # Step 5: Fairness and distribution check
    fairness_report = data_handler.check_data_fairness(data)
    print("\n[✓] Fairness Report Summary:")
    for feature, dist in fairness_report.items():
        print(f"  • {feature}: {dist}")

    # Step 6: Preprocess and export data
    features, labels = data_handler.preprocess_data(data)
    output_path = "data/processed_dataset.csv"
    data_handler.export_data(features, labels, output_path=output_path)
    print(f"\n[✓] Preprocessed dataset saved to: {output_path}")

    # Step 7: SafeAI Risk Evaluation
    safe_ai = SafeAI_Agent(shared_memory=shared_memory, risk_threshold=0.2)
    result = safe_ai.execute({
        "policy_risk_score": 0.27,
        "task_type": "reinforcement_learning"
    })

    print("\n[✓] SafeAI Risk Assessment:")
    print(json.dumps(result, indent=2))

    # Step 8: Security Policy Checks
    policy = {
        "safe_ai": {
            "can_access_data": False,
            "can_modify_model": False,
            "can_export": False
        },
        "model_trainer": {
            "can_access_data": True,
            "can_modify_model": True,
            "can_export": True
        }
    }

    sec_mgr = SecurityManager(shared_memory=shared_memory, policy_config=policy)
    if not sec_mgr.is_action_allowed("safe_ai", "can_export"):
        print("[SECURITY] SafeAI is restricted from exporting data.")

    if sec_mgr.is_action_allowed("model_trainer", "can_access_data"):
        print("[SECURITY] ModelTrainer has access to training data.")

    sec_mgr.print_report()

    # Step 9: SafeAI Training and Evaluation
    safe_ai.train()
    eval_summary = safe_ai.evaluate()
    print("\n[✓] SafeAI Evaluation:")
    print(json.dumps(eval_summary, indent=2))

    # Step 10: Final Monitoring Logs
    monitor.record("model_trainer", {
        "accuracy": eval_summary.get("accuracy", 0.0),
        "f1_score": eval_summary.get("f1_score", 0.0)
    })
    monitor.record("safe_ai", {
        "risk_score": result.get("risk_score", 0.0)
    })
    monitor.print_summary()

    print("\n=== SLAI Data Audit Complete ===")

if __name__ == "__main__":
    main()
