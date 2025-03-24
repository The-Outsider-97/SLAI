import os
import sys
import torch
import subprocess
from modules.data_handler import DataHandler
from modules.compliance_auditor import ComplianceAuditor
from collaborative.shared_memory import SharedMemory
from agents.safe_ai_agent import SafeAI_Agent

def main():
    print("\n=== SLAI Data Preflight Audit ===")

    # Shared memory for audit traceability
    shared_memory = SharedMemory()

    # === 1. Run Compliance Audit ===
    auditor = ComplianceAuditor(config_path="config.yaml", logs_path="logs/", output_dir="audits/")
    auditor.run_audit()

    if not auditor.is_compliant():
        print("[!] ⚠️  Compliance violations detected:")
        for v in auditor.get_report().get("violations", []):
            print("  -", v)

    # === 2. Load & Analyze Dataset ===
    dataset_path = "data/dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"[X] Dataset not found at path: {dataset_path}")
        return

    data_handler = DataHandler(shared_memory=shared_memory)

    data = data_handler.load_data(dataset_path)
    schema_valid = data_handler.validate_schema(
        data,
        required_columns=["gender", "age", "income", "label"]
    )

    if not schema_valid:
        print("[X] Missing required columns. Aborting audit.")
        return

    # === 3. Fairness & Distribution Check ===
    fairness_report = data_handler.check_data_fairness(data)

    print("\n[✓] Fairness Report Summary:")
    for feature, dist in fairness_report.items():
        print(f"  • {feature}: {dist}")

    # === 4. Preprocessing ===
    features, labels = data_handler.preprocess_data(data)
    data_handler.export_data(features, labels, output_path="data/processed_dataset.csv")

    print("\n[✓] Preprocessed dataset saved to: data/processed_dataset.csv")

    # === 5. SafeAI Agent Analysis (Optional) ===
    safe_ai = SafeAI_Agent(shared_memory=shared_memory, risk_threshold=0.2)
    result = safe_ai.execute({
        "policy_risk_score": 0.27,
        "task_type": "reinforcement_learning"
    })

    print("\n[✓] SafeAI Risk Assessment:")
    print(result)

    # === 6. Optional Training Phase ===
    safe_ai.train()
    eval_summary = safe_ai.evaluate()

    print("\n[✓] SafeAI Evaluation:")
    print(eval_summary)

    print("\n=== SLAI Data Audit Complete ===\n")

if __name__ == "__main__":
    main()
