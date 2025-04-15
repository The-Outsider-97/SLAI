class InterpretabilityHelper:
    """Helper to explain evaluation results in plain language"""

    @staticmethod
    def explain_performance(score: float, threshold: float = 0.8) -> str:
        if score >= threshold:
            return f"Performance is acceptable ({score:.2f}), above the required threshold of {threshold}."
        return f"Performance is low ({score:.2f}), action required to improve reliability."

    @staticmethod
    def explain_risk(risk: dict) -> str:
        mean = risk.get('mean', 0.0)
        std = risk.get('std_dev', 0.0)
        risk_level = "HIGH" if mean > 0.7 else "MODERATE" if mean > 0.4 else "LOW"
        return f"System risk is {risk_level}: mean={mean:.3f}, deviation={std:.3f}"

    @staticmethod
    def summarize_certification(cert_result: dict) -> str:
        status = cert_result.get("status", "UNKNOWN")
        level = cert_result.get("level", "UNSPECIFIED")
        if status == "PASSED":
            return f"Certification PASSED at level {level}. All requirements met."
        elif status == "FAILED":
            return f"Certification FAILED at level {level}. Unmet criteria detected."
        else:
            return f"Certification conditionally approved at level {level}. Requires further validation."
