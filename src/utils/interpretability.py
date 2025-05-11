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

    @staticmethod
    def explain_feature_importance(features: dict, top_n: int = 3) -> str:
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:top_n]
        explanation = "Most influential factors:\n"
        for i, (feature, importance) in enumerate(sorted_features, 1):
            explanation += f"{i}. {feature}: {importance:.2f} impact\n"
        return explanation

    @staticmethod
    def generate_compliance_report(metrics: dict) -> str:
        report = "Compliance Summary:\n"
        report += f"- Safety: {metrics.get('safety', 'Not assessed')}\n"
        report += f"- Security: {metrics.get('security', 'Not assessed')}\n"
        report += f"- Ethics: {metrics.get('ethics', 'Not assessed')}\n"
        report += f"Overall Status: {metrics.get('overall', 'Pending')}"
        return report

    @staticmethod
    def explain_confusion_matrix(matrix: dict) -> str:
        return (
            f"Error Analysis:\n"
            f"- True Positives: {matrix.get('tp', 0)}\n"
            f"- False Positives: {matrix.get('fp', 0)} (Type I errors)\n"
            f"- True Negatives: {matrix.get('tn', 0)}\n"
            f"- False Negatives: {matrix.get('fn', 0)} (Type II errors)"
        )

if __name__ == "__main__":
    helper = InterpretabilityHelper()
    
    # Test performance explanations
    print("=== Performance Explanations ===")
    print(helper.explain_performance(0.85))
    print(helper.explain_performance(0.72))
    
    # Test risk assessment
    print("\n=== Risk Assessments ===")
    print(helper.explain_risk({'mean': 0.8, 'std_dev': 0.15}))
    print(helper.explain_risk({'mean': 0.35, 'std_dev': 0.2}))
    
    # Test certification summaries
    print("\n=== Certification Summaries ===")
    print(helper.summarize_certification({'status': 'PASSED', 'level': 'A'}))
    print(helper.summarize_certification({'status': 'FAILED', 'level': 'B'}))
    print(helper.summarize_certification({'status': 'PENDING', 'level': 'C'}))
    
    # Test feature importance
    print("\n=== Feature Importance ===")
    features = {
        'data_quality': 0.92,
        'model_complexity': 0.85,
        'training_samples': 0.78,
        'feature_engineering': 0.65
    }
    print(helper.explain_feature_importance(features))
    
    # Test compliance report
    print("\n=== Compliance Report ===")
    print(helper.generate_compliance_report({
        'safety': 'Compliant',
        'security': 'Partial',
        'overall': 'Conditional Approval'
    }))
    
    # Test confusion matrix
    print("\n=== Error Analysis ===")
    print(helper.explain_confusion_matrix({
        'tp': 120, 'fp': 15,
        'tn': 200, 'fn': 25
    }))
