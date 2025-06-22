
import numpy as np

from typing import Dict, List, Tuple, Union

class InterpretabilityHelper:
    """Helper to explain evaluation results in plain language with enhanced interpretability features"""
    
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
    def generate_compliance_report(metrics: dict) -> str:
        report = "Compliance Summary:\n"
        report += f"- Safety: {metrics.get('safety', 'Not assessed')}\n"
        report += f"- Security: {metrics.get('security', 'Not assessed')}\n"
        report += f"- Ethics: {metrics.get('ethics', 'Not assessed')}\n"
        report += f"Overall Status: {metrics.get('overall', 'Pending')}"
        return report

    @staticmethod
    def explain_feature_importance(features: dict, top_n: int = 3) -> str:
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:top_n]
        explanation = "Most influential factors:\n"
        for i, (feature, importance) in enumerate(sorted_features, 1):
            explanation += f"{i}. {feature}: {importance:.2f} impact\n"
        return explanation

    @staticmethod
    def explain_confusion_matrix(matrix: dict) -> str:
        return (
            f"Error Analysis:\n"
            f"- True Positives: {matrix.get('tp', 0)}\n"
            f"- False Positives: {matrix.get('fp', 0)} (Type I errors)\n"
            f"- True Negatives: {matrix.get('tn', 0)}\n"
            f"- False Negatives: {matrix.get('fn', 0)} (Type II errors)"
        )

    @staticmethod
    def explain_validation_metrics(metrics: Dict) -> str:
        return (
            f"Validation Summary:\n"
            f"- Passed: {metrics['passed']}\n"
            f"- Failed: {metrics['failed']}\n"
            f"- Requirement Coverage: {metrics['coverage']}"
        )

    @staticmethod
    def explain_attention_pattern(attention: np.ndarray, tokens: List[str]) -> str:
        """Explain attention patterns in natural language"""
        max_idx = np.unravel_index(np.argmax(attention), attention.shape)
        min_idx = np.unravel_index(np.argmin(attention), attention.shape)
        
        explanation = (
            f"The model focuses most on '{tokens[max_idx[1]]}' when processing '{tokens[max_idx[0]]}'. "
            f"The weakest attention is between '{tokens[min_idx[0]]}' and '{tokens[min_idx[1]]}'."
        )
        
        # Check for diagonal dominance
        diag = np.diag(attention)
        if np.mean(diag) > 0.7:
            explanation += " Strong diagonal pattern suggests token-to-self attention dominance."
            
        return explanation

    @staticmethod
    def explain_anomaly(anomaly_score: float, thresholds: Dict[str, float]) -> str:
        """Explain attention anomaly scores"""
        if anomaly_score > thresholds.get('critical', 0.9):
            return f"CRITICAL anomaly ({anomaly_score:.3f}): Possible adversarial manipulation"
        elif anomaly_score > thresholds.get('high', 0.7):
            return f"High anomaly ({anomaly_score:.3f}): Potential attention hijacking"
        elif anomaly_score > thresholds.get('medium', 0.5):
            return f"Moderate anomaly ({anomaly_score:.3f}): Unusual attention patterns"
        else:
            return "Normal attention patterns detected"

    @staticmethod
    def explain_entropy(entropy: float, normal_range: Tuple[float, float]) -> str:
        """Explain attention entropy values"""
        low, high = normal_range
        if entropy < low:
            return (f"Low attention entropy ({entropy:.3f}): Model is focusing too narrowly, "
                    "potentially ignoring contextual information")
        elif entropy > high:
            return (f"High attention entropy ({entropy:.3f}): Model is attending too broadly, "
                    "potentially lacking focus")
        else:
            return f"Normal attention entropy ({entropy:.3f}): Balanced focus distribution"

    @staticmethod
    def explain_security_assessment(assessment: Dict) -> str:
        """Generate human-readable security assessment"""
        status = "SECURE" if assessment['secure'] else "VULNERABLE"
        confidence = assessment.get('confidence', 0.0)
        
        report = [
            f"Security Status: {status}",
            f"Confidence: {confidence:.1%}",
            "Findings:"
        ]
        
        for finding in assessment.get('findings', []):
            report.append(f"- {finding}")
            
        if assessment.get('recommendations'):
            report.append("Recommendations:")
            for rec in assessment['recommendations']:
                report.append(f"- {rec}")
                
        return "\n".join(report)

    @staticmethod
    def explain_head_importance(importances: List[float]) -> str:
        """Explain attention head importance distribution"""
        if not importances:
            return "No head importance data available"
            
        max_idx = np.argmax(importances)
        min_idx = np.argmin(importances)
        variance = np.var(importances)
        
        explanation = (
            f"Head #{max_idx} has the strongest influence ({importances[max_idx]:.3f}), "
            f"while head #{min_idx} has the weakest ({importances[min_idx]:.3f}). "
        )
        
        if variance > 0.1:
            explanation += "Significant variance in head importance suggests specialized roles."
        else:
            explanation += "Consistent head importance suggests redundant processing."
            
        return explanation

    @staticmethod
    def attention_to_text(attention: np.ndarray, tokens: List[str]) -> str:
        """Convert attention matrix to human-readable text representation"""
        output = []
        for i, row in enumerate(attention):
            focus_idx = np.argmax(row)
            output.append(
                f"When processing '{tokens[i]}', "
                f"the model focuses most on '{tokens[focus_idx]}' "
                f"(attention: {row[focus_idx]:.2f})"
            )
        return "\n".join(output)

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
    
    # Test new methods
    print("\n=== Enhanced Interpretability ===")
    attention_matrix = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.3, 0.6]
    ])
    tokens = ["The", "movie", "was"]
    
    print("\nAttention Pattern Explanation:")
    print(helper.explain_attention_pattern(attention_matrix, tokens))
    
    print("\nAnomaly Explanation:")
    print(helper.explain_anomaly(0.85, {'critical': 0.9, 'high': 0.7, 'medium': 0.5}))
    
    print("\nEntropy Explanation:")
    print(helper.explain_entropy(0.25, (0.4, 0.8)))
    
    print("\nSecurity Assessment:")
    print(helper.explain_security_assessment({
        'secure': False,
        'confidence': 0.65,
        'findings': [
            "Low attention entropy detected",
            "High variance in head importance"
        ],
        'recommendations': [
            "Investigate attention patterns in layer 4",
            "Apply attention regularization"
        ]
    }))
    
    print("\nHead Importance Explanation:")
    print(helper.explain_head_importance([0.4, 0.1, 0.3, 0.2]))
    
    print("\nAttention to Text:")
    print(helper.attention_to_text(attention_matrix, tokens))
