import torch
import base64
import re, sys
import hashlib
import yaml, json
import torch.nn as nn
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Union

from src.utils.interpretability import InterpretabilityHelper
from src.agents.safety.utils.config_loader import load_global_config, get_config_section
from src.agents.safety.secure_memory import SecureMemory
from logs.logger import get_logger

logger = get_logger("Security Attention Monitor")

class AttentionMonitor(torch.nn.Module):
    """Mechanistic interpretability tool (Bereska & Gavves, 2024)"""
    def __init__(self, device='cpu'):
        super().__init__()
        self.config = load_global_config()
        self.attention_config = get_config_section('attention_monitor')
        memory = SecureMemory()
        self.memory = memory
        self.interpreter = InterpretabilityHelper()
        self.device = device
        
        # Load configuration parameters
        self.entropy_threshold = self.config.get('entropy_threshold', 3.0)
        self.uniformity_threshold = self.config.get('uniformity_threshold', 0.2)
        self.anomaly_detection = self.config.get('anomaly_detection', True)
        
        logger.info("Attention Monitor initialized with entropy threshold: %.2f", 
                   self.entropy_threshold)

    def get_anomaly_interpretation(self, analysis: Dict) -> str:
        """Generate human-readable interpretation of attention anomalies"""
        findings = []
        
        if analysis.get("anomaly", False):
            if analysis["entropy"] < self.entropy_threshold * 0.7:
                findings.append("Low attention entropy indicates potential over-focusing on specific tokens")
            if analysis["uniformity"] > self.uniformity_threshold * 1.5:
                findings.append("High attention variance suggests inconsistent pattern distribution")
            if analysis["anomaly_score"] > 0.8:
                findings.append("Severe attention anomaly detected - possible adversarial manipulation")
        
        return "; ".join(findings) if findings else "Normal attention patterns"

    def analyze_attention(self, attention_matrix: torch.Tensor, context: Dict = None) -> Dict:
        """Comprehensive attention pattern analysis with security insights"""
        # Basic metrics
        metrics = {
            "max_attention": attention_matrix.max().item(),
            "min_attention": attention_matrix.min().item(),
            "mean_attention": attention_matrix.mean().item(),
            "entropy": self._calculate_entropy(attention_matrix),
            "uniformity": self._calculate_uniformity(attention_matrix),
            "head_importance": self._calculate_head_importance(attention_matrix),
            "anomaly_score": 0.0
        }
        
        # Advanced analysis
        metrics["dispersion"] = self._calculate_dispersion(attention_matrix)
        metrics["focus_pattern"] = self._identify_focus_pattern(attention_matrix)
        
        # Anomaly detection
        if self.anomaly_detection:
            metrics["anomaly_score"] = self._detect_anomalies(attention_matrix)
            metrics["anomaly"] = metrics["anomaly_score"] > self.config.get('anomaly_threshold', 0.7)
        
        # Security assessment
        metrics["security_assessment"] = self._assess_security(metrics)
        
        # Store analysis if configured
        if self.config.get('store_analysis', True):
            self._store_analysis(metrics, context)
        
        return metrics

    def generate_report(self, analysis: Dict) -> str:
        """Generate comprehensive attention analysis report"""
        report = [
            "# Attention Analysis Report",
            f"**Generated**: {datetime.now().isoformat()}",
            "## Core Metrics",
            f"- **Entropy**: {analysis.get('entropy', 0):.3f} ",
            f"- **Uniformity**: {analysis.get('uniformity', 0):.3f} ",
            f"- **Anomaly Score**: {analysis.get('anomaly_score', 0):.3f} ",
            "## Security Assessment"
        ]
        
        # Add security assessment
        security = analysis.get('security_assessment', {})
        report.append(f"- **Status**: {'✅ Secure' if security.get('secure') else '⚠️ Needs Review'}")
        report.append(f"- **Confidence**: {security.get('confidence', 0):.1%}")
        report.append(f"- **Key Findings**: {security.get('findings', 'No significant issues detected')}")
        
        # Add interpretability explanations
        report.append("\n## Interpretability Insights")
        report.append(self.interpreter.explain_risk({
            'mean': analysis.get('anomaly_score', 0),
            'std_dev': analysis.get('uniformity', 0)
        }))
        
        # Visualization
        if 'attention_plot' in analysis:
            report.append("\n## Attention Visualization")
            report.append(f"![Attention Map](data:image/png;base64,{analysis['attention_plot']})")
        
        return "\n".join(report)

    def _calculate_entropy(self, matrix: torch.Tensor) -> float:
        """Information-theoretic attention analysis with normalization"""
        flat = matrix.flatten()
        flat = flat / flat.sum()  # Normalize to probability distribution
        # Filter out zero probabilities to avoid log(0)
        mask = flat > 0
        probs = flat[mask]
        return (-(probs * torch.log2(probs)).sum().item())

    def _calculate_uniformity(self, matrix: torch.Tensor) -> float:
        """Measure of attention dispersion with scaling"""
        std = matrix.std()
        mean = matrix.mean()
        return (std / mean).item() if mean > 0 else 0

    def _calculate_head_importance(self, matrix: torch.Tensor) -> List[float]:
        """Calculate importance of each attention head"""
        if matrix.dim() < 3:
            return [1.0]  # Single head
        
        # Calculate variance contribution of each head
        head_importances = []
        for head in range(matrix.shape[0]):
            head_matrix = matrix[head]
            std = head_matrix.std()
            mean = head_matrix.mean()
            head_importances.append((std / mean).item() if mean > 0 else 0.0)
        
        # Normalize to sum to 1
        total = sum(head_importances)
        return [imp/total for imp in head_importances] if total > 0 else head_importances

    def _calculate_dispersion(self, matrix: torch.Tensor) -> float:
        """Calculate attention dispersion index (0-1 scale)"""
        max_val = matrix.max()
        min_val = matrix.min()
        return ((max_val - min_val) / (max_val + min_val)).item() if max_val + min_val > 0 else 0

    def _identify_focus_pattern(self, matrix: torch.Tensor) -> str:
        """Identify attention focus pattern"""
        row_max = matrix.max(dim=1).values.mean().item()
        col_max = matrix.max(dim=0).values.mean().item()
        diag = torch.diag(matrix).mean().item()
        
        if diag > row_max and diag > col_max:
            return "diagonal"
        elif row_max > col_max:
            return "row-focused"
        return "column-focused"

    def _detect_anomalies(self, matrix: torch.Tensor) -> float:
        """Detect anomalous attention patterns using statistical methods"""
        # Entropy-based anomaly detection
        entropy = self._calculate_entropy(matrix)
        entropy_dev = abs(entropy - self.entropy_threshold) / self.entropy_threshold
        
        # Uniformity-based anomaly detection
        uniformity = self._calculate_uniformity(matrix)
        uniformity_dev = abs(uniformity - self.uniformity_threshold) / self.uniformity_threshold
        
        # Combined anomaly score
        return (entropy_dev + uniformity_dev) / 2

    def _assess_security(self, metrics: Dict) -> Dict:
        """Assess security implications of attention patterns"""
        findings = []
        secure = True
        confidence = 1.0
        
        # Check entropy security implications
        if metrics["entropy"] < self.entropy_threshold * 0.7:
            findings.append("Low attention entropy detected - potential overfocusing")
            confidence *= 0.8
            secure = False
            
        # Check uniformity security implications
        if metrics["uniformity"] > self.uniformity_threshold * 1.5:
            findings.append("High attention variance detected - potential erratic behavior")
            confidence *= 0.7
            secure = False
            
        # Check anomaly score
        if metrics["anomaly_score"] > 0.8:
            findings.append("High anomaly score detected - potential adversarial manipulation")
            confidence *= 0.5
            secure = False
            
        return {
            "secure": secure,
            "confidence": confidence,
            "findings": self.get_anomaly_interpretation(metrics)  # Updated to use new method
        }

    def visualize_attention(self, matrix: torch.Tensor) -> str:
        """Generate attention visualization and return as base64"""
        # Convert to numpy for visualization (only place where conversion happens)
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix.cpu().numpy(), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title("Attention Heatmap")
        plt.xlabel("Key Positions")
        plt.ylabel("Query Positions")
        
        # Save to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def _store_analysis(self, metrics: Dict, context: Dict = None):
        """Store attention analysis in secure memory"""
        record = {
            "metrics": metrics,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        self.memory.add(
            record,
            tags=["attention_analysis", "security"],
            sensitivity=0.6
        )

class AttentionAdapter:
    def __init__(self, monitor):
        self.monitor = monitor
        
    def log_attention(self, attention_matrix: torch.Tensor):
        # Reduce to 2D: average over heads and take first batch
        if attention_matrix.dim() == 4:
            # Average over heads: [batch, heads, seq, seq] -> [batch, seq, seq]
            attention_matrix = attention_matrix.mean(dim=1)
            # Take first batch: [batch, seq, seq] -> [seq, seq]
            attention_matrix = attention_matrix[0]
        
        # Ensure the tensor is on the correct device
        attention_matrix = attention_matrix.to(self.monitor.device)
        self.monitor.analyze_attention(attention_matrix)

# Usage Example
if __name__ == "__main__":
    print("\n=== Running Security Attention Monitor ===\n")
    device='cpu'
    
    # Create sample attention matrix using torch
    attention_matrix = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8]
    ], dtype=torch.float32)
    
    # Create minimal transformer config for the test
    transformer_config = {
        'transformer': {
            'embed_dim': 512,
            'num_heads': 8
        },
        'attention': {
            'dropout_rate': 0.1,
            'initializer': 'xavier_uniform'
        }
    }
    
    # Create the attention layer with proper config
    from src.agents.perception.modules.attention import EfficientAttention
    attention_layer = EfficientAttention(transformer_config, device)
    
    monitor = AttentionMonitor(device)
    adapter = AttentionAdapter(monitor)
    attention_layer.add_observer(adapter)
    
    # Create a dummy input tensor
    input_tensor = torch.randn(1, 3, 512)  # (batch_size, seq_len, embed_dim)
    
    output = attention_layer(input_tensor) 
    analysis = monitor.analyze_attention(attention_matrix)
    report = monitor.generate_report(analysis)
    print(report)
    print("\n=== Attention Analysis Complete ===\n")
