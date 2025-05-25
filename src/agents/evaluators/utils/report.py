import os
import math
import hashlib
import yaml, json

from datetime import datetime
from collections import deque
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter, QPixmap, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QRect, QPointF, QSize, QBuffer

from src.agents.evaluators.utils.certification_framework import CertificationManager
from src.agents.evaluators.utils.documentation import AuditTrail, DocumentVersioner, AuditBlock
from logs.logger import get_logger

logger = get_logger("Performance Visualizer")

CONFIG_PATH = "src/agents/evaluators/configs/evaluator_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

_global_visualizer = None

def get_visualizer(config=None):
    global _global_visualizer
    if not _global_visualizer:
        config = config or load_config()
        _global_visualizer = PerformanceVisualizer(config)
    return _global_visualizer

class PerformanceVisualizer:
    def __init__(self, config):
        config = load_config() or {}
        self.config = config.get('performance_visualizer', {})
        self.max_points = self.config.get('max_points', 100)
        
        # Initialize metrics with configurable initial values
        initial_metrics = self.config.get('initial_metrics', {})
        self.metrics = {
            'rewards': deque(maxlen=self.max_points),
            'risks': deque(maxlen=self.max_points),
            'successes': initial_metrics.get('successes', 0),
            'failures': initial_metrics.get('failures', 0),
            'hazard_rates': deque(initial_metrics.get('hazard_rates', []), maxlen=self.max_points),
            'operational_times': deque(initial_metrics.get('operational_times', []), maxlen=self.max_points)
        }
    
        # Load color scheme from config with fallback defaults
        default_colors = {
            'background': QColor(30, 30, 30),
            'text': QColor(255, 255, 255),
            'reward_line': QColor(0, 191, 99),
            'risk_line': QColor(255, 144, 0),
            'success': QColor(0, 191, 99),
            'failure': QColor(231, 76, 60)
        }
        config_colors = self.config.get('colors', {})
        self.colors = {
            key: QColor(*config_colors.get(key, default_color.getRgb()[:3]))
            for key, default_color in default_colors.items()
        }
    
        # Load visualization parameters
        line_styles = self.config.get('line_styles', {})
        self.line_thickness = line_styles.get('thickness', 2)
        self.grid_style = Qt.DotLine if line_styles.get('grid_style', 'dot') == 'dot' else Qt.SolidLine
    
        # Store chart dimensions from config
        self.chart_dimensions = self.config.get('chart_dimensions', {})

        # Add certification and documentation components
        self.cert_manager = CertificationManager(config)
        self.audit_trail = AuditTrail(config)
        self.doc_versioner = DocumentVersioner(config)

        logger.info(f"Performance Visualizer succesfully initialized")

    def update_metrics(self, evaluation_data):
        """Update metrics from EvaluationAgent log data"""
        if not isinstance(evaluation_data, dict):
            raise TypeError(f"Expected dict for evaluation_data, got {type(evaluation_data)}")
        self.metrics['successes'] += evaluation_data.get('successes', 0)
        self.metrics['failures'] += evaluation_data.get('failures', 0)
        
        if 'hazards' in evaluation_data:
            self.metrics['hazard_rates'].append(
                evaluation_data['hazards'].get('system_failure', 0)
            )
        
        if 'operational_time' in evaluation_data:
            self.metrics['operational_times'].append(
                evaluation_data['operational_time']
            )

    def add_metrics(self, metric_type: str, values: dict):
        """Generic metric update method"""
        for key, value in values.items():
            if key in self.metrics:
                if isinstance(self.metrics[key], deque):
                    self.metrics[key].append(value)
                else:
                    self.metrics[key] += value
            else:
                logger.warning(f"Ignoring unknown metric: {key}")

    def render_tradeoff_chart(self, size=None):
        """Render risk-reward tradeoff as QPixmap"""
        if not size:
            # Get dimensions from config
            tradeoff_dims = self.config.get('chart_dimensions', {}).get('tradeoff', [800, 600])
            size = QSize(*tradeoff_dims)
        pixmap = QPixmap(size)
        pixmap.fill(self.colors['background'])
        painter = QPainter(pixmap)
        
        # Calculate bounds
        max_risk = max(self.metrics['hazard_rates']) if self.metrics['hazard_rates'] else 1
        max_reward = max(self.metrics['operational_times']) if self.metrics['operational_times'] else 1
        
        # Draw grid
        self._draw_grid(painter, size, x_max=max_risk, y_max=max_reward)
        
        # Plot points
        pen = QPen(self.colors['reward_line'], 2)
        painter.setPen(pen)
        
        for i in range(1, len(self.metrics['hazard_rates'])):
            x1 = (self.metrics['hazard_rates'][i-1]/max_risk) * size.width()
            y1 = size.height() - (self.metrics['operational_times'][i-1]/max_reward) * size.height()
            x2 = (self.metrics['hazard_rates'][i]/max_risk) * size.width()
            y2 = size.height() - (self.metrics['operational_times'][i]/max_reward) * size.height()
            
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        
        # Add labels
        self._draw_labels(painter, size, "Risk Estimate", "Operational Time")
        painter.end()
        return pixmap

    def render_temporal_chart(self, size, metric):
        """Generic time-series chart renderer"""
        valid_metrics = ['hazard_rates', 'operational_times']
        if metric not in valid_metrics:
            return QPixmap()
            
        pixmap = QPixmap(size)
        pixmap.fill(self.colors['background'])
        painter = QPainter(pixmap)
        
        # Calculate bounds
        data = list(self.metrics[metric])
        max_val = max(data) if data else 1
        x_step = size.width() / self.max_points
        
        # Draw line
        color = self.colors['reward_line'] if metric == 'operational_times' else self.colors['risk_line']
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        for i in range(1, len(data)):
            x1 = (i-1) * x_step
            y1 = size.height() - (data[i-1]/max_val) * size.height()
            x2 = i * x_step
            y2 = size.height() - (data[i]/max_val) * size.height()
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        
        # Add labels
        label = "Operational Time" if metric == 'operational_times' else "Hazard Rate"
        self._draw_labels(painter, size, "Time Steps", label)
        painter.end()
        return pixmap

    def _draw_grid(self, painter, size, x_max=1, y_max=1):
        """Draw chart grid with labels"""
        pen = QPen(self.colors['text'], 1, Qt.DotLine)
        painter.setPen(pen)
        
        # Vertical lines - cast to int
        for x in range(0, 11):
            x_pos = int((x/10) * size.width())  # Convert to integer
            painter.drawLine(x_pos, 0, x_pos, size.height())
            painter.drawText(x_pos-20, size.height()-10, f"{x_max*(x/10):.1f}")
    
        # Horizontal lines - cast to int
        for y in range(0, 11):
            y_pos = int(size.height() - (y/10) * size.height())  # Convert to integer
            painter.drawLine(0, y_pos, size.width(), y_pos)
            painter.drawText(5, y_pos+15, f"{y_max*(y/10):.1f}")

    def _draw_labels(self, painter, size, x_label, y_label):
        """Add axis labels to charts"""
        painter.setPen(QPen(self.colors['text']))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        
        # X-axis label
        painter.drawText(
            QRect(0, size.height()-30, size.width(), 20),
            Qt.AlignCenter, x_label
        )
        
        # Y-axis label
        painter.save()
        painter.translate(20, size.height()/2)
        painter.rotate(-90)
        painter.drawText(QRect(0, 0, size.height(), 20), Qt.AlignCenter, y_label)
        painter.restore()

    def get_current_metrics(self):
        """Return current metrics for status display"""
        return {
            'success_rate': self.metrics['successes']/(self.metrics['successes']+self.metrics['failures']) 
                if (self.metrics['successes']+self.metrics['failures']) > 0 else 0,
            'current_risk': self.metrics['hazard_rates'][-1] if self.metrics['hazard_rates'] else 0,
            'operational_time': self.metrics['operational_times'][-1] if self.metrics['operational_times'] else 0
        }

    def generate_full_report(self):
        """Generate comprehensive report with visuals, certs, and docs"""
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "report_version": "1.2.0"
            },
            "performance_metrics": self.get_current_metrics(),
            "certification": self._generate_cert_section(),
            "documentation": self._generate_docs_section(),
            "visualizations": {
                "tradeoff_chart": self._chart_to_base64(self.render_tradeoff_chart()),
                "temporal_chart": self._chart_to_base64(
                    self.render_temporal_chart(QSize(600, 400), 'hazard_rates')
                )
            }
        }
        
        # Audit and version the report
        self._audit_report(report)
        return report

    def _generate_cert_section(self):
        """Run certification checks and format results"""
        cert_status = self.cert_manager.evaluate_certification()
        return {
            "current_level": self.cert_manager.current_level.name,
            "requirements_passed": cert_status[0],
            "unmet_requirements": cert_status[1],
            "certificate": self.cert_manager.generate_certificate()
        }

    def _generate_docs_section(self):
        """Generate documentation artifacts"""
        return {
            "audit_chain": json.loads(self.audit_trail.export_chain()),
            "latest_doc_version": self.doc_versioner.get_latest()
        }

    def _chart_to_base64(self, pixmap):
        """Convert QPixmap to base64 for JSON serialization"""
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        pixmap.save(buffer, "PNG")
        return bytes(buffer.data().toBase64()).decode('utf-8')

    def _audit_report(self, report):
        """Create audit record and version the report"""
        audit_data = {
            "report_hash": hashlib.sha256(json.dumps(report).encode()).hexdigest(),
            "metrics_snapshot": report['performance_metrics']
        }
        
        # Create new audit block
        prev_hash = self.audit_trail.chain[-1].hash if self.audit_trail.chain else "0"*64
        new_block = AuditBlock(
            config=load_config(),
            data=audit_data,
            previous_hash=prev_hash
        )
        new_block.mine_block(difficulty=4)
        self.audit_trail.chain.append(new_block)
        
        # Store document version
        self.doc_versioner.add_version(report)

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Adaptive Risk ===\n")
    import sys

    config = load_config()

    visual = PerformanceVisualizer(config)
    logger.info(visual)
    print(f"\n* * * * * Phase 2 * * * * *\n")
    evaluation_data = {
        "successes": 5,
        "failures": 2,
        "hazards": {"system_failure": 0.12},
        "operational_time": 150.0
    }

    update = visual.update_metrics(evaluation_data)
    print(visual.get_current_metrics())
    print(f"\n* * * * * Phase 3 * * * * *\n")
    app = QApplication(sys.argv)
    chart_size = QSize(*visual.config['chart_dimensions']['tradeoff'])
    chart = visual.render_tradeoff_chart(chart_size)

    test_pixmap = QPixmap(chart_size)
    test_painter = QPainter(test_pixmap)
    visual._draw_grid(test_painter, chart_size, x_max=1.0, y_max=1.0)
    test_painter.end()
    print(f"\n* * * * * Phase 4 * * * * *\n")

    # Generate comprehensive report
    full_report = visual.generate_full_report()
    
    # ===== ADD EVIDENCE SUBMISSION HERE =====
    # Submit evidence to certification manager
    visual.cert_manager.submit_evidence({
        "timestamp": datetime.now().isoformat(),
        "type": ["performance_report"],
        "content": full_report
    })
    
    # ===== ADD COMPLIANCE EXPORT HERE =====
    # Export compliance package
    compliance_package = {
        "report": full_report,
        "certificate": visual.cert_manager.generate_certificate(),
        "audit_proof": visual.audit_trail.export_chain(format='yaml')
    }
    
    # Output results
    print("\n=== Full System Report ===")
    print(json.dumps(full_report, indent=2))
    
    print("\n=== Audit Chain ===")
    print(visual.audit_trail.export_chain())
    
    print("\n=== Certification Status ===")
    print(f"Current Level: {full_report['certification']['current_level']}")
    print(f"Passed: {full_report['certification']['requirements_passed']}")

    # ===== OPTIONAL: Print compliance package =====
    print("\n=== Compliance Package ===")
    print(yaml.safe_dump(compliance_package, default_flow_style=False))

    print("\n=== Successfully Ran Adaptive Risk ===\n")
    # sys.exit(app.exec_())
