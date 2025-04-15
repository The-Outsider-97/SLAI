import os
import math
from datetime import datetime
from collections import deque
from PyQt5.QtGui import QPainter, QPixmap, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QRect, QPointF

class PerformanceVisualizer:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.metrics = {
            'rewards': deque(maxlen=max_points),
            'risks': deque(maxlen=max_points),
            'successes': 0,
            'failures': 0,
            'hazard_rates': deque(maxlen=max_points),
            'operational_times': deque(maxlen=max_points)
        }
        
        # Color scheme matching main_window
        self.colors = {
            'background': QColor(30, 30, 30),
            'text': QColor(255, 255, 255),
            'reward_line': QColor(0, 191, 99),
            'risk_line': QColor(255, 144, 0),
            'success': QColor(0, 191, 99),
            'failure': QColor(231, 76, 60)
        }

    def update_metrics(self, evaluation_data):
        """Update metrics from EvaluationAgent log data"""
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

    def render_tradeoff_chart(self, size):
        """Render risk-reward tradeoff as QPixmap"""
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
        
        # Vertical lines
        for x in range(0, 11):
            x_pos = (x/10) * size.width()
            painter.drawLine(x_pos, 0, x_pos, size.height())
            painter.drawText(x_pos-20, size.height()-10, f"{x_max*(x/10):.1f}")

        # Horizontal lines
        for y in range(0, 11):
            y_pos = size.height() - (y/10) * size.height()
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
