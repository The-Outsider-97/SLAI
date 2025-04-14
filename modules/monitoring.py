import logging
import json
import os
import time
import math
import platform
from collections import deque, defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger('SafeAI.Monitoring')

class Monitoring:
    """Enhanced monitoring system with statistical process control"""
    
    def __init__(self, shared_memory=None, alert_config=None, 
                 output_path="logs/metrics_log.jsonl", max_history=1000,
                 window_size=30, slo_config=None):
        """
        Args:
            window_size: Rolling window size for statistical calculations
            slo_config: Service Level Objectives {'metric': {'target': 0.99, 'period': 'day'}}
        """
        self.shared_memory = shared_memory
        self.alert_config = alert_config or {}
        self.slo_config = slo_config or {}
        self.output_path = output_path
        self.window_size = window_size
        
        # Initialize data structures
        self.history = deque(maxlen=max_history)
        self._rolling_windows = defaultdict(lambda: deque(maxlen=window_size))
        self._slo_budgets = {}
        self._system_baseline = {}

        # Initialize system monitoring
        self._init_resource_baseline()
        
        # Persistence setup
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._load_history()

    def _init_resource_baseline(self):
        """Establish baseline for system resource metrics"""
        self._system_baseline = {
            'cpu': self._get_cpu_usage(),
            'memory': self._get_memory_usage(),
            'timestamp': time.time()
        }

    def _get_cpu_usage(self):
        """Cross-platform CPU usage approximation"""
        if platform.system() == 'Windows':
            return float(os.popen('wmic cpu get loadpercentage').read().split()[2])
        else:
            return float(os.popen("top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\([0-9.]*\)%* id.*/\1/'").read().strip())

    def _get_memory_usage(self):
        """Memory usage percentage without psutil"""
        if platform.system() == 'Windows':
            total = int(os.popen('wmic ComputerSystem get TotalPhysicalMemory').read().split()[2])
            free = int(os.popen('wmic OS get FreePhysicalMemory').read().split()[2])
            return (total - free) / total * 100
        else:
            with open('/proc/meminfo') as f:
                mem = f.readlines()
                total = int(mem[0].split()[1])
                free = int(mem[1].split()[1])
                return (total - free) / total * 100

    def _load_history(self):
        """Load existing metrics from log file"""
        if os.path.isfile(self.output_path):
            with open(self.output_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        self.history.append(entry)
                        self._update_rolling_stats(entry)
                    except json.JSONDecodeError:
                        continue

    def record(self, module_name, metrics: dict, tags=None):
        """Enhanced recording with statistical tracking"""
        timestamp = datetime.utcnow().isoformat()
        entry = {
            "timestamp": timestamp,
            "module": module_name,
            "metrics": metrics,
            "tags": tags or {},
            "system": {
                "cpu": self._get_cpu_usage(),
                "memory": self._get_memory_usage()
            }
        }

        # Update data structures
        self.history.append(entry)
        self._update_rolling_stats(entry)
        self._write_to_file(entry)
        self._check_anomalies(entry)
        self._check_slo_compliance(entry)
        
        if self.shared_memory:
            self.shared_memory.set(f"monitoring_{module_name}", entry)

        return entry

    def _update_rolling_stats(self, entry):
        """Maintain rolling windows for statistical analysis"""
        for metric, value in entry['metrics'].items():
            if isinstance(value, (int, float)):
                self._rolling_windows[metric].append(value)

    def _check_anomalies(self, entry):
        """Statistical anomaly detection using Z-score"""
        for metric, value in entry['metrics'].items():
            window = self._rolling_windows.get(metric, [])
            if len(window) < 10:  # Minimum data points
                continue
                
            mean = sum(window) / len(window)
            std_dev = math.sqrt(sum((x - mean)**2 for x in window) / len(window))
            
            if std_dev == 0:
                continue
                
            z_score = abs(value - mean) / std_dev
            if z_score > 3:  # 3σ rule
                alert_msg = f"[ANOMALY] {entry['module']}: {metric} Z-score {z_score:.2f}"
                logger.warning(alert_msg)
                self._trigger_alert('anomaly', alert_msg)

    def _check_slo_compliance(self, entry):
        """Track error budgets against SLO targets"""
        for metric, config in self.slo_config.items():
            if metric in entry['metrics']:
                value = entry['metrics'][metric]
                budget = self._slo_budgets.get(metric, 1.0)
                
                # Simple error budget calculation
                error = 1 - value/config['target']
                new_budget = budget - error
                
                if new_budget < 0:
                    alert_msg = f"[SLO] {metric} budget exhausted!"
                    logger.error(alert_msg)
                    self._trigger_alert('slo', alert_msg)
                
                self._slo_budgets[metric] = new_budget

    def _trigger_alert(self, alert_type, message):
        """Centralized alert handling"""
        logger.warning(message)
        if self.shared_memory:
            self.shared_memory.set(f"alerts_{alert_type}", message)

    def calculate_metrics(self):
        """Generate system health metrics"""
        return {
            'mtbf': self._calculate_mtbf(),
            'availability': self._calculate_availability(),
            'error_rate': self._calculate_error_rate()
        }

    def _calculate_mtbf(self):
        """Mean Time Between Failures calculation"""
        failures = [e for e in self.history if 'error' in e.get('tags', {})]
        if len(failures) < 2:
            return float('inf')
            
        intervals = []
        prev_time = None
        for entry in failures:
            current_time = datetime.fromisoformat(entry['timestamp'])
            if prev_time:
                intervals.append((current_time - prev_time).total_seconds())
            prev_time = current_time
            
        return sum(intervals)/len(intervals) if intervals else float('inf')

    def _calculate_availability(self):
        """Calculate system availability percentage"""
        uptime = sum(1 for e in self.history if e['metrics'].get('status') == 'up')
        return uptime / len(self.history) * 100 if self.history else 100

    def _calculate_error_rate(self):
        """Calculate errors per hour"""
        if not self.history:
            return 0
            
        timespan = datetime.fromisoformat(self.history[-1]['timestamp']) - \
                 datetime.fromisoformat(self.history[0]['timestamp'])
        errors = sum(1 for e in self.history if e['metrics'].get('error_count', 0) > 0)
        return errors / (timespan.total_seconds() / 3600)

    def get_rolling_stats(self, metric):
        """Return rolling window statistics"""
        window = self._rolling_windows.get(metric, [])
        if not window:
            return {}
            
        mean = sum(window)/len(window)
        variance = sum((x - mean)**2 for x in window)/len(window)
        return {
            'mean': mean,
            'std_dev': math.sqrt(variance),
            'min': min(window),
            'max': max(window),
            'trend': self._calculate_trend(window)
        }

    def _calculate_trend(self, window):
        """Simple linear trend calculation"""
        x = list(range(len(window)))
        y = window
        x_mean = sum(x)/len(x)
        y_mean = sum(y)/len(y)
        
        numerator = sum((xi - x_mean)*(yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean)**2 for xi in x)
        
        return numerator / denominator if denominator != 0 else 0

    def _write_to_file(self, entry):
        """Persist entry to log file"""
        try:
            with open(self.output_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write monitoring log: {e}")

    def print_summary(self, hours=24):
        """Enhanced summary with statistical overview"""
        print("\n=== Monitoring Summary ===")
        print(f"Time window: Last {hours} hours")
        
        # Calculate time boundary
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [e for e in self.history if datetime.fromisoformat(e['timestamp']) > cutoff]
        
        # System health metrics
        print(f"\nSystem Health:")
        print(f"  CPU Usage: {self._get_cpu_usage():.1f}%")
        print(f"  Memory Usage: {self._get_memory_usage():.1f}%")
        print(f"  MTBF: {self._calculate_mtbf():.1f}s")
        print(f"  Availability: {self._calculate_availability():.1f}%")
        
        # Metric trends
        print("\nKey Metrics:")
        for metric in self._rolling_windows:
            stats = self.get_rolling_stats(metric)
            print(f"  {metric}: {stats['mean']:.2f} ±{stats['std_dev']:.2f} (trend: {stats['trend']:+.2f}/hr)")

    def export_metrics(self, format='json'):
        """Export metrics in specified format"""
        if format == 'json':
            return list(self.history)
        elif format == 'prometheus':
            return self._format_prometheus()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _format_prometheus(self):
        """Convert metrics to Prometheus-like format"""
        output = []
        for entry in self.history:
            for metric, value in entry['metrics'].items():
                line = f"{metric}{{module='{entry['module']}'}} {value}"
                output.append(line)
        return '\n'.join(output)
