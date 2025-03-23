# SLAI Monitoring System

This module monitors system performance, model drift, health status, and sends alerts.

## Components

- **metrics_collector.py**: Collects CPU, memory, and disk metrics.
- **log_handler.py**: Central logging using loguru.
- **drift_detection.py**: Detects data drift with KS test.
- **health_check.py**: Checks if services are alive.
- **alert_manager.py**: Sends email alerts when thresholds are exceeded.

## Setup
1. Install dependencies:
   ```console
   pip install psutil loguru scipy
   ```

2. Configure alert emails in `alert_manager.py`.
3. Run each component as needed or integrate into the main SLAI loop.
