import psutil
import time
import logging

def collect_system_metrics():
    metrics = {
        'cpu_usage': psutil.cpu_percent(interval=1),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent
    }
    return metrics

if __name__ == "__main__":
    while True:
        print(collect_system_metrics())
        time.sleep(5)
