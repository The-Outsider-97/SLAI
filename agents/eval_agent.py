import time

class EvaluationAgent:
    def __init__(self):
        self.logs = []

    def log_interaction(self, agent_name: str, input_data: str, output_data: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append({
            "time": timestamp,
            "agent": agent_name,
            "input": input_data,
            "output": output_data
        })

    def summarize_logs(self):
        return self.logs[-10:]  # Last 10 entries

    def evaluate_performance(self, metric_fn):
        return metric_fn(self.logs)
