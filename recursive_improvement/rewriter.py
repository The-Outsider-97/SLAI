import importlib
import logging
import os, sys
import shutil
import torch
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from collaborative/shared_memory import SharedMemory  # <-- SLAI shared memory module

class Rewriter:
    def __init__(self, agent_path, config_path=None, trigger_mode="reward", reward_threshold=0.1, time_interval=300):
        self.agent_path = Path(agent_path)
        self.backup_path = self.agent_path.with_suffix(".bak")
        self.config_path = Path(config_path) if config_path else None
        self.trigger_mode = trigger_mode  # 'reward' or 'time'
        self.reward_threshold = reward_threshold
        self.time_interval = time_interval  # seconds
        self.last_reward = None
        self.last_edit_time = time.time()
        self.shared_memory = SharedMemory()

    def monitor_performance(self, current_reward):
        if self.trigger_mode == "reward":
            if self.last_reward is None:
                self.last_reward = current_reward
                return False
            improvement = current_reward - self.last_reward
            self.last_reward = current_reward
            return improvement < self.reward_threshold

        elif self.trigger_mode == "time":
            return time.time() - self.last_edit_time > self.time_interval

        return False

    def edit_config(self):
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                lines = f.readlines()

            with open(self.config_path, 'w') as f:
                for line in lines:
                    if "learning_rate" in line:
                        new_lr = round(random.uniform(1e-4, 1e-2), 5)
                        f.write(f"learning_rate: {new_lr}\n")
                    else:
                        f.write(line)

            print(f"[Rewriter] Config updated: learning_rate modified.")

    def edit_model_layer(self):
        if not self.backup_path.exists():
            shutil.copy(self.agent_path, self.backup_path)
            print(f"[Rewriter] Backup created at {self.backup_path}")

        model_code = self.agent_path.read_text()
        if "nn.Linear" in model_code:
            model_code = model_code.replace("nn.Linear", "nn.Sequential(nn.Linear")
            model_code = model_code.replace(")", ", nn.ReLU())", 1)
            self.agent_path.write_text(model_code)
            print("[Rewriter] Model layer structure rewritten.")

    def rollback_model(self):
        if self.backup_path.exists():
            shutil.copy(self.backup_path, self.agent_path)
            print(f"[Rewriter] Rollback complete: restored {self.agent_path.name} from backup.")
            self.shared_memory.set("last_rollback", time.time())
        else:
            print("[Rewriter] No backup available for rollback.")

    def reload_agent(self, module_name, class_name):
        try:
            if module_name in globals():
                importlib.reload(globals()[module_name])
            else:
                globals()[module_name] = importlib.import_module(module_name)
            cls = getattr(globals()[module_name], class_name)
            print(f"[Rewriter] {class_name} reloaded successfully.")
            return cls
        except Exception as e:
            print(f"[Rewriter] Failed to reload agent: {e}")
            self.rollback_model()
            return None

    def trigger_recursive_improvement(self, reward=None):
        if self.monitor_performance(reward):
            # Check SharedMemory before proceeding
            last_rewrite = self.shared_memory.get("last_rewrite") or 0
            if time.time() - last_rewrite < 30:
                print("[Rewriter] Skipping: recent rewrite already logged.")
                return False

            print("[Rewriter] Triggering recursive improvement...")
            if self.config_path:
                self.edit_config()
            self.edit_model_layer()
            self.last_edit_time = time.time()
            self.shared_memory.set("last_rewrite", time.time())
            self.shared_memory.set("rewrite_log", {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "trigger": self.trigger_mode,
                "agent_path": str(self.agent_path.name),
            })
            return True
        return False
