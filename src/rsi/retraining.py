import time
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Retrainer")

def run_retraining_loop(registered_agents: List, shared_memory, interval: int = 300):
    """
    Periodically checks agents for retraining flags and performs retraining.

    Args:
        registered_agents (List): List of agent instances (must implement .name and .retrain()).
        shared_memory: Shared memory interface.
        interval (int): Seconds between retraining scans (default 5 minutes).
    """
    logger.info("🔁 Retrainer loop started. Interval: %ds", interval)

    while True:
        logger.info("[Retrainer] Scanning agents for retraining...")
        for agent in registered_agents:
            try:
                flag_key = f"retraining_flag:{agent.name}"
                retrain_flag = shared_memory.get(flag_key)
                if retrain_flag:
                    logger.info(f"[Retrainer] 🔧 Retraining triggered for agent: {agent.name}")
                    agent.retrain()
                    shared_memory.set(flag_key, False)
                    logger.info(f"[Retrainer] ✅ Retraining complete for {agent.name}")
            except Exception as e:
                logger.error(f"[Retrainer] ❌ Failed to retrain {agent.name}: {e}", exc_info=True)

        time.sleep(interval)
