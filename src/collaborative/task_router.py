import logging

from src.collaborative.registry import AgentRegistry

FALLBACK_PLANS = {
    "TranslateAndSummarize": ["Translate", "Summarize"],
    "AnalyzeData": ["PreprocessData", "Analyze"],
    "ExplainConcept": ["RetrieveFact", "Summarize"]
}

logger = logging.getLogger("SLAI.TaskRouter")
logger.setLevel(logging.INFO)

class TaskRouter:
    FALLBACK_PLANS = {
        "train_model": ["retry_simple_trainer", "notify_human"],
        "data_audit": ["emergency_data_cleaner"]
    }

    def __init__(self, registry, shared_memory):
        self.registry = registry
        self.shared_memory = shared_memory

    def route(self, task_type, task_data, context=None):
        eligible_agents = self.registry.get_agents_by_task(task_type)
        context = self.shared_memory.get('task_context', {})

        if not eligible_agents:
            raise Exception(f"No agents found for task type '{task_type}'")

        # Step 1: Rank agents by success history or priority
        sorted_agents = self._rank_agents(eligible_agents)

        # Step 2: Try each agent in order until success
        for agent_name, agent in sorted_agents:
            try:
                logger.info(f"Routing task '{task_type}' to agent: {agent_name}")
                result = agent.execute(task_data)

                # Step 3: Log success to shared memory
                self._record_success(agent_name)
                return result

            # Error handling
            except Exception as e:
                logger.exception(f"Agent '{agent_name}' failed...")
                self._record_failure(agent_name)
                
                #Fallback logic
                if task_type in self.FALLBACK_PLANS:
                    for subtask in self.FALLBACK_PLANS[task_type]:
                        logger.info(f"Attempting fallback: {subtask}")
                        try:
                            return self.route(subtask, task_data)  # Recursive retry
                        except Exception:
                            logger.warning(f"Subtask '{subtask}' failed: {e}")
                            continue

                    else:
                        logger.info(f"Fallback plan for '{task_type}' succeeded.")
                        self._record_success(agent_name)
                        return result

        # If all fail
        raise Exception(f"All agents failed for task type '{task_type}'")

    def _rank_agents(self, agents):
        ranked = []
        agent_stats = self.shared_memory.get("agent_stats", {})
        
        for name, agent in agents.items():
            meta = agent_stats.get(name, {})
            success = meta.get("successes", 0)
            failures = meta.get("failures", 0)
            total = success + failures
            
            # Dynamic weighting
            success_rate = success / total if total > 0 else 1.0  # Favor new agents
            priority = meta.get("priority", 0) * 0.2  # Configurable weight
            load = meta.get("active_tasks", 0) * 0.3  # Penalize busy agents
            
            score = success_rate + priority - load
            ranked.append((name, agent, score))
        
        # Sort by score descending
        return sorted(ranked, key=lambda x: x[2], reverse=True)

    def _record_success(self, agent_name):
        stats = self.shared_memory.get("agent_stats", {})
        entry = stats.get(agent_name, {"successes": 0, "failures": 0, "priority": 0})
        entry["successes"] += 1
        self.shared_memory.put("agent_stats", stats)

    def _record_failure(self, agent_name):
        stats = self.shared_memory.setdefault("agent_stats", {})
        entry = stats.setdefault(agent_name, {"successes": 0, "failures": 0, "priority": 0})
        entry["failures"] += 1
        self.shared_memory.put("agent_stats", stats)
