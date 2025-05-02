import torch.nn as nn
import time

from logs.logger import get_logger

FALLBACK_PLANS = {
    "TranslateAndSummarize": ["Translate", "Summarize"],
    "AnalyzeData": ["PreprocessData", "Analyze"],
    "ExplainConcept": ["RetrieveFact", "Summarize"]
}

logger = get_logger("SLAI.TaskRouter")

class AdaptiveRouter:
    def __init__(self, config):
        from src.agents.planning.task_scheduler import DeadlineAwareScheduler
        self.scheduler = DeadlineAwareScheduler(
            config['risk_threshold'],
            config['retry_policy']
        )
        self.policy = nn.Sequential(
            nn.Linear(config['state_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, config['num_handlers'])
        )

    def route_message(self, message, routing_table: dict):
        """Enhanced routing with task scheduling"""
        for condition, handler in routing_table.items():
            if condition in message:
                return handler(message)
        # Convert message to task format
        task = self._message_to_task(message)
        
        # Get available handlers from agent factory
        handlers = self.agent_factory.get_available_agents()
        
        # Schedule using risk-aware scheduler
        schedule = self.task_scheduler.schedule(
            tasks=[task],
            agents=handlers,
            risk_assessor=self.alignment_monitor.assess_risk
        )
        
        if not schedule:
            logger.warning("Routing failed, using fallback strategy")
            return super().route_message(message)

        # Extract best handler from schedule
        handler_id = next(iter(schedule.values()))['agent_id']
        reward = self._calculate_routing_reward(message, handler_id)
        
        # Store experience with priority
        self._store_experience(message, handler_id, reward)
        
        return handler_id

    def _message_to_task(self, message):
        """Convert message to scheduler task format"""
        return {
            'id': message.get('message_id', hash(message)),
            'requirements': self._extract_requirements(message),
            'deadline': time.time() + message.get('ttl', 60),
            'risk_score': self.alignment_monitor.assess_risk(message).get('risk_score', 0.5),
            'metadata': message.get('metadata', {})
        }


class TaskRouter:
    FALLBACK_PLANS = {
        "train_model": ["retry_simple_trainer", "notify_human"],
        "data_audit": ["emergency_data_cleaner"]
    }

    def __init__(self, registry, shared_memory):
        self.registry = registry
        self.shared_memory = shared_memory

    def route(self, task_type, task_data):
        eligible_agents = self.registry.get_agents_by_task(task_type)
        context = self.shared_memory.get('task_context', {})

        if not eligible_agents:
            raise Exception(f"No agents found for task type '{task_type}'")

        # Step 1: Rank agents by success history or priority
        sorted_agents = self._rank_agents(eligible_agents)

        # Step 2: Try each agent in order until success
        for agent_name, agent in sorted_agents:
            try:
                # Increment active tasks BEFORE execution
                agent_stats = self.shared_memory.get("agent_stats", {})
                current_tasks = agent_stats.get(agent_name, {}).get("active_tasks", 0)
                agent_stats[agent_name]["active_tasks"] = current_tasks + 1  # Thread-safe via SharedMemory

                self.shared_memory.set("agent_stats", agent_stats)
                logger.info(f"Routing task '{task_type}' to agent: {agent_name}")
                result = agent.execute(task_data)

                # Decrement on SUCCESS
                agent_stats = self.shared_memory.get("agent_stats", {})
                agent_stats[agent_name]["active_tasks"] = max(0, current_tasks - 1)
                self.shared_memory.set("agent_stats", agent_stats)

                # Step 3: Log success to shared memory
                self._record_success(agent_name)
                return result

            # Error handling
            except Exception as e:
                # Decrement on FAILURE
                agent_stats = self.shared_memory.get("agent_stats", {})
                agent_stats[agent_name]["active_tasks"] = max(0, current_tasks - 1)
                self.shared_memory.set("agent_stats", agent_stats)
                logger.exception(f"Agent '{agent_name}' failed...")
                self._record_failure(agent_name)
                
                #Fallback logic
                if task_type in self.FALLBACK_PLANS:
                    for subtask in self.FALLBACK_PLANS[task_type]:
                        logger.info(f"Attempting fallback: {subtask}")
                        try:
                            return self.route(subtask, task_data)  # Recursive retry
                        except Exception:
                            raise RuntimeError(f"No agents found for task type '{task_type}'")

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
