import logging
import os, sys
import time
import difflib
import traceback

class BaseAgent:
    def __init__(self, shared_memory, agent_factory, config=None):
            self.shared_memory=shared_memory,
            self.agent_factory=agent_factory,
            self.config=config or {}
            self.logger = logging.getLogger(self.__class__.__name__)
            self.name = self.__class__.__name__

    def execute(self, input_data):
        try:
            self.logger.info(f"[{self.name}] Executing task...")
            result = self.perform_task(input_data)
            self.logger.info(f"[{self.name}] Task execution completed.")
            
            # Log successful stats
            self.shared_memory.set(f"agent_stats:{self.name}", {
                "last_run": time.time(),
                "success": True,
                "result_summary": str(result)[:200]
            })

            return result

        except Exception as e:
            error_entry = {
                "timestamp": time.time(),
                "error": str(e),
                "traceback": traceback.format_exc()
            }

            # Retrieve and trim old errors
            error_key = f"errors:{self.name}"
            MAX_ERRORS = 50
            errors = self.shared_memory.get(error_key) or []
            errors.append(error_entry)
            if len(errors) > MAX_ERRORS:
                errors = errors[-MAX_ERRORS:]
            self.shared_memory.set(error_key, errors)

            # Log failure stats
            self.shared_memory.set(f"agent_stats:{self.name}", {
                "last_run": time.time(),
                "success": False,
                "error": str(e)
            })

            self.logger.error(f"[{self.name}] Task failed with error: {e}")
            self.logger.debug(traceback.format_exc())

            # Match against recent errors for similarity
            SIMILARITY_THRESHOLD = 0.75
            new_error = str(e)
            new_type = type(e).__name__

            for past in reversed(errors[:-1]):
                past_error = past.get("error", "")
                past_type = past_error.split(":")[0] if ":" in past_error else ""

                # Match exception type first (e.g., ValueError vs TypeError)
                if past_type == new_type:
                    # Check textual similarity
                    similarity = difflib.SequenceMatcher(None, new_error, past_error).ratio()
                    if similarity >= SIMILARITY_THRESHOLD:
                        self.logger.warning(
                            f"[{self.name}] A similar {new_type} occurred before "
                            f"(similarity: {similarity:.2f})"
                        )
                        break

            return {"status": "failed", "error": str(e)}
    
    def alternative_execute(self, task_data):
        """
        Fallback logic when normal execution fails or matches a known failure pattern.
        Attempts to simplify, sanitize, or reroute the input for safer processing.
        """
        try:
            # Step 1: Sanitize task data (remove noise, normalize casing, trim tokens)
            if isinstance(task_data, str):
                clean_data = task_data.strip().lower().replace('\n', ' ')
            elif isinstance(task_data, dict) and "text" in task_data:
                clean_data = task_data["text"].strip().lower()
            else:
                clean_data = str(task_data).strip()

            # Step 2: Apply a safer, simplified prompt or fallback logic
            fallback_prompt = f"Can you try again with simplified input:\n{clean_data}"
            if hasattr(self, "llm") and callable(getattr(self.llm, "generate", None)):
                return self.llm.generate(fallback_prompt)

            # Step 3: If the agent wraps another processor (e.g. GrammarProcessor, LLM), reroute
            if hasattr(self, "grammar") and callable(getattr(self.grammar, "compose_sentence", None)):
                facts = {"event": "fallback", "value": clean_data}
                return self.grammar.compose_sentence(facts)

            # Step 4: Otherwise just echo the cleaned input as confirmation
            return f"[Fallback response] I rephrased your input: {clean_data}"

        except Exception as e:
            # Final fallback — very safe and generic
            return "[Fallback failure] Unable to process your request at this time."
    
    def is_similar(self, task_data, past_task_data):
        """
        Compares current task with past task to detect similarity.
        Uses key overlap and value resemblance heuristics.
        """
        if type(task_data) != type(past_task_data):
            return False
    
        # Handle simple text-based tasks
        if isinstance(task_data, str) and isinstance(past_task_data, str):
            return task_data.strip().lower() == past_task_data.strip().lower()
    
        # Handle dict-based structured tasks
        if isinstance(task_data, dict) and isinstance(past_task_data, dict):
            shared_keys = set(task_data.keys()) & set(past_task_data.keys())
            similarity_score = 0
            for key in shared_keys:
                if isinstance(task_data[key], str) and isinstance(past_task_data[key], str):
                    if task_data[key].strip().lower() == past_task_data[key].strip().lower():
                        similarity_score += 1
            # Consider similar if 50% or more keys match closely
            return similarity_score >= (len(shared_keys) / 2)
    
        return False

    def handle_known_issue(self, task_data, error):
        """
        Attempt to recover from known failure patterns.
        Could apply input transformation or fallback logic.
        """
        self.logger.warning(f"Handling known issue from error: {error.get('error')}")
    
        # Fallback strategy #1: remove problematic characters
        if isinstance(task_data, str):
            cleaned = task_data.replace("🧠", "").replace("🔥", "")
            self.logger.info(f"Retrying with cleaned input: {cleaned}")
            return self.perform_task(cleaned)
    
        # Fallback strategy #2: modify specific fields in structured input
        if isinstance(task_data, dict):
            cleaned_data = task_data.copy()
            for key, val in cleaned_data.items():
                if isinstance(val, str) and "emoji" in error.get("error", ""):
                    cleaned_data[key] = val.encode("ascii", "ignore").decode()
            self.logger.info("Retrying task with cleaned structured data.")
            return self.perform_task(cleaned_data)
    
        # Fallback strategy #3: return a graceful degradation response
        self.logger.warning("Returning fallback response for unresolvable input.")
        return {"status": "failed", "reason": "Repeated known issue", "fallback": True}
    
    def perform_task(self, task_data):
        """
        Simulated execution method — replace with actual agent logic.
        This is where core functionality would happen.
        """
        self.logger.info(f"Executing task with data: {task_data}")
    
        if isinstance(task_data, str) and "fail" in task_data.lower():
            raise ValueError("Simulated failure due to blacklisted word.")
    
        if isinstance(task_data, dict):
            # Simulate failure on missing required keys
            required_keys = ["input", "context"]
            for key in required_keys:
                if key not in task_data:
                    raise KeyError(f"Missing required key: {key}")
    
        # Simulate result
        return {"status": "success", "result": f"Processed: {task_data}"}
