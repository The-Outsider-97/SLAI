import os
import sys
import subprocess
import logging

class CollaborativeAgent:
    def __init__(self, shared_memory, task_router):
        self.shared_memory = shared_memory
        self.task_router = task_router
        self.logger = logging.getLogger("CollaborativeAgent")

    def preprocess(self, task_type, task_data):
        self.logger.debug(f"Preprocessing task: {task_type}")
        task_data["preprocessed"] = True

        # Example system-level enrichment: get current working directory
        task_data["cwd"] = os.getcwd()

        # Optional: verify Python environment
        task_data["python_version"] = sys.version

        return task_data

    def postprocess(self, task_type, result):
        self.logger.debug(f"Postprocessing result for task: {task_type}")
        result["postprocessed"] = True
        self.shared_memory.set(f"last_result_{task_type}", result)
        return result

    def execute(self, task_type, task_data):
        self.logger.info(f"CollaborativeAgent executing task: {task_type}")
        try:
            preprocessed_data = self.preprocess(task_type, task_data)
            result = self.task_router.route(task_type, preprocessed_data)
            processed_result = self.postprocess(task_type, result)
            self.logger.info(f"Task '{task_type}' executed successfully.")
            return processed_result
        except Exception as e:
            self.logger.error(f"Failed to execute task '{task_type}': {e}", exc_info=True)
            return self._system_fallback(task_type, task_data, str(e))

    def _system_fallback(self, task_type, task_data, error_message):
        self.logger.warning(f"Executing fallback for task: {task_type}")
        fallback_result = {
            "status": "fallback",
            "message": None,
            "original_error": error_message,
            "task_type": task_type
        }

        try:
            fallback_output = subprocess.check_output(["echo", f"Fallback triggered for {task_type}"], shell=True)
            decoded_output = fallback_output.decode("utf-8").strip()
            fallback_result["message"] = decoded_output
        except Exception as sub_error:
            fallback_result["status"] = "fallback-failed"
            fallback_result["message"] = str(sub_error)

        # Save fallback result to shared memory
        self.shared_memory.set(f"fallback_result_{task_type}", fallback_result)

        # Save fallback result to disk
        fallback_dir = "logs/fallbacks"
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_file = os.path.join(fallback_dir, f"fallback_{task_type}.log")
        try:
            with open(fallback_file, "w") as f:
                f.write(str(fallback_result))
        except Exception as file_error:
            self.logger.error(f"Failed to write fallback result to disk: {file_error}")

        return fallback_result

    def get_registered_tasks(self):
        return list(self.task_router.routing_table.keys())

    def get_agent_summary(self):
        return {
            "status": "ready",
            "registered_tasks": self.get_registered_tasks(),
            "shared_keys": list(self.shared_memory.data.keys())
        }

    def shutdown(self):
        self.logger.info("Shutting down CollaborativeAgent.")
        return True
