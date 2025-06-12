

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Execution Memory")
printer = PrettyPrinter

class ExecutionMemory:
    def __init__(self):
        """
        Memory for:
        - Cookie Management
        - Caching
        - Checkpointing and tagging
        """
        self.config = load_global_config()
        self.manager_config = get_config_section('execution_memory')

        logger.info(f"Execution Manager succesfully initialized")

if __name__ == "__main__":
    print("\n=== Running Execution Memory Test ===\n")
    printer.status("Init", "Execution Memory initialized", "success")

    memory = ExecutionMemory()
    print(memory)

    print("\n=== Simulation Complete ===")
