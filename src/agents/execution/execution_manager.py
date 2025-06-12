

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Execution Manager")
printer = PrettyPrinter

class ExecutionManager:
    def __init__(self):
        self.config = load_global_config()
        self.manager_config = get_config_section('')

        logger.info(f"Execution Manager succesfully initialized")

if __name__ == "__main__":
    print("\n=== Running Execution Manager Test ===\n")
    printer.status("Init", "Execution Manager initialized", "success")

    manager = ExecutionManager()
    print(manager)

    print("\n=== Simulation Complete ===")
