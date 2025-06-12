

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

    def clear_cache(self, expired_after=None):
        """Clear cache entries, optionally older than specified timestamp"""
        if not self.cache_dir:
            return
            
        if expired_after is None:
            self.cache.clear()
            return
            
        now = time.time()
        to_delete = []
        for key, entry in self.cache.items():
            if now - entry.get('timestamp', 0) > expired_after:
                to_delete.append(key)
        
        for key in to_delete:
            del self.cache[key]

    def save_cookies(self, path):
        """Persist cookies to disk"""
        self.cookie_jar.save(path, ignore_discard=True)

    def load_cookies(self, path):
        """Load cookies from disk"""
        self.cookie_jar.load(path, ignore_discard=True)

    def register_parser(self, name, parser_func):
        """Add custom parser to the parser registry"""
        self.parsers[name] = parser_func

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'cache') and not isinstance(self.cache, dict):
            self.cache.close()


if __name__ == "__main__":
    print("\n=== Running Execution Memory Test ===\n")
    printer.status("Init", "Execution Memory initialized", "success")

    memory = ExecutionMemory()
    print(memory)

    print("\n=== Simulation Complete ===")
