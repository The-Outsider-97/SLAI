# ===== main.py - SLAI Core Launcher =====
import os
import sys
import yaml
import queue
import logging
import uvicorn
import concurrent
from threading import Thread
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication # no-audit

from logs.logger import get_logger, get_log_queue
from shared_memory_cleaner import SharedMemoryCleaner
from src.collaborative.shared_memory import SharedMemory
from src.utils.agent_factory import AgentFactory
from src.utils.system_optimizer import SystemOptimizer
from src.agents.collaborative_agent import CollaborativeAgent
from frontend.startup_screen import StartupScreen # no-audit
from frontend.main_window import MainWindow # no-audit



# Configure logging early to capture all events
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/slai_core.log'),
            logging.StreamHandler()
        ]
    )
logger = get_logger("SLAI-Launcher")

_config_cache = None

def load_config(config_path: str = "config.yaml") -> dict:
    global _config_cache
    if _config_cache is None:
        try:
            with open(config_path) as f:
                _config_cache = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.critical(f"Failed to load configuration: {str(e)}")
            sys.exit(1)
    return _config_cache

def initialize_core_components(config: dict) -> tuple:
    """Initialize fundamental system components"""
    shared_memory = SharedMemory()
    shared_memory.configure(
        default_ttl=config.get('memory_ttl', 3600),
        max_versions=config.get('max_versions', 10)
    )
    optimizer = SystemOptimizer()
    
    agent_factory = AgentFactory(
        config=config,
        shared_resources={
            "shared_memory": shared_memory,
            "optimizer": optimizer
        },
        optimizer=optimizer
    )
    
    return shared_memory, optimizer, agent_factory

#def start_research_api():
#    uvicorn.run("api.research_api:app", host="127.0.0.1", port=8000, log_level="info")

#api_thread = Thread(target=start_research_api, daemon=True)
#api_thread.start()
#logger.info("SLAI Research API started on http://127.0.0.1:8000")

def launch_ui(init_agents, components: dict):
    from PyQt5.QtWidgets import QSplashScreen
    from PyQt5.QtGui import QPixmap


    """Start the PyQt application with splash screen and main interface"""
    app = QApplication([])
    app.setStyle('Fusion')

    # 👇 Launches MainWindow after startup screen completes
    def start_main_ui():
        main_window = MainWindow(
            collaborative_agent=components["collaborative_agent"],
            shared_memory=components["shared_memory"],
            log_queue=components["log_queue"],
            metric_queue=components["metric_queue"],
            shared_resources=components["agent_factory"].shared_resources,
            optimizer=components["optimizer"]
        )
        main_window.show()
        main_window.showMaximized()

    splash = StartupScreen(on_ready_to_proceed=start_main_ui)
    splash.show()
    QTimer.singleShot(100, splash.notify_launcher_ready)

    sys.exit(app.exec_())

def main():
    """Core application entry point"""
    logger.info("Starting SLAI Core System")
    config = load_config()

    # Parallelize memory and factory creation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        shared_future = executor.submit(SharedMemory)
        optimizer_future = executor.submit(SystemOptimizer)
        
        shared_memory = shared_future.result()
        optimizer = optimizer_future.result()

    # Load configuration and initialize components
    shared_memory, optimizer, agent_factory = initialize_core_components(config)
    cleaner = SharedMemoryCleaner(shared_memory, interval=120)
    cleaner.start()    
    # Create collaborative agent (system coordinator)
    collaborative_agent = CollaborativeAgent(
        shared_memory=shared_memory,
        agent_factory=agent_factory,
        agent_network=config.get("agent-network"),
        config_path="config.yaml",
        risk_threshold=config.get('risk_threshold', 0.35)
    )
    
    # Create communication queues
    log_queue = get_log_queue()
    metric_queue = queue.Queue()
    
    # Prepare UI launch components
    launch_components = {
        'collaborative_agent': collaborative_agent,
        'shared_memory': shared_memory,
        'optimizer': optimizer,
        'agent_factory': agent_factory,
        'log_queue': log_queue,
        'metric_queue': metric_queue
    }

    logger.info("Launching SLAI Interface")
    launch_ui(lambda: None, launch_components)

class IdleMonitor:
    def __init__(self, idle_threshold=1800):  # 30 mins
        import time
        self.last_active = time.time()
        self.idle_threshold = idle_threshold
        self.running = True
        Thread(target=self._watchdog, daemon=True).start()

    def reset_timer(self):
        import time
        self.last_active = time.time()

    def _watchdog(self):
        import time
        while self.running:
            if time.time() - self.last_active > self.idle_threshold:
                self._trigger_audit()
                self.last_active = time.time()  # Reset after audit
            time.sleep(60)  # Check every minute

    def _run_pylint(target_path):
        import subprocess
        try:
            result = subprocess.run(
                ["pylint", target_path, "--output-format=text", "--score=n"],
                capture_output=True, text=True
            )
            with open("logs/pylint_audit.log", "w") as f:
                f.write(result.stdout)
        except Exception as e:
            logging.getLogger("SLAI.Pylint").error(f"Pylint failed: {e}")

    def _trigger_audit(self):
        from models.auditor import CodeAuditor, _run_pylint_scan
        auditor = CodeAuditor(target_path="src/")
        issues = auditor.run_audit()
        auditor.log_issues(issues)
        _run_pylint_scan("src/")


if __name__ == "__main__":
    # Windows multiprocessing support
    if sys.platform.startswith('win'):
        from multiprocessing import freeze_support
        freeze_support()
    
    main()
