# ===== main.py - SLAI Core Launcher =====
import os
import sys
import logging
import yaml
import queue
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from src.collaborative.shared_memory import SharedMemory
from src.utils.agent_factory import AgentFactory
from src.utils.system_optimizer import SystemOptimizer
from src.agents.collaborative_agent import CollaborativeAgent
from frontend.startup_screen import StartupScreen
from frontend.main_window import MainWindow

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
logger = logging.getLogger('SLAI-Launcher')

def load_config(config_path: str = "config.yaml") -> dict:
    """Load and validate configuration file"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        logger.critical(f"Failed to load configuration: {str(e)}")
        sys.exit(1)

def initialize_core_components(config: dict) -> tuple:
    """Initialize fundamental system components"""
    # Shared memory for inter-agent communication
    shared_memory = SharedMemory(
        default_ttl=config.get('memory_ttl', 3600),
        max_versions=config.get('max_versions', 10)
    )
    
    # System optimization engine
    optimizer = SystemOptimizer()
    
    # Agent factory with shared resources
    agent_factory = AgentFactory(
        shared_resources={
            "shared_memory": shared_memory,
            "optimizer": optimizer
        },
        optimizer=optimizer
    )
    
    return shared_memory, optimizer, agent_factory

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
    
    # Load configuration and initialize components
    config = load_config()
    shared_memory, optimizer, agent_factory = initialize_core_components(config)
    
    # Create collaborative agent (system coordinator)
    collaborative_agent = CollaborativeAgent(
        shared_memory=shared_memory,
        agent_factory=agent_factory,
        agent_network=config.get("agent-network"),
        config_path="config.yaml",
        risk_threshold=config.get('risk_threshold', 0.35)
    )
    
    # Create communication queues
    log_queue = queue.Queue()
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

if __name__ == "__main__":
    # Windows multiprocessing support
    if sys.platform.startswith('win'):
        from multiprocessing import freeze_support
        freeze_support()
    
    main()
