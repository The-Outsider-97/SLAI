from loguru import logger
import sys
import logging

logger.remove()  # Remove default handler
logger.add(sys.stdout, level="INFO")  # Console logging
logger.add("logs/slai_monitor.log", rotation="10 MB", retention="10 days", level="DEBUG")

def log_event(message: str, level="info"):
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)

if __name__ == "__main__":
    log_event("Monitoring system initialized.", level="info")
