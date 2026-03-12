from logs.logger import PrettyPrinter, get_logger
from src.agents.browser.utils.config_loader import get_config_section, load_global_config

logger = get_logger("Type")
printer = PrettyPrinter


class DoType:
    def __init__(self, driver):
        self.config = load_global_config()
        self.type_config = get_config_section("do_type")
        self.driver = driver
        logger.info("Browser typing functionality initiated.")

    def type_text(self, selector: str, raw_input: str, clear_before: bool = True) -> dict:
        try:
            element = self.driver.find_element("css selector", selector)
            if clear_before:
                element.clear()
            element.send_keys(raw_input)
            logger.info("Typed text into selector '%s'", selector)
            return {"status": "success", "message": f"Typed: {raw_input}"}
        except Exception as exc:
            logger.error("Typing failed: %s", exc)
            return {"status": "error", "message": str(exc)}
