from selenium.common.exceptions import NoSuchElementException, WebDriverException

from logs.logger import PrettyPrinter, get_logger
from src.agents.browser.utils.config_loader import get_config_section, load_global_config

logger = get_logger("Scroll")
printer = PrettyPrinter


class DoScroll:
    def __init__(self, driver):
        self.config = load_global_config()
        self.scroll_config = get_config_section("do_scroll")
        self.driver = driver
        logger.info("Browser scrolling functionality initiated.")

    def scroll_to(self, x: int, y: int, smooth: bool = False) -> dict:
        try:
            behavior = "smooth" if smooth else "auto"
            self.driver.execute_script(
                "window.scrollTo({top: arguments[1], left: arguments[0], behavior: arguments[2]});",
                x,
                y,
                behavior,
            )
            return {"status": "success", "message": f"Scrolled to ({x},{y})"}
        except Exception as exc:
            logger.error("Scroll failed: %s", exc)
            return {"status": "error", "message": str(exc)}

    def scroll_element_into_view(self, selector: str, position: str = "center") -> dict:
        try:
            element = self.driver.find_element("css selector", selector)
            if position not in {"start", "center", "end", "nearest"}:
                position = "center"
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block: arguments[1], behavior: 'smooth'});",
                element,
                position,
            )
            return {"status": "success", "message": f"Scrolled to element {selector} ({position})"}
        except NoSuchElementException:
            return {"status": "error", "message": f"Element not found: {selector}"}
        except WebDriverException as exc:
            return {"status": "error", "message": str(exc)}

    def scroll_by(self, dx: int, dy: int, smooth: bool = False) -> dict:
        try:
            behavior = "smooth" if smooth else "auto"
            self.driver.execute_script(
                "window.scrollBy({top: arguments[1], left: arguments[0], behavior: arguments[2]});",
                dx,
                dy,
                behavior,
            )
            return {"status": "success", "message": f"Scrolled by ({dx},{dy})"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def scroll_direction(self, direction: str, amount: int = 200) -> dict:
        mapping = {"up": (0, -amount), "down": (0, amount), "left": (-amount, 0), "right": (amount, 0)}
        direction = direction.lower()
        if direction not in mapping:
            return {"status": "error", "message": f"Invalid direction: {direction}"}
        dx, dy = mapping[direction]
        return self.scroll_by(dx, dy)
