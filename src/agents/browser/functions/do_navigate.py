import time
from urllib.parse import urlparse

from selenium.common.exceptions import WebDriverException

from logs.logger import PrettyPrinter, get_logger
from src.agents.browser.utils.config_loader import get_config_section, load_global_config

logger = get_logger("Navigate")
printer = PrettyPrinter


class DoNavigate:
    def __init__(self, driver):
        self.config = load_global_config()
        self.scroll_config = get_config_section("do_navigate")
        self.driver = driver
        self.history = []
        logger.info("Web navigation functionality initiated.")

    def _is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)

    def _wait_for_page_load(self, timeout: int = 5):
        start = time.time()
        while time.time() - start < timeout:
            state = self.driver.execute_script("return document.readyState")
            if state == "complete":
                return True
            time.sleep(0.1)
        return False

    def go_to_url(self, url: str) -> dict:
        if not self._is_valid_url(url):
            return {"status": "error", "message": f"Invalid URL: {url}"}
        try:
            self.driver.get(url)
            self._wait_for_page_load()
            self.history.append(url)
            return {"status": "success", "message": f"Navigated to {url}", "url": url}
        except WebDriverException as exc:
            return {"status": "error", "message": str(exc)}

    def go_back(self) -> dict:
        try:
            self.driver.back()
            self._wait_for_page_load()
            return {"status": "success", "message": "Went back"}
        except WebDriverException as exc:
            return {"status": "error", "message": str(exc)}

    def go_forward(self) -> dict:
        try:
            self.driver.forward()
            self._wait_for_page_load()
            return {"status": "success", "message": "Went forward"}
        except WebDriverException as exc:
            return {"status": "error", "message": str(exc)}

    def refresh_page(self) -> dict:
        try:
            self.driver.refresh()
            self._wait_for_page_load()
            return {"status": "success", "message": "Page refreshed"}
        except WebDriverException as exc:
            return {"status": "error", "message": str(exc)}

    def get_current_url(self) -> dict:
        try:
            return {"status": "success", "url": self.driver.current_url}
        except WebDriverException as exc:
            return {"status": "error", "message": str(exc)}

    def get_navigation_history(self) -> list:
        return self.history
