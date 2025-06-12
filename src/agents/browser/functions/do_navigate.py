
import re
import time

from selenium import webdriver
from urllib.parse import urlparse
from selenium.common.exceptions import WebDriverException

from src.agents.browser.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Navigate")
printer = PrettyPrinter

class DoNavigate:
    def __init__(self, driver):
        self.config = load_global_config()
        self.scroll_config = get_config_section("do_navigate")
        self.driver = driver
        driver = webdriver.Chrome()
        self.history = []

        logger.info(f"Web navigation functionality initiated.")

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
            logger.warning(f"Invalid URL: {url}")
            return {"status": "error", "message": f"Invalid URL: {url}"}

        try:
            self.driver.get(url)
            self._wait_for_page_load()
            self.history.append(url)
            logger.info(f"Navigated to: {url}")
            return {"status": "success", "message": f"Navigated to {url}", "url": url}
        except WebDriverException as e:
            logger.error(f"Navigation error: {e}")
            return {"status": "error", "message": str(e)}

    def go_back(self) -> dict:
        try:
            self.driver.back()
            self._wait_for_page_load()
            logger.info("Went back")
            return {"status": "success", "message": "Went back"}
        except WebDriverException as e:
            return {"status": "error", "message": str(e)}

    def go_forward(self) -> dict:
        try:
            self.driver.forward()
            self._wait_for_page_load()
            logger.info("Went forward")
            return {"status": "success", "message": "Went forward"}
        except WebDriverException as e:
            return {"status": "error", "message": str(e)}

    def refresh_page(self) -> dict:
        try:
            self.driver.refresh()
            self._wait_for_page_load()
            logger.info("Page refreshed")
            return {"status": "success", "message": "Page refreshed"}
        except WebDriverException as e:
            return {"status": "error", "message": str(e)}

    def get_current_url(self) -> dict:
        try:
            current_url = self.driver.current_url
            return {"status": "success", "url": current_url}
        except WebDriverException as e:
            return {"status": "error", "message": str(e)}

    def get_navigation_history(self) -> list:
        return self.history


if __name__ == "__main__":
    print("\n=== Running Web navigation functionality Test ===\n")
    printer.status("Init", "Web navigation functionality initialized", "success")
    driver = webdriver.Chrome()

    navigator = DoNavigate(driver=driver)
    print(navigator)
    print("\n* * * * * Phase 2 * * * * *\n")
    navigator.go_to_url("https://google.com")
    navigator.get_current_url()
    navigator.go_back()
    navigator.refresh_page()

    print("\n=== Successfully Ran Web navigation functionality ===\n")
