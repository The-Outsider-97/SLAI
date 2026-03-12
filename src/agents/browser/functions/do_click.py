import asyncio
import time

from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

from logs.logger import get_logger

logger = get_logger("Click")


class DoClick:
    def __init__(self, driver):
        self.driver = driver

    async def do_click(self, selector: str, wait_before_execution: float = 0.0) -> dict:
        return await asyncio.to_thread(self._perform_click, selector, wait_before_execution)

    def _perform_click(self, selector: str, wait_before_execution: float) -> dict:
        try:
            if wait_before_execution > 0:
                time.sleep(wait_before_execution)

            element = self._wait_for_element(selector)
            if element is None:
                return self._build_error_message(f"Element not found: {selector}")

            self._prepare_element(element, selector)

            if self._handle_special_elements(element):
                return self._build_success_message("Special element handled", element)

            return self._attempt_click(element, selector)
        except Exception as exc:
            error_msg = f"Click failed on {selector}: {exc}"
            logger.error(error_msg)
            return self._build_error_message(error_msg)

    def _wait_for_element(self, selector: str, timeout: int = 2):
        try:
            return WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
        except TimeoutException:
            return None

    def _prepare_element(self, element: WebElement, selector: str):
        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
        try:
            WebDriverWait(self.driver, 1).until(EC.visibility_of(element))
        except TimeoutException:
            logger.warning("Element %s not visible but proceeding", selector)

    def _handle_special_elements(self, element: WebElement) -> bool:
        if element.tag_name.lower() == "option":
            value = element.get_attribute("value")
            parent = element.find_element(By.XPATH, "./..")
            Select(parent).select_by_value(value)
            return True
        return False

    def _attempt_click(self, element: WebElement, selector: str) -> dict:
        try:
            element.click()
            return self._build_success_message("Standard click succeeded", element)
        except WebDriverException as exc:
            logger.warning("Standard click failed for %s, trying JS fallback: %s", selector, exc)
            return self._perform_javascript_click(element, selector)

    def _perform_javascript_click(self, element: WebElement, selector: str) -> dict:
        try:
            self.driver.execute_script("arguments[0].click();", element)
            return self._build_success_message("JavaScript click succeeded", element)
        except Exception:
            return self._build_error_message(f"All click attempts failed for {selector}")

    def _build_success_message(self, message: str, element: WebElement) -> dict:
        return {
            "status": "success",
            "message": message,
            "element": {
                "tag": element.tag_name,
                "text": element.text[:50],
                "outer_html": (element.get_attribute("outerHTML") or "")[:200],
            },
        }

    def _build_error_message(self, error_msg: str) -> dict:
        return {"status": "error", "message": error_msg, "element": None}

    def click_element(self, selector: str, wait_time: float = 0.0) -> str:
        result = self._perform_click(selector, wait_time)
        return result["message"]
