import asyncio
import time
import traceback

from typing import Annotated
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException

#from playwright.async_api import ElementHandle
#from playwright.async_api import Page

from core.utils.dom_helper import get_element_outer_html
from core.utils.dom_mutation_observer import subscribe  # type: ignore
from core.utils.dom_mutation_observer import unsubscribe  # type: ignore
from core.utils.ui_messagetype import MessageType
from src.agents.browser.manager import Manager
from logs.logger import get_logger

logger = get_logger("Click")

class DoClick:
    def __init__(self, driver):
        """Initialize with a Selenium WebDriver instance"""
        self.driver = driver
        self.browser_manager = Manager() 

    async def do_click(self, selector: str, wait_before_execution: float = 0.0) -> dict:
        """Asynchronous wrapper for click operations"""
        return await asyncio.to_thread(self._perform_click, selector, wait_before_execution)

    def _perform_click(self, selector: str, wait_before_execution: float) -> dict:
        """
        Core click functionality using Selenium
        Converted to synchronous for Selenium compatibility
        """
        logger.info(f"Executing click on '{selector}' with {wait_before_execution}s wait")
        
        try:
            if wait_before_execution > 0:
                time.sleep(wait_before_execution)

            element = self._wait_for_element(selector)
            if not element:
                raise ValueError(f"Element {selector} not found")

            self._prepare_element(element, selector)
            
            if self._handle_special_elements(element, selector):
                return self._build_success_message("Special element handled", element)

            return self._attempt_click(element, selector)
            
        except Exception as e:
            error_msg = f"Click failed on {selector}: {str(e)}"
            logger.error(error_msg)
            return self._build_error_message(error_msg)

    def _wait_for_element(self, selector: str, timeout: int = 2) -> WebElement:
        """Wait for element presence with explicit waits"""
        try:
            return WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
        except TimeoutException:
            return None

    def _prepare_element(self, element: WebElement, selector: str):
        """Element preparation handling"""
        self.driver.execute_script("arguments[0].scrollIntoViewIfNeeded();", element)
        try:
            WebDriverWait(self.driver, 0.2).until(
                EC.visibility_of(element)
            )
        except TimeoutException:
            logger.warning(f"Element {selector} not visible but proceeding")

    def _handle_special_elements(self, element: WebElement, selector: str) -> bool:
        """Special element handling"""
        tag_name = element.tag_name.lower()
        if tag_name == "option":
            value = element.get_attribute("value")
            select = Select(element.find_element(By.XPATH, "./.."))
            select.select_by_value(value)
            logger.info(f"Selected option: {value}")
            return True
        return False

    def _attempt_click(self, element: WebElement, selector: str) -> dict:
        """Click attempt sequence"""
        try:
            element.click()
            return self._build_success_message("Standard click succeeded", element)
        except WebDriverException as e:
            logger.warning(f"Standard click failed, trying JS: {str(e)}")
            return self._perform_javascript_click(element, selector)

    def _perform_javascript_click(self, element: WebElement, selector: str) -> dict:
        """JavaScript click fallback"""
        try:
            self.driver.execute_script("arguments[0].click();", element)
            return self._build_success_message("JavaScript click succeeded", element)
        except Exception as e:
            error_msg = f"All click attempts failed for {selector}"
            logger.error(error_msg)
            return self._build_error_message(error_msg)

    def _build_success_message(self, message: str, element: WebElement) -> dict:
        """Success response format"""
        return {
            "status": "success",
            "message": message,
            "element": {
                "tag": element.tag_name,
                "text": element.text[:50],
                "outer_html": element.get_attribute("outerHTML")[:200]
            }
        }

    def _build_error_message(self, error_msg: str) -> dict:
        """Error response format"""
        return {
            "status": "error",
            "message": error_msg,
            "element": None
        }

    def click_element(self, selector: str, wait_time: float = 0.0) -> str:
        """Public synchronous interface"""
        result = self._perform_click(selector, wait_time)
        return result["message"]
