import traceback

import pyperclip
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.common.by import By

from logs.logger import PrettyPrinter, get_logger
from src.agents.browser.utils.config_loader import get_config_section, load_global_config

logger = get_logger("CopyCutPaste")
printer = PrettyPrinter


class DoCopyCutPaste:
    def __init__(self, driver):
        self.config = load_global_config()
        self.ccp_config = get_config_section("do_ccp")
        self.driver = driver
        logger.info("Web Copy, Cut & Paste functionality initiated.")

    def _get_element_text(self, element) -> str:
        text = (element.text or "").strip()
        if not text:
            text = (element.get_attribute("value") or "").strip()
        if not text:
            text = (element.get_attribute("placeholder") or "").strip()
        return text

    def _extract_context_metadata(self, element) -> dict:
        return {
            "tag": element.tag_name,
            "role": element.get_attribute("role"),
            "name": element.get_attribute("name") or element.get_attribute("id"),
            "placeholder": element.get_attribute("placeholder"),
            "aria_label": element.get_attribute("aria-label"),
            "class": element.get_attribute("class"),
            "outer_html": (element.get_attribute("outerHTML") or "")[:400],
        }

    def copy(self, selector: str) -> dict:
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            text = self._get_element_text(element)
            pyperclip.copy(text)
            return {
                "status": "success",
                "action": "copy",
                "text": text,
                "metadata": self._extract_context_metadata(element),
            }
        except NoSuchElementException:
            return {"status": "error", "message": f"Selector '{selector}' not found"}
        except WebDriverException as exc:
            return {"status": "error", "message": f"Selenium failure: {exc}"}
        except Exception as exc:
            return {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}

    def cut(self, selector: str) -> dict:
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            text = self._get_element_text(element)
            pyperclip.copy(text)
            element.clear()
            return {
                "status": "success",
                "action": "cut",
                "text": text,
                "metadata": self._extract_context_metadata(element),
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}

    def paste(self, selector: str) -> dict:
        try:
            paste_text = pyperclip.paste()
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            element.clear()
            element.send_keys(paste_text)
            return {
                "status": "success",
                "action": "paste",
                "text": paste_text,
                "metadata": self._extract_context_metadata(element),
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc), "traceback": traceback.format_exc()}
