import logging
import random
import time

from robotexclusionrulesparser import RobotExclusionRulesParser
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from src.agents.browser.content import ContentHandling
from src.agents.browser.security import SecurityFeatures, exponential_backoff
from src.agents.browser.utils import Utility

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
MAX_RETRIES = 5
WINDOW_SIZE = (random.randint(1200, 1400), random.randint(800, 1000))


class WorkFlow:
    def __init__(self, driver):
        self.driver = driver
        self.robots = RobotExclusionRulesParser()
        self.driver.set_window_size(*WINDOW_SIZE)

    def _safe_get(self, url: str) -> bool:
        retries = 0
        while retries < MAX_RETRIES:
            try:
                self.driver.get(url)
                if SecurityFeatures.detect_captcha(self.driver):
                    raise RuntimeError("CAPTCHA detected during navigation")
                return True
            except Exception as exc:
                logger.warning("Navigation error (%s): %s", url, exc)
                retries += 1
                exponential_backoff(retries)
        return False

    def web_agent(self, query: str, max_depth: int = 2):
        try:
            if not self._safe_get("https://www.google.com"):
                return

            search_box = self.driver.find_element(By.NAME, "q")
            Utility.human_type(search_box, query)
            search_box.send_keys(Keys.RETURN)
            time.sleep(random.uniform(1, 2))

            for _ in range(max_depth):
                if SecurityFeatures.detect_captcha(self.driver):
                    logger.warning("CAPTCHA detected! Human intervention required.")
                    return

                links = self.driver.find_elements(By.XPATH, "//a[@href]")
                valid_links = []
                for link in links:
                    href = link.get_attribute("href")
                    if not href or not href.startswith("http"):
                        continue
                    if self.robots.is_allowed(USER_AGENT, href):
                        valid_links.append(link)

                if not valid_links:
                    break

                chosen_link = Utility.select_link(query, valid_links)
                logger.info("Selected: %s... (%s)", chosen_link.text[:50], chosen_link.get_attribute("href"))

                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        Utility.human_click(self.driver, chosen_link)
                        time.sleep(random.uniform(1, 2))
                        if SecurityFeatures.detect_captcha(self.driver):
                            raise RuntimeError("CAPTCHA detected after click")
                        break
                    except Exception as exc:
                        logger.warning("Navigation click error: %s", exc)
                        self.driver.back()
                        retries += 1
                        exponential_backoff(retries)

                current_url = self.driver.current_url
                if current_url.endswith(".pdf"):
                    content = ContentHandling.handle_pdf(current_url)
                    logger.info("PDF Content: %s...", content[:500])
                elif "arxiv.org" in current_url:
                    content = ContentHandling.handle_arxiv(self.driver)
                    logger.info("ArXiv Abstract: %s...", content[:500])
                else:
                    content = self.driver.find_element(By.TAG_NAME, "body").text[:1000]
                    logger.info("Content Preview: %s", content.replace("\n", " "))
        finally:
            self.driver.quit()
