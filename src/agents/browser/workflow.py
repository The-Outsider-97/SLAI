import time
import random
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from robotexclusionrulesparser import RobotExclusionRulesParser
from src.agents.browser.content import ContentHandling
from src.agents.browser.utils import Utility

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Or DEBUG
logger.info("Activating Research Mode with Browser Agent")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
MAX_RETRIES = 5
WINDOW_SIZE = (random.randint(1200, 1400), random.randint(800, 1000))

class WorkFlow:
    def __init__(self, driver):
        self.driver = driver
        self.robots = RobotExclusionRulesParser()

    def web_agent(query, max_depth=2):
        try:
            # Navigate with retry logic
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    driver.get("https://www.google.com")
                    if detect_captcha(driver):
                        raise Exception("CAPTCHA detected during initial navigation")
                    break
                except Exception as e:
                    print(f"Error: {str(e)} - Retrying...")
                    retries += 1
                    exponential_backoff(retries)
            
            search_box = driver.find_element(By.NAME, "q")
            Utility.human_type(search_box, query)
            search_box.send_keys(Keys.RETURN)
            time.sleep(random.uniform(1, 2))

            for _ in range(max_depth):
                if detect_captcha(driver):
                    print("CAPTCHA detected! Human intervention required.")
                    return

                links = driver.find_elements(By.XPATH, "//a[@href]")
                valid_links = [
                    l for l in links 
                    if "http" in l.get_attribute("href") and 
                    robots.is_allowed(USER_AGENT, l.get_attribute("href"))
                ]

                if not valid_links:
                    break

                chosen_link = Utility.select_link(query, valid_links)
                print(f"Selected: {chosen_link.text[:50]}... ({chosen_link.get_attribute('href')})")

                # Enhanced navigation with retries
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        Utility.human_click(driver, chosen_link)
                        time.sleep(random.uniform(1, 2))
                        if detect_captcha(driver):
                            raise Exception("CAPTCHA detected after click")
                        break
                    except Exception as e:
                        print(f"Navigation error: {str(e)} - Retrying...")
                        driver.back()
                        retries += 1
                        exponential_backoff(retries)

                # Special content handling
                current_url = driver.current_url
                if current_url.endswith(".pdf"):
                    content = ContentHandling.handle_pdf(current_url)
                    print(f"PDF Content: {content[:500]}...\n---\n")
                elif "arxiv.org" in current_url:
                    content = ContentHandling.handle_arxiv(driver)
                    print(f"ArXiv Abstract: {content[:500]}...\n---\n")
                else:
                    content = driver.find_element(By.TAG_NAME, "body").text[:1000]
                    print(f"Content Preview: {content.replace('\n', ' ')}\n---\n")

        finally:
            driver.quit()
