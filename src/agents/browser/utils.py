import random
import time

from selenium.webdriver.common.action_chains import ActionChains


class Utility:
    @staticmethod
    def human_type(element, text, min_delay=0.05, max_delay=0.15):
        for character in text:
            element.send_keys(character)
            time.sleep(random.uniform(min_delay, max_delay))

    @staticmethod
    def human_click(driver, element):
        actions = ActionChains(driver)
        actions.move_to_element(element).pause(random.uniform(0.1, 0.3)).click().perform()

    @staticmethod
    def select_link(query, elements):
        if not elements:
            raise ValueError("No elements available for selection")
        query_terms = set(query.lower().split())
        return max(
            elements,
            key=lambda el: len(query_terms.intersection((el.text or "").lower().split())),
        )
