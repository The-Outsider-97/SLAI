import time
import random
from selenium.webdriver.common.action_chains import ActionChains

class Utility:
    @staticmethod
    def human_type(element, text, min_delay=0.05, max_delay=0.15):
        """Simulate human-like typing."""
        for character in text:
            element.send_keys(character)
            time.sleep(random.uniform(min_delay, max_delay))

    @staticmethod
    def human_click(driver, element):
        """Simulate human-like clicking."""
        actions = ActionChains(driver)
        actions.move_to_element(element).pause(random.uniform(0.1, 0.3)).click().perform()

    @staticmethod
    def select_link(query, elements):
        """Basic relevance-based link selection."""
        query_terms = set(query.lower().split())
        best_match = max(
            elements,
            key=lambda el: len(query_terms.intersection(el.text.lower().split())),
            default=elements[0]
        )
        return best_match
