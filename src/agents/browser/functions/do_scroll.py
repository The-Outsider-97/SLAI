

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, WebDriverException

from src.agents.browser.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Scroll")
printer = PrettyPrinter

class DoScroll:
    def __init__(self, driver):
        self.config = load_global_config()
        self.driver = self.config.get("driver")
        self.scroll_config = get_config_section("do_scroll")
        self.driver = driver
        driver = webdriver.Chrome()

        logger.info("Browser scrolling functionality initiated.")

    def scroll_to(self, x: int, y: int, smooth: bool = False) -> dict:
        try:
            behavior = "smooth" if smooth else "auto"
            self.driver.execute_script(
                f"window.scrollTo({{ top: {y}, left: {x}, behavior: '{behavior}' }});"
            )
            logger.info(f"Scrolled to ({x}, {y}) with smooth={smooth}")
            return {"status": "success", "message": f"Scrolled to ({x},{y})"}
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return {"status": "error", "message": str(e)}

    def scroll_element_into_view(self, selector: str, position: str = "center") -> dict:
        try:
            element = self.driver.find_element("css selector", selector)
            if position not in {"start", "center", "end", "nearest"}:
                position = "center"
            self.driver.execute_script(
                f"arguments[0].scrollIntoView({{block: '{position}', behavior: 'smooth'}});",
                element
            )
            logger.info(f"Scrolled element '{selector}' into view at '{position}'")
            return {"status": "success", "message": f"Scrolled to element {selector} ({position})"}
        except NoSuchElementException:
            logger.error(f"Element not found: {selector}")
            return {"status": "error", "message": f"Element not found: {selector}"}
        except WebDriverException as e:
            logger.error(f"Scroll failed: {e}")
            return {"status": "error", "message": str(e)}

    def scroll_by(self, dx: int, dy: int, smooth: bool = False) -> dict:
        try:
            behavior = "smooth" if smooth else "auto"
            self.driver.execute_script(
                f"window.scrollBy({{ top: {dy}, left: {dx}, behavior: '{behavior}' }});"
            )
            logger.info(f"Scrolled by dx={dx}, dy={dy}")
            return {"status": "success", "message": f"Scrolled by ({dx},{dy})"}
        except Exception as e:
            logger.error(f"ScrollBy failed: {e}")
            return {"status": "error", "message": str(e)}

    def scroll_direction(self, direction: str, amount: int = 200) -> dict:
        direction = direction.lower()
        dx, dy = 0, 0
        if direction == "up":
            dy = -amount
        elif direction == "down":
            dy = amount
        elif direction == "left":
            dx = -amount
        elif direction == "right":
            dx = amount
        else:
            return {"status": "error", "message": f"Invalid direction: {direction}"}

        return self.scroll_by(dx, dy)

# Interactive CLI test
if __name__ == "__main__":
    print("\n=== Running Browser Scrolling Functionality Test ===\n")
    printer.status("Init", "Browser scrolling functionality initialized", "success")
    driver = webdriver.Chrome()    

    # Inject mock driver
    scroll = DoScroll(driver=driver)

    print("\n=== Interactive Scroll Tester ===")
    print("Commands: scroll_to, scroll_by, scroll_element_into_view, scroll_direction")
    print("Type 'exit' to quit.\n")

    while True:
        cmd = input("Command: ").strip()
        if cmd == "exit":
            break
        elif cmd == "scroll_to":
            x = int(input("x: "))
            y = int(input("y: "))
            print(scroll.scroll_to(x, y))
        elif cmd == "scroll_by":
            dx = int(input("dx: "))
            dy = int(input("dy: "))
            print(scroll.scroll_by(dx, dy))
        elif cmd == "scroll_element_into_view":
            sel = input("CSS selector: ")
            pos = input("Alignment (start, center, end, nearest): ")
            print(scroll.scroll_element_into_view(sel, pos))
        elif cmd == "scroll_direction":
            dir = input("Direction (up/down/left/right): ")
            amt = int(input("Amount (px): "))
            print(scroll.scroll_direction(dir, amt))
        else:
            print("Unknown command.")

    print("\n=== Scroll Test Complete ===\n")
