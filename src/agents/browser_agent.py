from __future__ import annotations

__version__ = "2.0.0"

"""Browser agent facade that composes browser domain modules for web automation tasks."""

from datetime import time
from typing import Any, Dict, List, Optional

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from src.agents.base.utils.main_config_loader import get_config_section, load_global_config
from src.agents.base_agent import BaseAgent
from src.agents.browser.content import ContentHandling
from src.agents.browser.functions.do_click import DoClick
from src.agents.browser.functions.do_copy_cut_paste import DoCopyCutPaste
from src.agents.browser.functions.do_navigate import DoNavigate
from src.agents.browser.functions.do_scroll import DoScroll
from src.agents.browser.functions.do_type import DoType
from src.agents.browser.security import SecurityFeatures, exponential_backoff
from src.agents.browser.workflow import WorkFlow
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Browser Agent")
printer = PrettyPrinter


class BrowserAgent(BaseAgent):
    """High-level browser orchestration agent.

    Responsibilities:
    - Own browser lifecycle
    - Orchestrate retries/security checks
    - Delegate concrete interactions to browser function modules
    - Expose a stable agent-style API + perform_task contract
    """

    def __init__(self, shared_memory, agent_factory, config=None, driver=None):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = load_global_config()
        self.browser_agent_config = get_config_section("browser_agent")

        self.max_retries = self.browser_agent_config.get("max_retries", 3)
        self.default_wait = self.browser_agent_config.get("default_wait", 0.0)
        self.default_search_engine = self.browser_agent_config.get("default_search_engine", "https://www.google.com")
        self.headless = self.browser_agent_config.get("headless", True)
        self.user_agent = self.browser_agent_config.get(
            "user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        self.window_size = self.browser_agent_config.get("window_size", [1366, 920])

        self.driver = driver or self._init_browser()

        # Compose domain-level browser modules
        self.navigator = DoNavigate(self.driver)
        self.clicker = DoClick(self.driver)
        self.scroller = DoScroll(self.driver)
        self.typer = DoType(self.driver)
        self.clipboard = DoCopyCutPaste(self.driver)
        self.workflow = WorkFlow()

        logger.info("BrowserAgent initialized with composed browser modules.")

    def _init_browser(self):
        options = webdriver.ChromeOptions()
        options.add_argument(f"user-agent={self.user_agent}")
        options.add_argument(f"window-size={self.window_size[0]},{self.window_size[1]}")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        if self.headless:
            options.add_argument("--headless=new")
        return webdriver.Chrome(options=options)

    def _with_retry(self, operation_name: str, operation, *args, **kwargs) -> Dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                result = operation(*args, **kwargs)
                if SecurityFeatures.detect_captcha(self.driver):
                    raise RuntimeError("CAPTCHA detected")
                return result
            except Exception as exc:
                logger.warning("%s failed on attempt %s: %s", operation_name, attempt + 1, exc)
                if attempt >= self.max_retries:
                    return {"status": "error", "action": operation_name, "message": str(exc)}
                exponential_backoff(attempt)
        return {"status": "error", "action": operation_name, "message": "Unknown retry failure"}

    def navigate(self, url: str) -> Dict[str, Any]:
        result = self._with_retry("navigate", self.navigator.go_to_url, url)
        if result.get("status") == "success":
            result["page"] = self.extract_page_content(preview_only=True)
        return result

    def search(self, query: str, engine_url: Optional[str] = None, search_box_selector: str = "input[name='q']") -> Dict[str, Any]:
        navigate_result = self.navigate(engine_url or self.default_search_engine)
        if navigate_result.get("status") != "success":
            return navigate_result
    
        # Wait for page to load and handle cookie consent
        self._handle_cookie_consent()
    
        # Try multiple selectors for the search box
        possible_selectors = [
            search_box_selector,
            "textarea[name='q']",          # newer Google interface
            "input[type='text']",           # generic
            "input[type='search']",
            "[aria-label='Search']",
            "[aria-label='Cerca']",          # other languages
            ".gLFyf",                        # Google's class for search input
        ]
        
        element = None
        for selector in possible_selectors:
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                break
            except TimeoutException:
                continue
    
        if not element:
            return {"status": "error", "action": "search", "message": "Could not locate search box"}
    
        # Clear and type
        element.clear()
        element.send_keys(query)
        element.send_keys(Keys.RETURN)
    
        # Check for CAPTCHA
        if SecurityFeatures.detect_captcha(self.driver):
            return {"status": "error", "action": "search", "message": "CAPTCHA detected"}
    
        # Wait for results to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div#search, div#rso"))
            )
        except TimeoutException:
            pass
    
        # Extract results (use a more specific selector for result links)
        links = self.driver.find_elements(By.CSS_SELECTOR, "h3 a, .yuRUbf a, .g a")
        results = []
        for link in links[:10]:
            href = link.get_attribute("href")
            text = (link.text or "").strip()
            if href and href.startswith("http"):
                item = {"link": href, "text": text}
                results.append(ContentHandling.postprocess_if_special(item, self.driver))
    
        return {"status": "success", "action": "search", "query": query, "results": results}
    
    def _handle_cookie_consent(self):
        """Accept common cookie consent banners."""
        consent_selectors = [
            "button[aria-label='Accept all']",
            "button:contains('Accept all')",
            "button:contains('Accept')",
            "button:contains('I agree')",
            "button[aria-label='Consent']",
            "#L2AGLb",  # Google's "Accept all" button ID
            "form button[jsname]",
        ]
        for selector in consent_selectors:
            try:
                button = WebDriverWait(self.driver, 3).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                button.click()
                time.sleep(1)  # allow overlay to disappear
                break
            except:
                continue

    def click(self, selector: str, wait_before_execution: Optional[float] = None) -> Dict[str, Any]:
        wait = self.default_wait if wait_before_execution is None else wait_before_execution
        result = self.clicker._perform_click(selector, wait)
        result["action"] = "click"
        return result

    def type(self, selector: str, text: str, clear_before: bool = True) -> Dict[str, Any]:
        result = self.typer.type_text(selector, text, clear_before=clear_before)
        result["action"] = "type"
        return result

    def scroll(self, mode: str = "by", **kwargs) -> Dict[str, Any]:
        if mode == "to":
            result = self.scroller.scroll_to(kwargs.get("x", 0), kwargs.get("y", 0), kwargs.get("smooth", False))
        elif mode == "element":
            result = self.scroller.scroll_element_into_view(kwargs.get("selector", ""), kwargs.get("position", "center"))
        elif mode == "direction":
            result = self.scroller.scroll_direction(kwargs.get("direction", "down"), kwargs.get("amount", 200))
        else:
            result = self.scroller.scroll_by(kwargs.get("dx", 0), kwargs.get("dy", 200), kwargs.get("smooth", False))
        result["action"] = "scroll"
        return result

    def copy(self, selector: str) -> Dict[str, Any]:
        return self.clipboard.copy(selector)

    def cut(self, selector: str) -> Dict[str, Any]:
        return self.clipboard.cut(selector)

    def paste(self, selector: str) -> Dict[str, Any]:
        return self.clipboard.paste(selector)

    def extract_page_content(self, preview_only: bool = False) -> Dict[str, Any]:
        current_url = self.driver.current_url
        title = self.driver.title
        text = self.driver.find_element(By.TAG_NAME, "body").text
        result = {"url": current_url, "title": title, "text": text[:1000] if preview_only else text}
        return ContentHandling.postprocess_if_special(result, self.driver)

    def execute_workflow(self, workflow_script: List[Dict[str, Any]]) -> Dict[str, Any]:
        steps = self.workflow.normalize(workflow_script)
        executed = []
        for step in steps:
            action = step["action"]
            params = step.get("params", {})
            if action == "navigate":
                result = self.navigate(params.get("url", ""))
            elif action == "search":
                result = self.search(params.get("query", ""), params.get("engine_url"))
            elif action == "click":
                result = self.click(params.get("selector", ""), params.get("wait_before_execution", self.default_wait))
            elif action == "type":
                result = self.type(params.get("selector", ""), params.get("text", ""), params.get("clear_before", True))
            elif action == "scroll":
                mode = params.get("mode", "by")
                result = self.scroll(mode=mode, **params)
            elif action == "copy":
                result = self.copy(params.get("selector", ""))
            elif action == "cut":
                result = self.cut(params.get("selector", ""))
            elif action == "paste":
                result = self.paste(params.get("selector", ""))
            elif action == "extract":
                result = {"status": "success", "action": "extract", "content": self.extract_page_content()}
            else:
                result = {"status": "error", "action": action, "message": f"Unsupported action '{action}'"}

            executed.append({"step": step, "result": result})
            if result.get("status") != "success":
                return {"status": "error", "executed": executed}

        return {"status": "success", "executed": executed}

    def perform_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Agent-standard task entrypoint."""
        task_type = (task_data or {}).get("task", "").lower()

        if task_type == "navigate":
            return self.navigate(task_data.get("url", ""))
        if task_type == "search":
            return self.search(task_data.get("query", ""), task_data.get("engine_url"))
        if task_type == "click":
            return self.click(task_data.get("selector", ""), task_data.get("wait_before_execution"))
        if task_type == "type":
            return self.type(task_data.get("selector", ""), task_data.get("text", ""), task_data.get("clear_before", True))
        if task_type == "scroll":
            return self.scroll(mode=task_data.get("mode", "by"), **task_data)
        if task_type == "copy":
            return self.copy(task_data.get("selector", ""))
        if task_type == "cut":
            return self.cut(task_data.get("selector", ""))
        if task_type == "paste":
            return self.paste(task_data.get("selector", ""))
        if task_type == "extract":
            return {"status": "success", "action": "extract", "content": self.extract_page_content()}

        if "workflow" in (task_data or {}):
            return self.execute_workflow(task_data["workflow"])

        if "query" in (task_data or {}):
            return self.search(task_data.get("query", ""))

        return {"status": "error", "message": "Unsupported browser task payload"}

    def close(self):
        if getattr(self, "driver", None):
            self.driver.quit()
            self.driver = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    print("\n=== Running Browser Agent ===\n")
    printer.status("TEST", "Browser Agent initialized", "info")
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.agent_factory import AgentFactory
    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    agent = BrowserAgent(shared_memory=shared_memory,agent_factory=agent_factory, config=None, driver=None)
    print(agent)

    print("\n* * * * * Phase 2 - Task * * * * *\n")
    data = {"task": "search", "query": "python selenium documentation"}

    task = agent.perform_task(task_data=data)
    print(task)

    print("\n* * * * * Phase 3 - Browsing * * * * *\n")
    
    # 1. Navigate to Google
    print("1. Navigating to Google...")
    result = agent.navigate("https://www.google.com")
    print(result)
    
    # 2. Search for something
    print("\n2. Searching for 'rocket lab neutron'...")
    result = agent.search("rocket lab neutron")
    print(result.get("results", [])[:3])  # show first 3 results
    
    # 3. Click on the first result (if any)
    if result.get("results"):
        # Wait for the result link to be clickable
        try:
            first_result_link = WebDriverWait(agent.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "h3 a, .yuRUbf a, .g a"))
            )
            click_result = agent.clicker._perform_click(selector="h3 a", wait_before_execution=1)
            print(click_result)
        except:
            print("Could not click the first result")
    
    # 4. Extract content from the new page
    print("\n4. Extracting page content...")
    content = agent.extract_page_content(preview_only=False)
    print(f"Title: {content.get('title')}")
    print(f"Text preview: {content.get('text', '')[:500]}...")

    print("\n=== Successfully ran the Browser Agent ===\n")
