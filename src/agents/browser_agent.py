import os, sys
import time
import random
import logging
import requests
# import pytesseract
import numpy as np

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from robotexclusionrulesparser import RobotExclusionRulesParser

from src.agents.reasoning_agent import ReasoningAgent
from src.agents.language_agent import LanguageAgent, DialogueContext
from src.agents.learning_agent import LearningAgent
from src.agents.base_agent import BaseAgent
from src.agents.browser.security import SecurityFeatures, exponential_backoff
from src.agents.browser.workflow import WorkFlow
from src.agents.browser.utils import Utility

logging.getLogger(__name__)

# --------------------------
# Core Configuration
# --------------------------
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
SEARCH_DELAY = (0.1, 0.02)
WINDOW_SIZE = (random.randint(1200, 1400), random.randint(800, 1000))
SNAPSHOT_DIR = os.path.join("debug", "dom_snapshots")
CAPTCHA_LOG_DIR = os.path.join("debug", "captcha_logs")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(CAPTCHA_LOG_DIR, exist_ok=True)


# --------------------------
# Browser Setup & Utilities
# --------------------------

class BrowserAgent(BaseAgent):
    def __init__(self, shared_memory, agent_factory, slai_lm, config=None):
        super().__init__(shared_memory, agent_factory, config)
        # Initialize core browser components
        self.slai_lm = slai_lm
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.driver = self._init_browser()
        self.robots_parser = RobotExclusionRulesParser()

        # Initialize integrated agents
        shared_llm = slai_lm(shared_memory=self.shared_memory, agent_factory=self.agent_factory)
        self.agent_factory.shared_resources["llm"] = shared_llm
        self._init_agents()
        AGENT_CONFIGS = {
            "reasoning": {"init_args": {"llm": shared_llm}},
            "language": {},
            "learning": {}
        }
        for agent_name, cfg in AGENT_CONFIGS.items():
            setattr(self, agent_name, self.agent_factory.create(agent_name, config=cfg))

        # State tracking
        self.session_context = DialogueContext()
        self.knowledge_cache = {}
        self.last_snapshot = None

    def _init_browser(self):
        options = webdriver.ChromeOptions()
        options.add_argument(f"user-agent={USER_AGENT}")
        options.add_argument(f"window-size={WINDOW_SIZE[0]},{WINDOW_SIZE[1]}")
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        # options.add_argument("--remote-debugging-port=9222")  # Uncomment if needed
        return webdriver.Chrome(options=options)

    def navigate(self, url):
        self._capture_dom_snapshot("pre_navigation")
        self.driver.get(url)
        self._capture_dom_snapshot("post_navigation")
        return self._get_page_state()

    # Integrated Processing Pipeline --------------------------------
    def process_page(self, query):
        try:
            self.session_context.last_query = query
            self._capture_dom_snapshot("pre_processing")

            # 1. Language Understanding
            parsed_query = self.nlp_processor.parse(query)
            # 2. Execute Browser Actions
            search_results = self._perform_search(parsed_query)
            # 3. Reason about Content
            analyzed_content = self.reasoning.forward_chaining(
                self._content_to_facts(search_results)
            )
            # 4. Learn from Interaction
            reward = self._calculate_relevance(parsed_query, analyzed_content)
            self.learner.record_interaction(parsed_query, analyzed_content, reward)
            self._capture_dom_snapshot("post_processing")
            return analyzed_content

        except Exception as e:
            self.logger.error(f"Processing failed at {self.driver.current_url}", exc_info=True)
            raise

    def scrape(self, query: str) -> list:
        """Returns raw results with no reasoning/learning for fast data access"""
        parsed_query = self.nlp_processor.parse(query)
        return self._perform_search(parsed_query)

    # Debugging & Diagnostics ----------------------------------------------------
    def _capture_dom_snapshot(self, phase):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{phase}_{timestamp}.html"
        path = os.path.join(SNAPSHOT_DIR, filename)
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            self.last_snapshot = path
        except Exception as e:
            self.logger.error(f"Failed to capture DOM snapshot: {str(e)}")

    def _log_captcha_fallback(self, image_path):
        error_entry = {
            "timestamp": time.time(),
            "url": self.driver.current_url,
            "image_path": image_path,
            "page_text_snippet": self.driver.page_source[:500],
            "context": self._get_page_state(),
            "captcha_type": "image" if os.path.exists(image_path) else "text-based"
        }
        self.shared_memory.set("captcha_errors", 
            self.shared_memory.get("captcha_errors", []) + [error_entry]
        )

    # Agent Integration Helpers -------------------------------------
    def _init_agents(self):
        """Initialize all integrated agents in proper order"""
        shared_llm = self.agent_factory.shared_resources["llm"]
        
        self.reasoning = self.agent_factory.create('reasoning', config={
            "init_args": {"llm": shared_llm}
        })
        self.nlp_processor = self.agent_factory.create('language', config={})
        self.learner = self.agent_factory.create('learning', config={})

    def _content_to_facts(self, content):
        return [(element['text'], 'RELATED_TO', self.session_context.last_query)
                for element in content if element['relevance'] > 0.5]

    def _safe_navigate(self, func, *args, retries=3):
        for attempt in range(retries):
            try:
                return func(*args)
            except Exception as e:
                self.logger.warning(
                    f"Navigation failed attempt {attempt+1}: {str(e)}\n"
                    f"Current URL: {self.driver.current_url}"
                )
                exponential_backoff(attempt)
        raise RuntimeError(f"Navigation failed after {retries} retries")

    def _create_shared_memory(self):
        return {
            'browser_state':self._get_page_state(),
            'knowledge_base': lambda: self.knowledge_cache,
            'user_preferences': self._load_user_profile()
        }

    def _content_to_facts(self, content):
        return [(element['text'], 'RELATED_TO', self.session_context.last_query)
                for element in content if element['relevance'] > 0.5]

    def _calculate_relevance(self, query, content):
        return self.nlp_processor.semantic_similarity(
            query, 
            ' '.join([c['key_terms'] for c in content])
        )

    # Enhanced Navigation -------------------------------------------
    def _perform_search(self, parsed_query):
        # Use language agent to expand query
        expanded_terms = self.nlp_processor.expand_query(
            parsed_query['processed_text']
        )
        
        # Execute browser actions
        self._type_search(expanded_terms)
        results = self._process_results()
        
        # Use reasoning to filter results
        return self.reasoning.probabilistic_query(
            results,
            evidence=parsed_query['entities']
        )

    def _type_search(self, text):
        search_box = self.driver.find_element(By.NAME, 'q')
        self._human_like_type(search_box, text)
        search_box.send_keys(Keys.RETURN)
        time.sleep(random.uniform(1, 2))
        if self._detect_captcha():
            self._handle_captcha()

    # Security & Learning Integration -------------------------------
    def _handle_captcha(self):
        if self._detect_captcha():
            solution = self.learner.solve_challenge(
                challenge_type='CAPTCHA',
                context=self._get_page_state()
            )
            self._submit_captcha_solution(solution)

    def _adapt_to_blocks(self):
        if self.learner.should_adapt():
            new_strategy = self.learner.get_adaptation_plan()
            self._apply_navigation_strategy(new_strategy)

    # Content Processing --------------------------------------------
    def _process_results(self):
        return [
            self._analyze_element(element)
            for element in self.driver.find_elements(By.XPATH, "//a[@href]")
            if self._is_valid_link(element)
        ]

    def _analyze_element(self, element):
        from src.agents.browser.content import ContentHandling
        text = element.text
        link = element.get_attribute('href')
        content_handler = ContentHandling()
        processed_result = content_handler._postprocess_if_special({"link": link}, self.driver)
        full_text = processed_result.get("text", text)
        return {
            'text': full_text,
            'link': link,
            'key_terms': self.nlp_processor.extract_key_phrases(full_text),
            'relevance': self.reasoning.neuro_symbolic_verify(
                ('SearchResult', 'matches', full_text)
            )
        }
    
    def _get_page_state(self):
        return {
            'url': self.driver.current_url,
            'title': self.driver.title,
            'content': self.driver.page_source[:2000]
        }
    
    def _human_like_type(self, element, text):
        Utility.human_type(element, text)
    
    def _detect_captcha(self):
        return SecurityFeatures.detect_captcha(self.driver)
    
    def _is_valid_link(self, element):
        href = element.get_attribute('href')
        return href and self.robots_parser.is_allowed(USER_AGENT, href)

    def __del__(self):
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
        except Exception:
            pass

# --------------------------
# Execution
# --------------------------
if __name__ == "__main__":
    from models.slai_lm import SLAILM

    try:
        class MinimalFactory:
            def create(self, agent_type, config=None):
                if agent_type == 'language':
                    return LanguageAgent(shared_memory={}, agent_factory=self)
                elif agent_type == 'reasoning':
                    return ReasoningAgent(shared_memory={}, agent_factory=self, tuple_key='demo')
                elif agent_type == 'learning':
                    return LearningAgent(shared_memory={}, agent_factory=self, slai_lm=self.shared_resources["llm"], safety_agent=None, env=None)
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")

            def __init__(self):
                self.shared_resources = {}

        factory = MinimalFactory()
        shared_memory = {}

        # Instantiate SLAILM once
        slai_lm = SLAILM(shared_memory=shared_memory, agent_factory=factory)
        factory.shared_resources["llm"] = slai_lm

        agent = BrowserAgent(shared_memory=shared_memory, agent_factory=factory, slai_lm=SLAILM)
        results = agent.process_page("SpaceX Raptor Engine 3 research papers")
        print("=== RESULTS ===")
        print(results)
    except Exception as e:
        agent.logger.critical(f"Critical failure: {str(e)}")
        agent.driver.save_screenshot("final_error_state.png")
