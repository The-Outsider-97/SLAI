
import os
import openai
import google.generativeai as genai

from selenium import webdriver
from dotenv import load_dotenv
load_dotenv()

from src.agents.browser.utils.config_loader import load_global_config, get_config_section
from src.agents.language.nlu_engine import Wordlist
from src.agents.language.nlp_engine import NLPEngine
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Type")
printer = PrettyPrinter

class DoType:
    def __init__(self, driver):
        self.config = load_global_config()
        self.scroll_config = get_config_section('do_type')
        self.driver = driver
        driver = webdriver.Chrome()

        self.wordlist = Wordlist()
        self.nlp = NLPEngine()

        # LLM fallback setup
        openai.api_key = os.getenv("OPENAI_API_KEY")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        logger.info(f"Browser typing functionality with NLU/NLP initiated.")

    def type_text(self, selector: str, raw_input: str, clear_before: bool = True) -> dict:
        """Type intelligently using SLAI NLP/NU + fallback LLMs"""
        try:
            # --- Step 1: Preprocess Input Locally ---
            tokens = self.nlp.process_text(raw_input)
            keywords = [t.lemma for t in tokens if not t.is_stop and t.pos in {"NOUN", "VERB", "ADJ"}]
            logger.info(f"Extracted keywords: {keywords}")

            # --- Step 2: Resolve Intent/Entities (NLU) ---
            intent, entities = self._resolve_intent_and_entities(keywords)

            if not intent:
                logger.warning("Local intent resolution failed, using fallback LLM.")
                intent, entities = self._fallback_to_llm(raw_input)

            if not intent:
                return {"status": "error", "message": "Unable to resolve input meaning"}

            # --- Step 3: Type into the browser ---
            formatted = self._format_input(intent, entities, raw_input)
            element = self.driver.find_element("css selector", selector)
            if clear_before:
                element.clear()
            element.send_keys(formatted)

            logger.info(f"Typed intent-driven text: {formatted}")
            return {"status": "success", "message": f"Typed: {formatted}"}

        except Exception as e:
            logger.error(f"Typing failed: {e}")
            return {"status": "error", "message": str(e)}

    def _resolve_intent_and_entities(self, keywords: list) -> tuple[str, list]:
        """Local resolution of intent and semantic features"""
        try:
            if not keywords:
                return None, []

            main = keywords[0]
            related = sorted(
                [(kw, self.wordlist.semantic_similarity(main, kw)) for kw in keywords[1:]],
                key=lambda x: x[1], reverse=True
            )
            intent = main
            entities = [r[0] for r in related if r[1] > 0.5]
            logger.debug(f"Inferred intent: {intent}, entities: {entities}")
            return intent, entities
        except Exception as e:
            logger.warning(f"Intent resolution failed: {e}")
            return None, []

    def _fallback_to_llm(self, prompt: str) -> tuple[str, list]:
        """Fallback to ChatGPT then Gemini for interpretation"""
        printer.status("INIT", "Fallback to LLM", "warning")

        try:
            logger.info("Trying OpenAI fallback...")
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract the intent and key entities from the user input."},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content
            return self._parse_llm_response(text)

        except Exception as openai_err:
            logger.warning(f"OpenAI fallback failed: {openai_err}")

            try:
                logger.info("Trying Gemini fallback...")
                model = genai.GenerativeModel("gemini-pro")
                chat = model.start_chat()
                reply = chat.send_message(f"Extract the main intent and key terms from: {prompt}")
                return self._parse_llm_response(reply.text)
            except Exception as gemini_err:
                logger.error(f"Gemini fallback also failed: {gemini_err}")
                return None, []

    def _parse_llm_response(self, response: str) -> tuple[str, list]:
        """Basic parser for LLM output like: 'Intent: Fill form. Entities: name, address'"""
        try:
            intent_line = next((line for line in response.splitlines() if "intent" in line.lower()), "")
            entity_line = next((line for line in response.splitlines() if "entities" in line.lower()), "")
            intent = intent_line.split(":")[-1].strip()
            entities = [e.strip() for e in entity_line.split(":")[-1].split(",")]
            return intent, entities
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {response} ({e})")
            return None, []

    def _format_input(self, intent: str, entities: list, original: str) -> str:
        """Optionally enrich/transform input based on context"""
        return f"{intent}: {' '.join(entities)}" if entities else original


if __name__ == "__main__":
    print("\n=== Running Browser typing functionality Test ===\n")
    printer.status("Init", "Browser typing functionality initialized", "success")

    class MockElement:
        def __init__(self):
            self.buffer = ""

        def clear(self):
            self.buffer = ""
            print("[MOCK] Element cleared.")

        def send_keys(self, text):
            self.buffer += text
            print(f"[MOCK] Typed into element: {text}")

    class MockDriver:
        def find_element(self, by, selector):
            print(f"[MOCK] Finding element by {by} with selector '{selector}'")
            return MockElement()

    print("=== Interactive DoType Tester ===")
    print("Type 'exit' to quit.\n")

    driver = MockDriver()
    typer = DoType(driver)

    while True:
        selector = input("Enter CSS selector: ").strip()
        if selector.lower() == "exit":
            break

        raw_input = input("Enter text to process: ").strip()
        if raw_input.lower() == "exit":
            break

        result = typer.type_text(selector, raw_input)
        printer.pretty("Result:", result, "success")
        print("-" * 50)

    print("\n=== Successfully Ran Browser typing functionality ===\n")
