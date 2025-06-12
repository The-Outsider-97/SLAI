
import pyperclip
import traceback

from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium import webdriver

from src.agents.browser.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("CopyCutPaste")
printer = PrettyPrinter

class DoCopyCutPaste:
    def __init__(self, driver):
        self.config = load_global_config()
        self.scroll_config = get_config_section("do_ccp")
        self.driver = driver
        driver = webdriver.Chrome()

        logger.info(f"Web Copy, Cut & Paste functionality initiated.")

    def _get_element_text(self, element) -> str:
        try:
            # Prefer textContent, fallback to 'value' for inputs
            text = element.text.strip()
            if not text:
                text = element.get_attribute("value") or ""
            if not text:
                text = element.get_attribute("placeholder") or ""
            return text.strip()
        except Exception as e:
            logger.warning(f"Text extraction failed: {str(e)}")
            return ""

    def _extract_context_metadata(self, element) -> dict:
        try:
            role = element.get_attribute("role")
            tag = element.tag_name
            name = element.get_attribute("name") or element.get_attribute("id")
            placeholder = element.get_attribute("placeholder")
            aria_label = element.get_attribute("aria-label")
            class_info = element.get_attribute("class")

            return {
                "tag": tag,
                "role": role,
                "name": name,
                "placeholder": placeholder,
                "aria_label": aria_label,
                "class": class_info,
                "outer_html": element.get_attribute("outerHTML")[:400]
            }
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}

    def _analyze_text(self, text: str) -> dict:
        try:
            tokens = self.nlp.process_text(text)
            summary = {
                "length": len(text),
                "token_count": len(tokens),
                "lemmas": list({t.lemma for t in tokens}),
                "pos_distribution": {t.pos: sum(1 for x in tokens if x.pos == t.pos) for t in tokens}
            }
            return {"tokens": tokens, "summary": summary}
        except Exception as e:
            logger.error(f"NLP analysis failed: {e}")
            return {"tokens": [], "summary": {}}

    def _nlu_analysis(self, text: str) -> dict:
        try:
            entities = self.nlu.phonetic_candidates(text)
            similar_words = [w for w in entities if self.nlu.semantic_similarity(w, text) > 0.7]
            return {
                "phonetic_matches": entities,
                "semantic_matches": similar_words,
            }
        except Exception as e:
            logger.error(f"NLU failed: {e}")
            return {"phonetic_matches": [], "semantic_matches": []}

    def _fallback_llm_analysis(self, text: str) -> dict:
        """Reserved hook for ChatGPT or Gemini fallback"""
        return {
            "llm_fallback": "LLM disabled (NLP + NLU primary)",
            "suggested_action": "None"
        }

    def copy(self, selector: str) -> dict:
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            text = self._get_element_text(element)
            pyperclip.copy(text)

            nlp_data = self._analyze_text(text)
            nlu_data = self._nlu_analysis(text)
            metadata = self._extract_context_metadata(element)

            logger.info(f"[COPY] '{text[:50]}' | NLP: {nlp_data['summary']}")

            return {
                "status": "success",
                "action": "copy",
                "text": text,
                "metadata": metadata,
                "nlp": nlp_data,
                "nlu": nlu_data
            }

        except NoSuchElementException:
            return {"status": "error", "message": f"Selector '{selector}' not found"}
        except WebDriverException as we:
            return {"status": "error", "message": f"Selenium failure: {we}"}
        except Exception as e:
            return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

    def cut(self, selector: str) -> dict:
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            text = self._get_element_text(element)
            pyperclip.copy(text)
            element.clear()

            nlp_data = self._analyze_text(text)
            nlu_data = self._nlu_analysis(text)
            metadata = self._extract_context_metadata(element)

            logger.info(f"[CUT] '{text[:50]}' | NLP: {nlp_data['summary']}")

            return {
                "status": "success",
                "action": "cut",
                "text": text,
                "metadata": metadata,
                "nlp": nlp_data,
                "nlu": nlu_data
            }

        except Exception as e:
            return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

    def paste(self, selector: str) -> dict:
        try:
            paste_text = pyperclip.paste()
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            element.clear()
            element.send_keys(paste_text)

            nlp_data = self._analyze_text(paste_text)
            nlu_data = self._nlu_analysis(paste_text)
            metadata = self._extract_context_metadata(element)

            logger.info(f"[PASTE] '{paste_text[:50]}' | NLP: {nlp_data['summary']}")

            return {
                "status": "success",
                "action": "paste",
                "text": paste_text,
                "metadata": metadata,
                "nlp": nlp_data,
                "nlu": nlu_data
            }

        except Exception as e:
            return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        

if __name__ == "__main__":
    print("\n=== Running Web navigation functionality Test ===\n")
    printer.status("Init", "Web navigation functionality initialized", "success")
    driver = webdriver.Chrome()

    action = DoCopyCutPaste(driver=driver)
    print(action)
    print("\n* * * * * Phase 2 * * * * *\n")
    print("\n=== Successfully Ran Web navigation functionality ===\n")
