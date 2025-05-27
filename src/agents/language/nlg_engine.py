"""
Core Function:
Performs Natural Language Generation — produces fluent and context-aware textual responses based on NLU output.

Responsibilities:
- Map intents + entities + context to a verbal response.
- Use templates or rule-based generation.
- Adjust tone, politeness, or verbosity if needed
- Support dynamic or neural generation methods.

Why it matters:
Even with perfect understanding, a language agent must communicate naturally and clearly to feel intelligent and helpful.
"""

import json, yaml
import random
import time
import re

from pathlib import Path
from typing import Dict, List, Optional, Any

from src.agents.language.utils.linguistic_frame import LinguisticFrame
from logs.logger import get_logger

logger = get_logger("NLG Engine")

CONFIG_PATH = "src/agents/language/configs/language_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config # ["nlg"]

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return

class NLGEngine:
    def __init__(self, config, llm_instance: Optional[Any] = None):
        self.coherence_checker = None
        self.config = config if config is not None else load_config()
        self.llm = llm_instance # uses my local llm called slai_lm

        templates_path = self.config.get("nlg", {}).get("templates_path")
        verbose_path = self.config.get("nlg", {}).get("verbose_phrases")

        self.templates = self._load_templates(templates_path)
        self.verbose_phrases = self._load_verbose_phrases(verbose_path)

        # Get style parameters from nlg config
        self.style = self.config.get("style", {}) # OR self.style = nlg_config.get("style", {})
        self.verbosity_mode = self.style.get("verbosity_mode", "default")

        logger.info(f"NLGEngine initialized with templates from {templates_path}")

    def _load_templates(self, path: str) -> Dict[str, List[str]]:
        """Load and validate response templates from JSON file, ensuring all entries are List[str]."""
        template_file_path = Path(path)
        fallback_default_responses = ["I'm not sure how to respond to that right now."]

        if not template_file_path.exists():
            logger.error(f"Template file missing at: {template_file_path}. Using minimal default.")
            return {"default": fallback_default_responses}

        try:
            with open(template_file_path, 'r', encoding='utf-8') as f:
                raw_templates = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {template_file_path}: {e}. Using minimal default.")
            return {"default": fallback_default_responses}
        except Exception as e:
            logger.error(f"Could not read template file {template_file_path}: {e}. Using minimal default.")
            return {"default": fallback_default_responses}


        if not isinstance(raw_templates, dict):
            logger.error(f"Templates file {template_file_path} should contain a JSON object. Using minimal default.")
            return {"default": fallback_default_responses}

        processed_templates: Dict[str, List[str]] = {}

        # Process the 'default' intent first to ensure it's a List[str]
        default_data = raw_templates.get("default")
        if isinstance(default_data, dict) and "responses" in default_data and isinstance(default_data["responses"], list):
            processed_templates["default"] = [str(r) for r in default_data["responses"] if isinstance(r, str) and r.strip()]
        elif isinstance(default_data, list):
            processed_templates["default"] = [str(r) for r in default_data if isinstance(r, str) and r.strip()]
        
        if not processed_templates.get("default"): # If still no valid default
            logger.warning(f"'default' intent in {template_file_path} is malformed or empty. Using fallback default responses.")
            processed_templates["default"] = fallback_default_responses


        # Process other intents
        for intent_key, data_value in raw_templates.items():
            if intent_key == "default": # Already processed
                continue

            current_intent_responses = []
            if isinstance(data_value, dict) and "responses" in data_value and isinstance(data_value["responses"], list):
                current_intent_responses = [str(r) for r in data_value["responses"] if isinstance(r, str) and r.strip()]
            elif isinstance(data_value, list): # If the intent directly maps to a list of strings
                current_intent_responses = [str(r) for r in data_value if isinstance(r, str) and r.strip()]
            
            if current_intent_responses:
                processed_templates[intent_key] = current_intent_responses
            else:
                logger.warning(f"Templates for intent '{intent_key}' in {template_file_path} are malformed or empty. "
                               f"This intent will use global default responses if called.")
                # No need to assign default here, as self.templates.get(intent, self.templates["default"]) will handle it in generate()

        return processed_templates

    def _load_verbose_phrases(self, path: str) -> Dict[str, List[str]]:
        """Load verbose phrases from JSON file."""
        verbose_file_path = Path(path)
        default_verbose = {"default": ["To elaborate further:", "Let me add that:", "Additionally:"]}
        if not verbose_file_path.exists():
            logger.warning(f"Verbose phrases file not found at {verbose_file_path}. Using minimal default.")
            return default_verbose
        try:
            with open(verbose_file_path, 'r', encoding='utf-8') as f:
                phrases = json.load(f)
            if not isinstance(phrases, dict) or not phrases.get("default"):
                logger.warning(f"Verbose phrases file {verbose_file_path} malformed. Using minimal default.")
                return default_verbose
            return phrases
        except Exception as e:
            logger.error(f"Error loading verbose phrases from {verbose_file_path}: {e}")
            return default_verbose


    def generate(self, frame: LinguisticFrame, context: Optional[Any] = None) -> str:
        """Generate a response given intent, entities, and context."""
        intent = frame.intent
        entities = frame.entities if frame.entities else {} # Ensure entities is a dict
        response_text = "Fallback response"
        if frame.intent == "greeting":
            response_text = "Hello!"

        if frame.intent == "farewell":
            response_text = "goodbye!"

        if frame.intent == "acknowledgement":
            response_text = "understood"
        # Attempt to get specific template list for the intent
        # Fallback to "default" templates if specific intent not found or has no valid responses
        template_list = self.templates.get(intent, self.templates["default"])

        if not template_list: # Should not happen if "default" is always initialized correctly
            logger.error(f"CRITICAL: No template list found for intent '{intent}' and no default templates available. THIS IS A BUG.")
            template_list = self.templates["default"] # ["I am truly unable to respond right now due to a configuration error."]

        # Select and format template
        try:
            chosen_template_str = random.choice(template_list)
            flat_entities = {k: str(v[0] if isinstance(v, list) else v) 
                            for k, v in entities.items()}

            # Flatten entities if they are lists for simple formatting
            # Ensure all entity values are strings for .format()
            flat_entities = {k: str(v[0] if isinstance(v, list) and v else v) for k, v in entities.items()}
            
            # Add current time if a {time} placeholder is in the template and not in entities
            if "{time}" in chosen_template_str and "time" not in flat_entities:
                flat_entities["time"] = time.strftime("%H:%M %Z") # e.g., 14:30 UTC

            response_text = chosen_template_str.format(**flat_entities)
        except (KeyError, IndexError) as e:
            logger.warning(f"Error formatting template for intent '{intent}' with entities {flat_entities}. "
                           f"Template: '{chosen_template_str}'. Error: {e}. Trying without entity formatting.")
            # Try to find a template for this intent that doesn't require formatting
            non_entity_templates = [t for t in template_list if '{' not in t]
            if non_entity_templates:
                response_text = random.choice(non_entity_templates)
            else: # If all templates for this intent require formatting, or no plain ones exist, use a plain default
                default_plain_templates = [t for t in self.templates.get("default", []) if '{' not in t]
                if default_plain_templates:
                    response_text = random.choice(default_plain_templates)
                else: # Ultimate fallback if even plain defaults are missing (should not happen)
                    response_text = "I seem to be having trouble with my response templates."
        
        # Neural fallback (if no good template response was generated and LLM is available)
        # For now, we assume template generation always produces something, even if it's a fallback template.
        # if not response_text and self.llm and hasattr(self.llm, 'generate'):
        #     logger.info(f"No template for intent '{intent}'. Attempting neural generation.")
        #     response_text = self._neural_generation(frame, context)
        
        if not response_text: # If after all attempts, response is still empty
             logger.error(f"NLGEngine failed to produce any response text for intent '{intent}'. Using hardcoded failsafe.")
             response_text = random.choice(self.templates.get("default", ["I'm unable to respond to that."]))

        # Adapt style (formality, verbosity)
        response_text = self._adapt_style(response_text)

        return response_text

    def _match_template(self, frame: LinguisticFrame) -> Optional[str]:
        """
        Match against handcrafted templates for common intents.
        This version directly uses the loaded self.templates.
        The original _match_template in input_file_0 seemed to be for a different structure.
        Here, we select a random template from the list associated with the intent.
        """
        intent_templates = self.templates.get(frame.intent)
        if intent_templates and isinstance(intent_templates, list) and intent_templates:
            return random.choice(intent_templates)
        return None # No template found for this specific intent directly

    def _neural_generation(self, frame: LinguisticFrame, context: Optional[Any]) -> str:
        """Generate using LLM with controlled prompting"""
        if not self.llm or not hasattr(self.llm, 'generate'):
            logger.error("LLM not available or does not have a 'generate' method for neural generation.")
            return random.choice(self.templates.get("default", ["I cannot generate a response right now."]))

        # Construct a prompt
        prompt_parts = []
        prompt_parts.append(f"Based on the following understanding of the user's request:")
        prompt_parts.append(f"- Intent: {frame.intent}")
        if frame.entities:
            prompt_parts.append(f"- Key Information (Entities): {json.dumps(frame.entities)}")
        if frame.sentiment != 0.0:
             prompt_parts.append(f"- User Sentiment: {'positive' if frame.sentiment > 0.3 else 'negative' if frame.sentiment < -0.3 else 'neutral'} (score: {frame.sentiment:.2f})")
        prompt_parts.append(f"- Expected Modality/Type of Response: {frame.modality}")

        if context:
            dialogue_summary = context.get_context_for_nlg()
        if context and hasattr(context, 'get_context_for_prompt'):
            dialogue_summary = context.get_context_for_prompt(include_history=True, history_messages_window=5) # Get last 5 messages
            if dialogue_summary:
                prompt_parts.append("\nRelevant conversation context:")
                prompt_parts.append(dialogue_summary)
        
        prompt_parts.append("\nGenerate an appropriate and helpful response:")
        prompt = "\n".join(prompt_parts)
        
        logger.info(f"Neural generation prompt: {prompt[:500]}...") # Log a snippet
        
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            return random.choice(self.templates.get("default", ["An error occurred while generating a response."]))


    def _adapt_style(self, text: str) -> str:
        """Adjust formality and verbosity based on self.style settings."""
        # Formality
        formality_level = self.style.get('formality', 0.5)
        # allow_slang = self.style.get('allow_slang', False)
        # ===== Formality Adjustments =====
        if formality_level > 0.7:  # Formal
            text = re.sub(r"\b(can't|don't|won't|isn't|aren't|I'm|you're|he's|she's|it's|we're|they're|let's|shouldn't|couldn't|wouldn't|didn't|haven't|hasn't|ain't)\b", 
                        lambda m: {
                            "can't": "cannot", "don't": "do not", "won't": "will not",
                            "isn't": "is not", "aren't": "are not", "I'm": "I am",
                            "you're": "you are", "he's": "he is", "she's": "she is",
                            "it's": "it is", "we're": "we are", "they're": "they are",
                            "let's": "let us", "shouldn't": "should not", 
                            "couldn't": "could not", "wouldn't": "would not",
                            "didn't": "did not", "haven't": "have not",
                            "hasn't": "has not", "ain't": "is not"
                        }[m.group(0).lower()], text, flags=re.IGNORECASE)
    
        elif formality_level < 0.3: # More informal
            text = re.sub(r"\b(cannot|do not|will not|is not|are not|I am|you are|he is|she is|it is|we are|they are|let us)\b",
                        lambda m: {
                            "cannot": "can't", "do not": "don't", "will not": "won't",
                            "is not": "isn't", "are not": "aren't", "I am": "I'm",
                            "you are": "you're", "he is": "he's", "she is": "she's",
                            "it is": "it's", "we are": "we're", "they are": "they're",
                            "let us": "let's"
                        # }[m.group(0).lower()], text, flags=re.IGNORECASE)
                        }.get(m.group(0).lower(), m.group(0)), text, flags=re.IGNORECASE)
    
            # Informal slang
            text = re.sub(r"\bgoing to\b", "gonna", text, flags=re.IGNORECASE)
            text = re.sub(r"\bwant to\b", "wanna", text, flags=re.IGNORECASE)
            text = re.sub(r"\bgot to\b", "gotta", text, flags=re.IGNORECASE)
            text = re.sub(r"\bkind of\b", "kinda", text, flags=re.IGNORECASE)
            text = re.sub(r"\blots of\b", "lotsa", text, flags=re.IGNORECASE)
    
            # Add informal phrases
            # if random.random() < 0.3:
            #     text = f"{random.choice(self.style.get('informal_phrases', []))} {text}"
            if self.style.get('informal_phrases') and random.random() < 0.3:
                text = f"{random.choice(self.style['informal_phrases'])} {text}"
    
        # ===== Verbosity Adjustments =====
        verbosity_level = self.style.get('verbosity', 1.0) # Default to normal verbosity
        if verbosity_level < 0.7: # More concise
            truncation_length = self.style.get('truncation_length', 25) # Default 25 words
            words = text.split()
            if len(words) > truncation_length:
                text = ' '.join(words[:truncation_length]) + '...'
        elif verbosity_level > 1.3: # More verbose
            if self.verbose_phrases: # Check if loaded
                # Use verbosity_mode to pick a pool, fallback to "default" pool
                phrase_pool = self.verbose_phrases.get(self.verbosity_mode, self.verbose_phrases.get("default", []))
                if phrase_pool and random.random() < 0.4: # Only add if pool is not empty
                    prepend = random.choice(phrase_pool)
                    text = f"{prepend} {text}"

        return text.strip()


class NLGFillingError(Exception):
    """Custom exception for template filling failures"""

class NLGValidationError(Exception):
    """Custom exception for response validation failures"""

class TemplateNotFoundError(Exception):
    """Custom exception for missing templates"""


if __name__ == "__main__":
    print("\n=== Running NLG Engine ===\n")
    config = load_config()

    engine = NLGEngine(config)

    print(engine)
    print("\n=== Successfully Ran NLG Engine ===\n")

if __name__ == "__main__":
    print("\n=== Testing NLG Engine ===\n")

    from src.agents.language.utils.linguistic_frame import SpeechActType

    # Load configuration
    config = load_config()
    
    # Initialize engine
    engine = NLGEngine(config)
    
    # Print engine config
    print("Loaded Configuration:")
    print(f"- Templates Path: {engine.config.get('templates_path')}")
    print(f"- Verbosity Mode: {engine.verbosity_mode}")
    print(f"- Style Settings: {engine.style}\n")

    # Test cases
    test_frames = [
        LinguisticFrame(
            intent="weather_inquiry",
            entities={"location": "Paris", "unit": "Celsius"},
            sentiment=0.4,
            modality="informative",
            confidence=0.9,
            act_type=SpeechActType.DIRECTIVE  # Add proper act_type
        ),
        LinguisticFrame(
            intent="joke_request",
            entities={"topic": "programmers"},
            sentiment=0.7,
            modality="entertaining",
            confidence=0.8,
            act_type=SpeechActType.EXPRESSIVE
        ),
        LinguisticFrame(
            intent="unknown_intent",
            entities={},
            sentiment=0.0,
            modality="neutral",
            confidence=0.2,
            act_type=SpeechActType.ASSERTIVE
        )
    ]

    # Run test cases
    for i, frame in enumerate(test_frames, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Intent: {frame.intent}")
        print(f"Entities: {frame.entities}")
        response = engine.generate(frame)
        print(f"\nGenerated Response ({engine.verbosity_mode}):")
        print(f"> {response}")

    # Test style variations
    print("\n--- Style Variations ---")
    sample_text = "I can't tell you how excited we are about this new feature! It's going to really improve user experience."
    
    # Formal
    engine.style['formality'] = 0.9
    print("\nFormal Version:")
    print(engine._adapt_style(sample_text))
    
    # Informal
    engine.style['formality'] = 0.1
    print("\nInformal Version:")
    print(engine._adapt_style(sample_text))

    print("\n=== NLG Engine Tests Complete ===\n")
