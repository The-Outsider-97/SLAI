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

from src.agents.language.utils.config_loader import load_global_config, get_config_section
from src.agents.language.utils.linguistic_frame import LinguisticFrame, SpeechActType
from src.agents.language.utils.language_error import NLGFillingError, NLGValidationError, TemplateNotFoundError, NLGGenerationError
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("NLG Engine")
printer = PrettyPrinter

class NLGEngine:
    def __init__(self):
        self.config = load_global_config()
        self.wordlist_path = self.config.get('main_wordlist_path')
        self.modality_markers = self.config.get('modality_markers_path')
        
        self.nlg_config = get_config_section('nlg')
        self.coherence_checker = self.nlg_config.get('coherence_checker')
        self._templates_path = self.nlg_config.get("templates_path")
        self.verbose_path = self.nlg_config.get("verbose_phrases")

        self.neural_config = get_config_section("neural_generation")
        self.max_retries = self.neural_config.get("max_retries")
        self.fallback_after_retries = self.neural_config.get("fallback_after_retries")
        self.max_tokens = self.neural_config.get("max_tokens")
        self.temperature = self.neural_config.get("temperature")

        # Get style parameters from nlg config
        self.style = self.nlg_config.get("style", {
            'formality', 'verbosity', 'verbosity_mode',
            'truncation_length', 'allow_slang'
        })

        self.templates = self._load_templates(self.templates_path)
        self.verbose_phrases = self._load_verbose_phrases(self.verbose_path)
        self.modality_markers = self._load_modality_markers(self.config.get('modality_markers_path'))

        logger.info(f"NLG Engine succesfully initialized")

    @property
    def templates_path(self):
        return self._templates_path
    
    @templates_path.setter
    def templates_path(self, value):
        self._templates_path = value

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

        # Process all intents
        for intent_key, data_value in raw_templates.items():
            # Store dict entries as-is
            if isinstance(data_value, dict):
                processed_templates[intent_key] = data_value
            # Convert lists to {responses: [...]} format
            elif isinstance(data_value, list):
                processed_templates[intent_key] = {"responses": data_value}
        
        # Ensure default exists
        if "default" not in processed_templates:
            processed_templates["default"] = {"responses": fallback_default_responses}
        
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
        
    def _load_modality_markers(self, path: str) -> Dict[str, List[str]]:
        """Load modality markers from JSON file"""
        if not path:
            logger.error("Modality markers path not provided in config")
            return {}
            
        file_path = Path(path)
        if not file_path.exists():
            logger.error(f"Modality markers file missing at: {file_path}")
            return {}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading modality markers: {e}")
            return {}

    def generate(self, frame: LinguisticFrame, context: Optional[Any] = None) -> str:
        """Generate a response given intent, entities, and context."""
        intent = frame.intent
        entities = frame.entities if frame.entities else {}
        sentiment = frame.sentiment
        modality = frame.modality
        confidence = frame.confidence
        act_type = frame.act_type

        # If intent is empty but we have text context, determine intent via triggers
        if not frame.intent and context and isinstance(context, str):
            frame.intent = self._match_intent_by_trigger(context)
        
        intent = frame.intent
        
        # Get templates for intent
        template_entry = self.templates.get(intent, self.templates["default"])
        template_list = template_entry.get("responses", [])
        if not template_list:
            logger.error(f"No templates available for intent '{intent}'")
            return "I'm unable to respond to that right now."
    
        # Add fallback entities for common cases
        flat_entities = self._get_flattened_entities(entities)
        if intent == "weather_inquiry":
            flat_entities.setdefault("weather_status", "unknown conditions")
            flat_entities.setdefault("temperature", "N/A")
        
        # Add modality marker if needed
        if modality and modality != "epistemic":
            modality_markers = self._get_modality_markers(modality)
            if modality_markers:
                flat_entities["modality_marker"] = random.choice(modality_markers)
    
        # Add confidence qualifier
        confidence_qualifier = self._get_confidence_qualifier(confidence)
        if confidence_qualifier:
            flat_entities["confidence_qualifier"] = confidence_qualifier
    
        # Add sentiment marker
        sentiment_marker = self._get_sentiment_marker(sentiment)
        if sentiment_marker:
            flat_entities["sentiment_marker"] = sentiment_marker
    
        # Add speech act marker
        act_type_marker = self._get_act_type_marker(act_type)
        if act_type_marker:
            flat_entities["act_type_marker"] = act_type_marker
    
        # Try all templates until one works
        shuffled_templates = random.sample(template_list, len(template_list))
        response_text = None
        
        for template_str in shuffled_templates:
            try:
                # Add current time if needed
                if "{time}" in template_str and "time" not in flat_entities:
                    flat_entities["time"] = time.strftime("%H:%M %Z")
                    
                response_text = template_str.format(**flat_entities)
                break
            except (KeyError, IndexError) as e:
                missing = self._find_missing_placeholders(template_str, flat_entities)
                logger.debug(f"Template failed: {template_str} | Missing: {missing} | Error: {e}")
    
        # Final fallback if all templates fail
        if not response_text:
            logger.warning(f"All templates failed for intent '{intent}'")
            plain_templates = [t for t in template_list if '{' not in t]
            response_text = random.choice(plain_templates) if plain_templates else \
                random.choice(self.templates["default"])
    
        # Adapt style before returning
        return self._adapt_style(response_text)
    
    def _match_intent_by_trigger(self, text: str) -> str:
        """Match user text to intent using trigger phrases"""
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        best_score = -1
        best_intent = "default"
    
        for intent_key, template_data in self.templates.items():
            triggers = template_data.get("triggers", [])
            if intent_key == "default":
                continue
                
            triggers = []
            if isinstance(template_data, dict) and "triggers" in template_data:
                triggers = template_data["triggers"]
            elif isinstance(template_data, list) and intent_key in self.templates and "triggers" in self.templates.get(intent_key, {}):
                triggers = self.templates[intent_key].get("triggers", [])
            
            for trigger in triggers:
                # Create a pattern that matches the entire trigger phrase
                pattern = r'\b' + re.escape(trigger.lower()) + r'\b'
                if re.search(pattern, text_clean):
                    # Calculate match quality score (longer matches are better)
                    match_length = len(trigger.split())  # Count words in trigger
                    if match_length > best_score:
                        best_score = match_length
                        best_intent = intent_key
                        
        return best_intent
    
    def _get_flattened_entities(self, entities: dict) -> dict:
        """Flatten entities ensuring all values are strings"""
        flat_entities = {}
        for k, v in entities.items():
            if isinstance(v, list) and v:
                flat_entities[k] = str(v[0])
            else:
                flat_entities[k] = str(v)
        return flat_entities
    
    # New helper methods ========================================
    def _get_modality_markers(self, modality: str) -> List[str]:
        """Get appropriate modality markers based on modality type"""
        modality_config = self.modality_markers
        return modality_config.get(modality, [])
    
    def _get_confidence_qualifier(self, confidence: float) -> str:
        """Get confidence qualifier based on confidence score"""
        if confidence < 0.4:
            return random.choice(["I'm not entirely sure, but ", "I might be mistaken, but "])
        elif confidence < 0.7:
            return random.choice(["I believe ", "It seems that "])
        return ""
    
    def _get_sentiment_marker(self, sentiment: float) -> str:
        """Get sentiment marker based on sentiment score"""
        if sentiment < -0.5:
            return random.choice(["I'm sorry to hear that. ", "That sounds difficult. "])
        elif sentiment > 0.5:
            return random.choice(["Great! ", "I'm glad to hear that! "])
        return ""
    
    def _get_act_type_marker(self, act_type: SpeechActType) -> str:
        """Get appropriate marker based on speech act type"""
        markers = {
            SpeechActType.ASSERTIVE: "",
            SpeechActType.DIRECTIVE: "Please ",
            SpeechActType.COMMISSIVE: "I will ",
            SpeechActType.EXPRESSIVE: random.choice(["I appreciate that. ", "Thank you. "]),
            SpeechActType.DECLARATION: "I declare that "
        }
        return markers.get(act_type, "")
    
    def _find_missing_placeholders(self, template: str, entities: dict) -> list:
        """Identify missing placeholders in template"""
        placeholders = re.findall(r'\{([^}]+)\}', template)
        return [p for p in placeholders if p not in entities]

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
        """Generate using controlled prompting with proper neural implementation"""
        printer.status("INIT", "Neural generation initialized", "info")
        
        # Build dynamic prompt
        prompt = self._build_neural_prompt(frame, context)
        logger.info(f"Neural prompt ({len(prompt)} chars):\n{prompt[:300]}...")
        
        # Generate response using neural model
        try:
            response = self._call_neural_model(prompt)
            if self._validate_response(response):
                return response
            else:
                logger.warning("Neural generation failed validation")
        except Exception as e:
            logger.error(f"Neural generation error: {e}")
        
        # Fallback to template-based generation
        printer.status("FALLBACK", "Reverting to template-based generation", "warning")
        return self.generate(frame)
    
    def _build_neural_prompt(self, frame: LinguisticFrame, context: Optional[Any]) -> str:
        """Construct context-aware prompt with linguistic features"""
        prompt_parts = [
            "# Role: Helpful AI Assistant",
            "## Task: Generate natural response based on:"
        ]
        
        # Core frame information
        prompt_parts.append(f"- Intent: {frame.intent}")
        prompt_parts.append(f"- Speech Act Type: {frame.act_type.value}")
        
        if frame.illocutionary_force:
            prompt_parts.append(f"- Illocutionary Force: {frame.illocutionary_force}")
        if frame.propositional_content:
            prompt_parts.append(f"- Content Focus: {frame.propositional_content}")
        
        # Entities and sentiment
        if frame.entities:
            prompt_parts.append(f"- Key Entities: {json.dumps(frame.entities)}")
        if abs(frame.sentiment) > 0.1:
            sentiment_label = "positive" if frame.sentiment > 0 else "negative"
            prompt_parts.append(f"- User Sentiment: {sentiment_label} (confidence: {abs(frame.sentiment):.1f})")
        
        # Context integration
        if context:
            if hasattr(context, 'get_relevant_context'):
                ctx = context.get_relevant_context(
                    history_window=3,
                    include_entities=True,
                    sentiment_threshold=0.3
                )
                if ctx:
                    prompt_parts.append("\n## Conversation Context:")
                    prompt_parts.append(ctx)
            else:
                prompt_parts.append(f"\n## Context: {context}")
        
        # Generation instructions
        prompt_parts.extend([
            "\n## Response Requirements:",
            f"- Modality: {frame.modality}",
            f"- Confidence: {frame.confidence:.1f}",
            "- Be coherent with conversation history",
            "- Maintain appropriate tone and style"
        ])
        
        # Add style preferences
        if self.style:
            style_notes = []
            if self.style.get('formality', 0.5) > 0.7:
                style_notes.append("Use formal language")
            elif self.style.get('formality') < 0.3:
                style_notes.append("Use casual/informal language")
            
            if style_notes:
                prompt_parts.append("- Style: " + "; ".join(style_notes))
        
        prompt_parts.append("\n## Generated Response:")
        return "\n".join(prompt_parts)
    
    def _call_neural_model(self, prompt: str) -> str:
        """Simulate neural model call - replace with actual API call in production"""
        # In a real implementation, this would call an external model API
        # For demo purposes, we'll simulate a simple response
        
        # First try to match against all triggers in templates
        for intent_key, template_data in self.templates.items():
            triggers = template_data.get("triggers", [])
            responses = template_data.get("responses", [])
            if isinstance(template_data, dict) and "triggers" in template_data:
                for trigger in template_data["triggers"]:
                    if re.search(r'\b' + re.escape(trigger.lower()) + r'\b', prompt.lower()):
                        responses = template_data.get("responses", [])
                        if responses:
                            return random.choice(responses)
        
        # If no trigger matched, use default responses
        default_data = self.templates.get("default")
        if isinstance(default_data, dict):
            responses = default_data.get("responses", [])
        elif isinstance(default_data, list):
            responses = default_data
        else:
            responses = ["I'm not sure how to respond to that."]
        
        return random.choice(responses)
    
    def _validate_response(self, response: str) -> bool:
        """Ensure response meets quality standards"""
        # Basic sanity checks
        if not response or len(response) < 2:
            return False
            
        # Check for common error patterns
        error_patterns = [
            r"as an AI language model",
            r"cannot (answer|respond)",
            r"apologi(es|ze)"
        ]
        if any(re.search(p, response, re.IGNORECASE) for p in error_patterns):
            return False
            
        # Check response length constraints
        word_count = len(response.split())
        min_words = self.neural_config.get("min_words", 1)
        max_words = self.neural_config.get("max_words", 100)
        
        return min_words <= word_count <= max_words

    def _match_by_trigger(self, user_text: str) -> str:
        user_text_lower = user_text.lower()
        user_text_clean = re.sub(r'[^\w\s]', '', user_text_lower)  # Remove punctuation
        best_score = -1
        best_responses = []
    
        for intent_key, template_entry in self.templates.items():
            if isinstance(template_entry, dict) and "triggers" in template_entry:
                for trigger in template_entry["triggers"]:
                    trigger_lower = trigger.lower()
                    trigger_clean = re.sub(r'[^\w\s]', '', trigger_lower)
    
                    # Smart match: full-word or phrase matching using regex
                    pattern = r'\b' + re.escape(trigger_clean) + r'\b'
                    match = re.search(pattern, user_text_clean)
    
                    if match:
                        position_score = 1.0 - (match.start() / max(1, len(user_text_clean)))  # Earlier is better
                        length_score = len(trigger_clean) / max(1, len(user_text_clean))  # Longer match is better
                        total_score = position_score + length_score  # Simple heuristic
    
                        if total_score > best_score:
                            best_score = total_score
                            best_responses = template_entry.get("responses", [])
    
        if best_responses:
            return random.choice(best_responses)
    
        # Fallback logic
        default_templates = self.templates.get("default")
        if isinstance(default_templates, dict):
            return random.choice(default_templates.get("responses", ["I'm not sure how to respond."]))
        elif isinstance(default_templates, list):
            return random.choice(default_templates)
        return "I'm not sure how to respond."

    def _adapt_style(self, text: str) -> str:
        """Adjust formality and verbosity based on self.style settings."""
        printer.status("INIT", "Style adaptor initialized", "info")

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
            informal_phrases = self.style.get('informal_phrases', [])
            if informal_phrases and random.random() < 0.3:
                text = f"{random.choice(informal_phrases)} {text}"
    
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
                verbosity_mode = self.style.get('verbosity_mode', 'default')
                phrase_pool = self.verbose_phrases.get(verbosity_mode, self.verbose_phrases.get("default", []))
                if phrase_pool and random.random() < 0.4: # Only add if pool is not empty
                    prepend = random.choice(phrase_pool)
                    text = f"{prepend} {text}"

        return text.strip()


if __name__ == "__main__":
    print("\n=== Running Natural Language Generator Engine (NLP Engine) ===\n")
    printer.status("Init", "NLG Engine initialized", "success")

    engine = NLGEngine()

    print(engine)

    print("\n* * * * * Phase 2 * * * * *\n")
    frame1 = LinguisticFrame(
        intent="weather_inquiry",
        entities={
            "location": "Paris",
            "temperature": "21",
            "unit": "C",
            "weather_status": "partly cloudy",
            "weather_condition": "cloudy with sunny breaks"
        },
        sentiment=0.1,
        modality="epistemic",
        confidence=0.95,
        act_type=SpeechActType.DIRECTIVE
    )
    generate = engine.generate(frame=frame1)
    match = engine._match_template(frame=frame1)
    printer.status("GENERATE", generate, "success" if generate else "error")
    printer.status("TEMPLATE", match, "success" if match else "error")

    print("\n* * * * * Phase 3 * * * * *\n")
    text="catch you"
    frame2 = LinguisticFrame(
        intent= "",
        entities={},
        sentiment=0.1,
        modality="epistemic",
        confidence=0.95,
        act_type=SpeechActType.DIRECTIVE
    )
    generation = engine._neural_generation(frame=frame2, context=text)
    printer.status("TEMPLATE", generation, "success" if generation else "error")

    print("\n=== NLG Engine Tests Complete ===\n")
