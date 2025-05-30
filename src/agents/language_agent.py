__version__ = "1.8.0"

import torch
import math
import time
import re
import yaml, json
import torch.nn as nn

from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict, deque
from typing import Dict, List, Any, Union, Optional, Callable

from src.agents.safety.safety_guard import SafetyGuard
from src.agents.base_agent import BaseAgent
from src.agents.language.orthography_processor import OrthographyProcessor, load_config
from src.agents.language.dialogue_context import DialogueContext
from src.agents.language.grammar_processor import (
    GrammarProcessor, InputToken as GrammarProcessorInputToken, GrammarAnalysisResult)
from src.agents.language.nlg_engine import NLGEngine
from src.agents.language.nlu_engine import NLUEngine, Wordlist
from src.agents.language.nlp_engine import NLPEngine, Token as NLPEngineToken
from src.agents.language.utils.rules import DependencyRelation
from src.agents.language.utils.linguistic_frame import LinguisticFrame, SpeechActType
from logs.logger import get_logger

logger = get_logger("Language Agent")

LOCAL_CONFIG_PATH = "src/agents/language/configs/language_config.yaml"

class LanguageAgent(BaseAgent):
    def __init__(self,
                 shared_memory, agent_factory,
                 config=None,
                 args=(), kwargs={}):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config)

        self.language_agent = []
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

        # Initialize Wordlist with config
        self.wordlist = Wordlist(config=load_config(LOCAL_CONFIG_PATH), n=3)

        # Initialize Components
        self.orthography_processor = OrthographyProcessor(config=load_config(LOCAL_CONFIG_PATH))
        self.grammar_processor = GrammarProcessor(config=load_config(LOCAL_CONFIG_PATH))
        self.dialogue_context = DialogueContext(config=load_config(LOCAL_CONFIG_PATH))
        self.nlp_engine = NLPEngine(config=load_config(LOCAL_CONFIG_PATH))
        self.nlu_engine = NLUEngine(
            config=load_config(LOCAL_CONFIG_PATH),
            tokenizer=None, #self.tokenizer,
            glove_vectors=None,#self.glove_vectors,
            structured_wordlist=None, #self.structured_wordlist,
            wordlist_instance=self.wordlist
        )
        self.nlg_engine = NLGEngine(config=load_config(LOCAL_CONFIG_PATH))
        self.safety_guard = SafetyGuard()
        self.response()

        print("Language Agent Successfully initialized local components")

    def perform_task(self, task_data: Union[str, Dict]) -> str:
        """
        Overrides BaseAgent.perform_task.
        This is the entry point for the agent's primary logic when called by BaseAgent.execute().
        """
        user_input = ""
        session_id = None

        if isinstance(task_data, str):
            user_input = task_data
        elif isinstance(task_data, dict):
            user_input = task_data.get("query", task_data.get("text", ""))
            session_id = task_data.get("session_id")
            if not user_input and "user_input" in task_data : # common alternative key
                 user_input = task_data.get("user_input","")

            # Extracting entities or intent directly if provided (for testing/sim)
            # This bypasses NLU but could be useful for debugging or specific flows
            # provided_intent = task_data.get("intent")
            # provided_entities = task_data.get("entities")

        if not user_input: # Check if user_input is still empty
            logger.error(f"LanguageAgent perform_task received no valid input from task_data: {str(task_data)[:100]}")
            # Fallback response if input is genuinely missing or uninterpretable
            error_frame = LinguisticFrame("input_error", {"error_message": "No input provided"}, 0.0, "error", 1.0, SpeechActType.ASSERTIVE)
            return self.nlg_engine.generate(error_frame, self.dialogue_context)


        logger.info(f"LanguageAgent performing task with input: '{str(user_input)[:50]}...', session_id: {session_id}")
        return self.pipeline(user_input_text=user_input, session_id=session_id)

    def pipeline(self, user_input_text: str, session_id: Optional[str] = None) -> str:
        """
        Processes user input through the language processing pipeline and returns the agent's response.

        User Input
        ↓
        OrthographyProcessor → spellcheck + normalize
        ↓
        NLPEngine → tokenize, lemmatize, POS, etc.
        ↓
        GrammarProcessor → parse + grammar checks (uses NLPEngine)
        ↓
        NLUEngine → determine intent/entities (uses NLPEngine + tokenizer + embeddings)
        ↓
        DialogueContext → tracks recent turns
        ↓
        NLGEngine → generate response
        ↓
        DialogueContext → log new turn
        ↓
        Agent Output
        """
        logger.info(f"Pipeline started for input: '{user_input_text[:50]}...'")
        # self.config = config
        # user_input = {}

        # 0a. Sanitize User Input
        try:
            # Keep original input for history logging before sanitization changes it
            original_user_input = user_input_text
            user_input_text = self.safety_guard.sanitize(user_input_text, depth='balanced')
            logger.debug(f"Sanitized user input: '{user_input_text[:50]}...'")
        except Exception as e:
            logger.error(f"SafetyGuard sanitization failed for user input: {e}. Aborting pipeline for safety.")
            # Return a safety-specific response
            safe_response = "I'm sorry, your input triggered a safety concern. Please rephrase or try something else."
            # Log original unsanitized input with a warning if needed for debugging/auditing
            logger.warning(f"Safety block triggered by original input: '{original_user_input}'")
            # Add to dialogue context as a user input that failed
            self.dialogue_context.add_message(role="user", content=original_user_input + " [Safety Blocked]")
            self.dialogue_context.add_message(role="agent", content=safe_response) # Log agent's safety response
            return safe_response

        # 0b. Update DialogueContext with session_id
        if session_id and self.dialogue_context.get_environment_state("session_id") != session_id:
            self.dialogue_context.update_environment_state("session_id", session_id)
            logger.info(f"Dialogue context session ID set to: {session_id}")

        # 1. Orthography Processing
        ortho_corrected_text = user_input_text # Start with sanitized text
        try:
            # batch_process might return None or empty string on failure/no change
            processed = self.orthography_processor.batch_process(user_input_text)
            if processed is not None and processed.strip(): # Check if processing yielded results
                 ortho_corrected_text = processed
            logger.debug(f"Orthographically corrected text: '{ortho_corrected_text[:50]}...'")
        except Exception as e:
            logger.error(f"OrthographyProcessor failed: {e}. Using sanitized text: '{user_input_text[:50]}...'.")

        # 2. NLP Engine Processing (Tokenization, Lemmatization, POS, etc.)
        nlp_tokens: List[NLPEngineToken] = []
        try:
            nlp_tokens = self.nlp_engine.process_text(ortho_corrected_text)
            if not nlp_tokens:
                logger.warning("NLPEngine produced no tokens. Aborting further processing for this input.")
                error_frame = LinguisticFrame("nlp_error", {"reason": "No tokens produced"}, 0.0, "error", 0.9, SpeechActType.ASSERTIVE)
                # Log initial user input and the generated response before returning
                self.dialogue_context.add_message(role="user", content=original_user_input) # Log original before processing errors
                agent_response = self.nlg_engine.generate(error_frame, self.dialogue_context)
                self.dialogue_context.add_message(role="agent", content=agent_response)
                return agent_response

            logger.debug(f"NLPEngine processed into {len(nlp_tokens)} tokens.")
        except Exception as e:
            logger.error(f"NLPEngine failed: {e}. Cannot proceed with grammar/NLU.")
            error_frame = LinguisticFrame(
                "nlp_error", {"error_module": "nlp_engine", "detail": str(e)}, 0.0,
                "error", 1.0, SpeechActType.ASSERTIVE)
            # Log initial user input and the generated response before returning
            self.dialogue_context.add_message(role="user", content=original_user_input)
            agent_response = self.nlg_engine.generate(error_frame, self.dialogue_context)
            self.dialogue_context.add_message(role="agent", content=agent_response)
            return agent_response

        # 3. Grammar Processing (Optional for core NLU)
        grammar_analysis_result: Optional[GrammarAnalysisResult] = None
        dependencies: List[DependencyRelation] = []
        try:
            dependencies = self.nlp_engine.apply_dependency_rules(nlp_tokens)
            logger.debug(f"NLPEngine produced {len(dependencies)} dependency relations.")

            # Prepare tokens for GrammarProcessor
            grammar_input_tokens: List[GrammarProcessorInputToken] = []
            current_char_offset = 0
            for i, nlp_tok in enumerate(nlp_tokens):
                # Find head/dep using index from NLPEngine dependencies
                token_head_idx = nlp_tok.index # Default to self
                token_dep_rel = "dep"        # Default to generic dep
                found_dep = False
                for dep_obj in dependencies:
                    if dep_obj.dependent_index == nlp_tok.index:
                        token_head_idx = dep_obj.head_index
                        token_dep_rel = dep_obj.relation
                        # Special handling for ROOT if NLPEngine dependency rules use a conceptual root
                        # If the token is the dependent of the conceptual "ROOT"
                        if dep_obj.head == "ROOT" and dep_obj.relation == "root":
                             token_head_idx = nlp_tok.index # GrammarProcessor.InputToken convention for root
                        found_dep = True
                        break
                # If no dependency rule matched this token as dependent, try if it's a head
                if not found_dep:
                     for dep_obj in dependencies:
                         if dep_obj.head_index == nlp_tok.index and dep_obj.relation == "root":
                             token_head_idx = nlp_tok.index # This token is the root
                             token_dep_rel = "root"
                             break # Found root

                # Simple character offset calculation (acknowledged simplification for BPE/complex text)
                start_char_abs = -1
                end_char_abs = -1
                try:
                     # Search in the current ortho_corrected_text string from the last offset
                     # This will fail or be incorrect for repeated words/subwords
                     start_char_abs = ortho_corrected_text.index(nlp_tok.text, current_char_offset)
                     end_char_abs = start_char_abs + len(nlp_tok.text) - 1
                     current_char_offset = end_char_abs + 1
                except ValueError:
                     # If not found, just increment offset based on token length + space
                     start_char_abs = current_char_offset
                     end_char_abs = current_char_offset + len(nlp_tok.text) - 1
                     current_char_offset = end_char_abs + 1
                     logger.debug(f"Could not find token '{nlp_tok.text}' in ortho_corrected_text starting from offset {current_char_offset}. Using estimated offset.")


                gp_token = GrammarProcessorInputToken(
                    text=nlp_tok.text, lemma=nlp_tok.lemma, pos=nlp_tok.pos,
                    index=nlp_tok.index, head=token_head_idx, dep=token_dep_rel,
                    start_char_abs=start_char_abs, end_char_abs=end_char_abs
                )
                grammar_input_tokens.append(gp_token)

            sentences_for_grammar: List[List[GrammarProcessorInputToken]] = [grammar_input_tokens] # Assuming single sentence
            
            grammar_analysis_result = self.grammar_processor.analyze_text(
                sentences_for_grammar,
                full_text_snippet=ortho_corrected_text
            )
            if grammar_analysis_result:
                logger.debug(f"Grammar analysis complete. Grammatical: {grammar_analysis_result.is_grammatical}")
                # Log issues if any
                for sent_analysis in grammar_analysis_result.sentence_analyses:
                    if sent_analysis.get('issues'):
                        logger.info(f"Grammar issues in '{sent_analysis.get('text','N/A')[:30]}...': {len(sent_analysis['issues'])}")
        except Exception as e:
            logger.error(f"GrammarProcessor failed: {e}")
            # Grammar check is not strictly blocking for NLU/NLG

        # 4. NLU Engine Processing (Intent, Entities)
        linguistic_frame: Optional[LinguisticFrame] = None
        try:
            # NLU parse takes the ortho_corrected_text string
            linguistic_frame = self.nlu_engine.parse(ortho_corrected_text)
            logger.debug(f"NLU result: Intent='{linguistic_frame.intent}', Entities='{linguistic_frame.entities}', Confidence='{linguistic_frame.confidence:.2f}'")

            # === Specific NLU Handling based on Detected Intent/Entities ===
            # Example: For 'time_request', if no specific time entity was found, add a default.
            # This compensates for the simple pattern matching not extracting "current time".
            if linguistic_frame.intent == "time_request":
                # Check if entities DATE_TIME or TIME were extracted
                time_entity_found = any(et in linguistic_frame.entities for et in ["DATE_TIME", "TIME"])
                if not time_entity_found:
                    # Populate a default entity expected by NLG templates
                    linguistic_frame.entities["time"] = ["current system time"] # Using lowercase 'time' to match NLG template key

            # Example: If sentiment is strongly negative, potentially flag for different response strategy
            if linguistic_frame.sentiment < -0.5:
                 logger.info("Detected strong negative sentiment. May influence NLG.")
                 # This sentiment info is already in the frame for NLG, but could trigger other actions

            logger.info(f"Low confidence intent detected ({linguistic_frame.confidence:.2f}).")
            # Track low-confidence state and tentative intent
            self.dialogue_context.add_unresolved(
                issue="low_confidence_intent",
                slot=None
            )
            self.dialogue_context.update_environment_state(
                "pending_intent",
                linguistic_frame.intent
            )
            self.dialogue_context.update_environment_state(
                "pending_entities",
                linguistic_frame.entities
            )

        except Exception as e:
            logger.error(f"NLUEngine failed: {e}. Using fallback linguistic frame.")
            linguistic_frame = LinguisticFrame(
                intent="nlu_error",
                entities={"error_module": "nlu_engine", "detail": str(e)},
                sentiment=0.0,
                modality="error",
                confidence=1.0,
                act_type=SpeechActType.ASSERTIVE
            )

        # 5. Dialogue Context Update (User Turn)
        try:
            # Log the original user input for historical accuracy, alongside processed info
            self.dialogue_context.add_message(role="user", content=original_user_input) 
            if linguistic_frame:
                self.dialogue_context.register_intent(intent=linguistic_frame.intent, confidence=linguistic_frame.confidence)
                if linguistic_frame.entities:
                    for entity_type, entity_values in linguistic_frame.entities.items():
                        # NLUEngine.parse has entities as Dict[str, Any]. If 'Any' is a list of values:
                        value_to_log = entity_values[0] if isinstance(entity_values, list) and entity_values else entity_values
                        # Ensure value_to_log is not None or empty before updating slot
                        if value_to_log is not None and (isinstance(value_to_log, str) and value_to_log.strip() != '' or not isinstance(value_to_log, str)):
                             self.dialogue_context.update_slot(entity_type, value_to_log)
            logger.debug("DialogueContext updated with user turn.")
        except Exception as e:
            logger.error(f"DialogueContext update (user turn) failed: {e}")

        # 6. NLG Engine Processing (Generate Response)
        pending_clarification = any(
            issue.get('description') == 'low_confidence_intent' 
            for issue in self.dialogue_context.unresolved_issues
        )
        
        if pending_clarification:
            # Generate clarification prompt
            clarification_frame = LinguisticFrame(
                intent="clarification_request",
                entities={
                    "pending_intent": self.dialogue_context.get_environment_state("pending_intent"),
                    "mentioned_entities": self.dialogue_context.get_environment_state("pending_entities")
                },
                sentiment=0.0,
                modality="clarification",
                confidence=1.0,
                act_type=SpeechActType.DIRECTIVE
            )
            agent_response_text = self.nlg_engine.generate(
                frame=clarification_frame,
                context=self.dialogue_context
            )
        else:
            # Proceed with normal response generation
            agent_response_text = self.nlg_engine.generate(
                frame=linguistic_frame,
                context=self.dialogue_context
            )

        # 6b. Sanitize Agent Output
        agent_response_to_log = agent_response_text  # Keep original before sanitization
        try:
            processed_response = self.safety_guard.sanitize(agent_response_text, depth='balanced')
            if processed_response is not None and processed_response.strip():
                agent_response_text = processed_response
            logger.debug(f"Sanitized agent response: '{agent_response_text[:50]}...'")
        except Exception as e:
            logger.error(f"SafetyGuard sanitization failed for agent response: {e}. Sending generic safe response.")
            agent_response_text = "I apologize, but I encountered an issue ensuring my response was safe. Please try rephrasing."
            logger.warning(f"Agent response failed safety check: '{agent_response_to_log}'")


        # 7. Dialogue Context Update (Agent Turn)
        try:
            self.dialogue_context.add_message(role="agent", content=agent_response_text)
            logger.debug("DialogueContext updated with agent turn.")
        except Exception as e:
            logger.error(f"DialogueContext update (agent turn) failed: {e}")

        # 8. Return Agent Output
        logger.info(f"Pipeline finished. Output: '{agent_response_text[:50]}...'")
        return agent_response_text

    def response(self):
        pass

    def _preprocess_with_ortho(self, text: str) -> str:
        """Add agent-specific spelling checks before standard processing"""
        if self.config.get("custom_dictionary_check"):
            return self.orthography_processor.correct_with_custom_rules(text)
        return self.orthography_processor.batch_process(text)
    
    def _get_enhanced_entities(self, tokens: list):
        """Add agent-specific entity enrichment"""
        entities = self.nlp_engine.extract_entities(tokens)
        if self.config.get("augment_with_wordnet"):
            return self._augment_entities_with_wordnet(entities)
        return entities

    def _dialogue_context(self, config=LOCAL_CONFIG_PATH):
        """
        Handle high-level dialogue flow and state management.
        - Manage clarification cycles
        - Reset context when needed
        - Check for conversation timeouts
        """
        # 1. Check for unresolved issues requiring agent action
        if any(issue['description'] == 'low_confidence_intent' 
               for issue in self.dialogue_context.unresolved_issues):
            logger.info("Pending intent clarification detected")
            return self._handle_clarification_flow()
    
        # 2. Check conversation timeout
        if self.dialogue_context.get_time_since_last_interaction() > 300:  # 5 minutes
            logger.info("Resetting stale conversation context")
            self.dialogue_context.clear()
            return True
    
        # 3. Check required slots fulfillment
        if not self.dialogue_context.required_slots_filled:
            missing = self.dialogue_context.get_missing_slots()
            logger.info(f"Missing required slots: {missing}")
            return False
    
        return None
    
    def _handle_clarification_flow(self):
        """Manage clarification attempts and fallbacks"""
        clarification_attempts = sum(
            1 for issue in self.dialogue_context.unresolved_issues
            if issue['description'] == 'low_confidence_intent'
        )
        
        if clarification_attempts > self.load_dialogue_policy().get('reprompt_limit', 2):
            logger.warning("Clarification attempt limit reached")
            self.dialogue_context.unresolved_issues = [
                issue for issue in self.dialogue_context.unresolved_issues
                if issue['description'] != 'low_confidence_intent'
            ]
            return False
        
        return True

    def load_dialogue_policy(self) -> Dict:
        """Load conversation rules from embedded config or a file."""
        # This can be expanded to load from a YAML/JSON if policy becomes complex
        policy_config = self.config.get("dialogue_policy", {}) # Assuming 'dialogue_policy' key in main agent config
        
        # Fallback to default if not found in config
        default_policy = {
            'clarification_triggers': ['unknown_intent', 'nlu_error', 'low_confidence_intent'], # Intent names or conditions
            'low_confidence_threshold': 0.4, # Confidence below which an intent might trigger clarification
            'reprompt_limit': 2, # Max number of reprompts for clarification before fallback
            'fallback_responses': [
                "Could you please rephrase that?",
                "I'm not quite sure I understand. Can you say that another way?",
                "I'm still learning. Could you provide more details?"
            ],
            'error_responses': {
                "nlp_error": "I had a bit of trouble understanding the structure of your message. Could you try again?",
                "nlu_error": "I'm having difficulty grasping the meaning. Please rephrase.",
                "nlg_error": "I apologize, I couldn't formulate a proper response.",
                "default_error": "Sorry, something went wrong on my end."
            }
        }
        # Merge loaded config with defaults, giving precedence to loaded config
        merged_policy = {**default_policy, **policy_config}
        # Deep merge for nested dicts like 'error_responses'
        if 'error_responses' in policy_config and isinstance(policy_config['error_responses'], dict):
            merged_policy['error_responses'] = {**default_policy['error_responses'], **policy_config['error_responses']}
        
        return merged_policy


#if __name__ == "__main__":

#    print("\n=== Running Language Agent ===\n")

#    config = load_config(LOCAL_CONFIG_PATH)
#    shared_memory = {}
#    agent_factory = lambda: None

#    language = LanguageAgent(shared_memory, agent_factory, config)

#    print(language)
#    print("\n=== Successfully Ran Language Agent ===\n")

if __name__ == "__main__":
    print("\n=== Running Language Agent ===\n")

    # Load configuration and initialize agent
    # For standalone testing, we create a dummy BaseAgent config if LanguageAgent expects it
    # or we pass None and let LanguageAgent use its defaults.
    base_agent_config = {
        "language_yaml_config_path": LOCAL_CONFIG_PATH # Point to the language specific config
    }
    class DummySharedMemory:
        def __init__(self): self._data = {}
        def get(self, key, default=None):
            # Simulate key retrieval, including agent_stats for BaseAgent logging
            if key.startswith("agent_stats:"):
                return self._data.get(key, {}) # Return empty dict if not found
            return self._data.get(key, default)
        def set(self, key, value): self._data[key] = value
        def delete(self, key):
            if key in self._data: del self._data[key]
        def get_all_versions(self,key): return [self._data[key]] if key in self._data else []
        def register_callback(self, key, cb): pass # Dummy method
        def clear_all(self): self._data.clear()
        # Add other methods expected by BaseAgent if needed
        def get_all_keys(self): return list(self._data.keys())
    config = Dict
    
    shared_memory_instance = DummySharedMemory()
    agent_factory_instance = lambda name, cfg: None # Dummy factory

    try:
        language_agent = LanguageAgent(
            shared_memory=shared_memory_instance,
            agent_factory=agent_factory_instance,
            config=base_agent_config # Pass the base_agent_config
        )
        print("Language Agent initialized. Type your message below.")
        print("Type 'exit' or 'quit' to end the session.\n")

        session_id = f"interactive-session-{time.time()}"

        while True:
            try:
                user_input = input("User: ").strip()
                if user_input.lower() in {"exit", "quit"}:
                    print("Exiting Language Agent. Goodbye!")
                    break
                if not user_input:
                    print("Agent: Please say something.\n")
                    continue

                # Using perform_task as the entry point
                response = language_agent.perform_task({"text": user_input, "session_id": session_id})
                # Or directly call pipeline:
                # response = language_agent.pipeline(user_input, session_id=session_id)
                
                print(f"Agent: {response}\n")

            except KeyboardInterrupt:
                print("\n[Interrupted] Exiting Language Agent.")
                break
            except Exception as e:
                logger.error(f"Error in interactive loop: {e}", exc_info=True)
                print(f"[Error] Something went wrong: {e}")
                # break # Optional: break on error or allow continuation
    
    except Exception as e:
        logger.error(f"Failed to initialize LanguageAgent: {e}", exc_info=True)
        print(f"[Fatal Error] Could not start Language Agent: {e}")
