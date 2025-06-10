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

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.safety.safety_guard import SafetyGuard
from src.agents.language.orthography_processor import OrthographyProcessor
from src.agents.language.dialogue_context import DialogueContext
from src.agents.language.grammar_processor import (
    GrammarProcessor, InputToken as GrammarProcessorInputToken, GrammarAnalysisResult)
from src.agents.language.nlg_engine import NLGEngine
from src.agents.language.nlu_engine import NLUEngine, Wordlist
from src.agents.language.nlp_engine import NLPEngine, Token as NLPEngineToken
from src.agents.language.utils.rules import DependencyRelation
from src.agents.language.utils.linguistic_frame import LinguisticFrame, SpeechActType
from src.agents.base_agent import BaseAgent
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Language Agent")
printer = PrettyPrinter

class LanguageAgent(BaseAgent):
    def __init__(self,
                 shared_memory, agent_factory,
                 config=None,):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config)
        self.config = load_global_config()
        self.language_config = get_config_section('language_agent')

        self.language_agent = []
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

        # Initialize Wordlist
        self.wordlist = Wordlist()

        # Initialize Components
        self.orthography_processor = OrthographyProcessor()
        self.grammar_processor = GrammarProcessor()
        self.dialogue_context = DialogueContext()
        self.nlp_engine = NLPEngine()
        self.nlu_engine = NLUEngine(wordlist_instance=self.wordlist)
        self.nlg_engine = NLGEngine()
        self.safety_guard = SafetyGuard()
        self.response()

        print("Language Agent Successfully initialized local components")

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

        # 0a. Sanitize User Input
        try:
            original_user_input = user_input_text
            user_input_text = self.safety_guard.sanitize(user_input_text, depth='balanced')
            logger.debug(f"Sanitized user input: '{user_input_text[:50]}...'")
        except Exception as e:
            logger.error(f"SafetyGuard sanitization failed for user input: {e}. Aborting pipeline for safety.")
            safe_response = "I'm sorry, your input triggered a safety concern. Please rephrase or try something else."
            logger.warning(f"Safety block triggered by original input: '{original_user_input}'")
            self.dialogue_context.add_message(role="user", content=original_user_input + " [Safety Blocked]")
            self.dialogue_context.add_message(role="agent", content=safe_response)
            return safe_response

        # 0b. Update DialogueContext with session_id
        if session_id and self.dialogue_context.get_environment_state("session_id") != session_id:
            self.dialogue_context.update_environment_state("session_id", session_id)
            logger.info(f"Dialogue context session ID set to: {session_id}")

        # 1. Orthography Processing
        ortho_corrected_text = user_input_text
        try:
            processed = self.orthography_processor.batch_process(user_input_text)
            if processed is not None and processed.strip():
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
                self.dialogue_context.add_message(role="user", content=original_user_input)
                agent_response = self.nlg_engine.generate(error_frame, self.dialogue_context)
                self.dialogue_context.add_message(role="agent", content=agent_response)
                return agent_response
            logger.debug(f"NLPEngine processed into {len(nlp_tokens)} tokens.")
        except Exception as e:
            logger.error(f"NLPEngine failed: {e}. Cannot proceed with grammar/NLU.")
            error_frame = LinguisticFrame(
                "nlp_error", {"error_module": "nlp_engine", "detail": str(e)}, 0.0,
                "error", 1.0, SpeechActType.ASSERTIVE)
            self.dialogue_context.add_message(role="user", content=original_user_input)
            agent_response = self.nlg_engine.generate(error_frame, self.dialogue_context)
            self.dialogue_context.add_message(role="agent", content=agent_response)
            return agent_response

        # 3. Grammar Processing
        grammar_analysis_result: Optional[GrammarAnalysisResult] = None
        dependencies: List[DependencyRelation] = []
        try:
            dependencies = self.nlp_engine.apply_dependency_rules(nlp_tokens)
            logger.debug(f"NLPEngine produced {len(dependencies)} dependency relations.")
        
            grammar_input_tokens: List[GrammarProcessorInputToken] = []
            # Map token original index to its NLPEngineToken object for quick lookup if needed
            indexed_nlp_tokens = {tok.index: tok for tok in nlp_tokens} # Not strictly needed with current dep structure
        
            # Use reconstructed text from tokens instead of ortho_corrected_text
            reconstructed_text = " ".join([tok.text for tok in nlp_tokens])
            current_char_offset = 0
        
            for nlp_tok in nlp_tokens:
                token_head_idx = nlp_tok.index  # Default: token is its own head
                token_dep_rel = "dep"          # Default: generic dependency relation (or "root" if it's a root)
        
                is_explicit_root = False
                # Find this token in the dependency relations
                for dep_rel in dependencies:
                    if dep_rel.dependent_index == nlp_tok.index:
                        if dep_rel.head == "ROOT" and dep_rel.relation == "root":
                            # This token is the root of the sentence as defined by NLPEngine
                            token_head_idx = nlp_tok.index # Root points to itself for GrammarProcessor
                            token_dep_rel = "root"
                            is_explicit_root = True
                        else:
                            # Ensure head_index refers to an actual token's index
                            # NLPEngine's DependencyRelation.head_index is the original index of the head token
                            token_head_idx = dep_rel.head_index
                            token_dep_rel = dep_rel.relation
                        break # Found relation for this token as a dependent
                
                if not is_explicit_root and not any(dep_rel.dependent_index == nlp_tok.index for dep_rel in dependencies):
                    # If token is not a dependent of anything, and not explicitly marked as ROOT's dependent,
                    # it could be an isolated token or the implicit root of a fragment.
                    # Check if it's marked as a head of a "root" relation (another way NLPEngine might mark roots)
                    is_head_of_root_rel = False
                    for dep_rel in dependencies:
                        if dep_rel.head_index == nlp_tok.index and dep_rel.relation == "root" and dep_rel.head == "ROOT":
                            token_head_idx = nlp_tok.index
                            token_dep_rel = "root"
                            is_head_of_root_rel = True
                            break
                    if not is_head_of_root_rel: # If truly unattached by rules, mark as its own root
                         token_head_idx = nlp_tok.index
                         token_dep_rel = "root" # Or "dep" if convention is different
        
                # Calculate character offsets using reconstructed text
                try:
                    start_char_abs = reconstructed_text.index(nlp_tok.text, current_char_offset)
                    end_char_abs = start_char_abs + len(nlp_tok.text) - 1
                    current_char_offset = start_char_abs + len(nlp_tok.text)
                except ValueError:
                    # Fallback to simple position estimation
                    start_char_abs = current_char_offset
                    end_char_abs = current_char_offset + len(nlp_tok.text) - 1
                    current_char_offset = end_char_abs + 2
                    logger.warning(f"Token '{nlp_tok.text}' position estimated: {start_char_abs}-{end_char_abs}")
        
                gp_token = GrammarProcessorInputToken(
                    text=nlp_tok.text,
                    lemma=nlp_tok.lemma,
                    pos=nlp_tok.pos,
                    index=nlp_tok.index, # Use the original index from NLPEngineToken
                    head=token_head_idx, # Head is the index of the head token
                    dep=token_dep_rel,
                    start_char_abs=start_char_abs,
                    end_char_abs=end_char_abs
                )
                grammar_input_tokens.append(gp_token)

            # Assuming single sentence for now, based on how nlp_tokens are structured
            sentences_for_grammar: List[List[GrammarProcessorInputToken]] = [grammar_input_tokens] if grammar_input_tokens else []
            
            if sentences_for_grammar: # Only analyze if there are tokens
                grammar_analysis_result = self.grammar_processor.analyze_text(
                    sentences_for_grammar,
                    full_text_snippet=ortho_corrected_text
                )
                if grammar_analysis_result:
                    logger.debug(f"Grammar analysis complete. Grammatical: {grammar_analysis_result.is_grammatical}")
                    for sent_analysis in grammar_analysis_result.sentence_analyses:
                        if sent_analysis.get('issues'):
                            logger.info(f"Grammar issues in '{sent_analysis.get('text','N/A')[:30]}...': {len(sent_analysis['issues'])}")
            else:
                logger.info("No tokens available for grammar processing.")

        except Exception as e:
            logger.error(f"GrammarProcessor or pre-processing failed: {e}", exc_info=True)
            # Grammar check is not strictly blocking for NLU/NLG

        # 4. NLU Engine Processing (Intent, Entities)
        linguistic_frame: Optional[LinguisticFrame] = None
        try:
            linguistic_frame = self.nlu_engine.parse(ortho_corrected_text) # Returns LinguisticFrame object
            logger.debug(f"NLU result: Intent='{linguistic_frame.intent}', Entities='{linguistic_frame.entities}', Sentiment='{linguistic_frame.sentiment:.2f}', Modality='{linguistic_frame.modality}', Confidence='{linguistic_frame.confidence:.2f}'")

            if linguistic_frame.intent == "time_request":
                time_entity_found = any(et in linguistic_frame.entities for et in ["DATE_TIME", "TIME", "time"])
                if not time_entity_found:
                    linguistic_frame.entities["time"] = "current system time"

            # Example: Low confidence intent handling (using dialogue_policy threshold)
            dialogue_policy = self.load_dialogue_policy()
            if linguistic_frame.confidence < dialogue_policy.get('low_confidence_threshold', 0.5): # Using 0.5 as a fallback default
                logger.info(f"Low confidence intent detected ({linguistic_frame.confidence:.2f}).")
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
            logger.error(f"NLUEngine failed: {e}", exc_info=True)
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
            self.dialogue_context.add_message(role="user", content=original_user_input)
            if linguistic_frame:
                self.dialogue_context.register_intent(intent=linguistic_frame.intent, confidence=linguistic_frame.confidence)
                if linguistic_frame.entities:
                    for entity_type, entity_values in linguistic_frame.entities.items():
                        value_to_log = entity_values[0] if isinstance(entity_values, list) and entity_values else entity_values
                        if value_to_log is not None and (isinstance(value_to_log, str) and value_to_log.strip() != '' or not isinstance(value_to_log, str)):
                             self.dialogue_context.update_slot(entity_type, value_to_log)
            logger.debug("DialogueContext updated with user turn.")
        except Exception as e:
            logger.error(f"DialogueContext update (user turn) failed: {e}", exc_info=True)

        # 6. NLG Engine Processing (Generate Response)
        pending_clarification = any(
            issue.get('description') == 'low_confidence_intent' 
            for issue in self.dialogue_context.unresolved_issues
        )
        
        agent_response_text = ""
        if pending_clarification:
            clarification_intent = self.dialogue_context.get_environment_state("pending_intent")
            clarification_entities = self.dialogue_context.get_environment_state("pending_entities") or {}
            
            # Ensure entities are in a format NLG can handle (e.g., string representations)
            formatted_entities_for_nlg = {}
            if isinstance(clarification_entities, dict):
                for k, v_list in clarification_entities.items():
                    if isinstance(v_list, list) and v_list:
                        # Take the first item if it's a list, or join if appropriate
                        formatted_entities_for_nlg[k] = str(v_list[0]) 
                    elif v_list is not None:
                         formatted_entities_for_nlg[k] = str(v_list)
            
            clarification_frame = LinguisticFrame(
                intent="clarification_request", # This intent should exist in nlg_templates.json
                entities={
                    "pending_intent": clarification_intent or "your request",
                    "mentioned_entities": ", ".join(f"{k}: {v}" for k,v in formatted_entities_for_nlg.items()) or "the details provided"
                },
                sentiment=0.0,
                modality="interrogative", # A clarification is often a question
                confidence=1.0,
                act_type=SpeechActType.DIRECTIVE # Requesting user to act (clarify)
            )
            agent_response_text = self.nlg_engine.generate(
                frame=clarification_frame,
                context=self.dialogue_context # Pass DialogueContext object
            )
        else:
            if linguistic_frame is None: # Should have been caught or defaulted by NLU error handling
                 linguistic_frame = LinguisticFrame("internal_error", {}, 0.0, "error", 1.0, SpeechActType.ASSERTIVE)
            agent_response_text = self.nlg_engine.generate(
                frame=linguistic_frame,
                context=self.dialogue_context # Pass DialogueContext object
            )

        # 6b. Sanitize Agent Output
        agent_response_to_log = agent_response_text 
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
            logger.error(f"DialogueContext update (agent turn) failed: {e}", exc_info=True)

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

    def _dialogue_context(self):
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
            'reprompt_limit': 5, # Max number of reprompts for clarification before fallback
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

if __name__ == "__main__":
    print("\n=== Running Language Agent ===\n")
    from src.agents.collaborative.shared_memory import SharedMemory
    shared_memory_instance = SharedMemory()
    agent_factory_instance = lambda name, cfg: None

    try:
        language_agent = LanguageAgent(
            shared_memory=shared_memory_instance,
            agent_factory=agent_factory_instance,
            config=None
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
                response = language_agent.pipeline(user_input_text=user_input, session_id=session_id)
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
