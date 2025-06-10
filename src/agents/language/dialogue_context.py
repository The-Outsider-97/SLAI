""""
Core Function:
Maintains conversational memory, tracks interaction state, and helps the agent behave coherently across turns.

Responsibilities:
Store user/system turns (short-term memory)
Track slots, filled entities, or unresolved issues
Maintain intent history, topic focus, etc.
Detect context-switching or follow-up questions

Why it matters:
Prevents the agent from being stateless or reactive-only.
Essential for multi-turn coherence, personalization, or contextual disambiguation.
"""
import re
import torch
import yaml, json
import torch.nn.functional as F

from typing import Optional, List, Callable, Dict, Any, Union, Tuple
from datetime import datetime
from pathlib import Path

from src.agents.language.utils.config_loader import load_global_config, get_config_section
from src.agents.language.utils.language_transformer import LanguageTransformer
from src.agents.language.utils.language_tokenizer import LanguageTokenizer
from src.agents.language.utils.language_cache import LanguageCache
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Dialogue Context")
printer = PrettyPrinter

class DialogueContext:
    def __init__(self):
        """
        Manages dialogue memory.
        """
        self.config = load_global_config()
        self.wordlist_path = self.config.get('main_wordlist_path')

        self.dialogue_config = get_config_section('dialogue_context')
        self.memory_limit = self.dialogue_config.get('memory_limit')
        self.required_slots = self.dialogue_config.get('required_slots')
        self.enable_summarization = self.dialogue_config.get('enable_summarization')
        self.initial_history = self.dialogue_config.get('initial_history')
        self.initial_summary = self.dialogue_config.get('initial_summary')
        self.default_initial_history = self.dialogue_config.get('default_initial_history')
        self.default_initial_summary = self.dialogue_config.get('default_initial_summary')
        self.follow_up_patterns_path = self.dialogue_config.get('follow_up_patterns_path')
        self.topic_detection = self.dialogue_config.get('topic_detection', {
            'similarity_threshold', 'lookback_window', 'encoder_model'
        })
        self.summarization = self.dialogue_config.get('summarization', {
            'retain_last_messages', 'summary_update_strategy', 'max_summary_length'
        })
        self.persistence = self.dialogue_config.get('persistence', {
            'auto_save_interval', 'default_save_path', 'encryption_key'
        })
        self.temporal = self.dialogue_config.get('temporal', {
            'session_timeout', 'time_reference_format'
        })
        self.environment_state = self.dialogue_config.get(
            "initial_environment_state", 
            self.dialogue_config.get("default_initial_environment_state", 
                {"session_id": None, "user_preferences": {}, "last_intent": None}
            )
        )

        self.cache = LanguageCache()

        self.history: List[Dict[str, str]] = []
        self.summarizer_fn: Optional[Callable[[List[Dict[str, str]], Optional[str]], str]] = {}

        # Slot/Entity Tracking System
        self.slot_values: Dict[str, Any] = {}
        self.unresolved_issues: List[Dict] = []
        self.summary = self.initial_summary or self.default_initial_summary or ""
        self.intent_history = []

        self._initialize_history()

        logger.info(f"DialogueContext initialized. Memory limit: {self.memory_limit}, Summarization: {self.enable_summarization}")

    def _initialize_history(self):
        initial_history_raw = self.dialogue_config.get("initial_history", [])
        if not initial_history_raw: # If initial_history is explicitly empty or not provided, use default
             default_history_raw = self.dialogue_config.get("default_initial_history", ["System: Hello! How can I assist you today?"])
             initial_history_raw = default_history_raw

        for item in initial_history_raw:
            if isinstance(item, str) and ":" in item:
                try:
                    role, content = item.split(":", 1)
                    self.history.append({"role": role.strip().lower(), "content": content.strip()})
                except ValueError:
                    logger.warning(f"Malformed history item string: '{item}'. Storing as 'unknown' role.")
                    self.history.append({"role": "unknown", "content": item})
            elif isinstance(item, dict) and "role" in item and "content" in item:
                self.history.append({"role": str(item["role"]).lower(), "content": str(item["content"])})
            else:
                logger.warning(f"Skipping malformed initial history item: {item}")

    def add_turn(self, user_input: str, agent_response: str):
        """Adds a user input and agent response as a turn."""
        self.add_message("user", user_input)
        self.add_message("agent", agent_response) # Or assistant, bot, etc.

    def add_message(self, role: str, content: str):
        """Adds a single message to the history."""
        if not isinstance(role, str) or not isinstance(content, str):
            logger.error(f"Invalid message format: role and content must be strings. Got role={type(role)}, content={type(content)}")
            return
            
        self.history.append({"role": role.lower(), "content": content})
        logger.debug(f"Added message: Role='{role.lower()}', Content='{content[:50]}...'")
        
        # Summarization is typically based on turns (user-agent pairs)
        # We count user messages to approximate turns for memory_limit.
        user_messages_count = sum(1 for msg in self.history if msg["role"] == "user")
        if self.enable_summarization and self.summarizer_fn and user_messages_count > self.memory_limit:
            self._summarize()

    def _summarize(self):
        """Summarize current history and reset memory if a summarizer function is provided."""
        retain_last = 2  # Keep immediate context
        self.history = self.history[-retain_last:]
        if not self.summarizer_fn:
            logger.warning("Summarization enabled, but no summarizer_fn provided. Skipping summarization.")
            return

        logger.info(f"Attempting to summarize. Current history length: {len(self.history)}")
        try:
            new_summary = self.summarizer_fn(self.history, self.summary)
            self.summary = new_summary
            # Retain only the last few messages post-summarization to avoid losing immediate context
            # Or clear completely if summarizer handles this. For now, we clear.
            self.history = [] 
            logger.info(f"Summarization complete. New summary: '{self.summary[:100]}...' History cleared.")
        except Exception as e:
            logger.error(f"Error during summarization: {e}", exc_info=True)


    def get_history_messages(self, window: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get history messages.
        Args:
            window (Optional[int]): Number of most recent messages to return. If None, returns all.
        """
        if window is not None and window > 0:
            return self.history[-window:]
        return list(self.history) # Return a copy

    def get_history_turns(self, window: Optional[int] = None) -> List[Tuple[Dict[str,str], Dict[str,str]]]:
        """
        Get history as (user_message, agent_message) turns.
        Args:
            window (Optional[int]): Number of most recent turns to return. If None, returns all.
        """
        turns = []
        i = 0
        while i < len(self.history):
            if self.history[i]["role"] == "user" and i + 1 < len(self.history) and self.history[i+1]["role"] == "agent":
                turns.append((self.history[i], self.history[i+1]))
                i += 2
            else: # Skip non-paired messages for turn view
                i += 1
        
        if window is not None and window > 0:
            return turns[-window:]
        return turns

    def get_summary(self) -> Optional[str]:
        """Returns the current summary."""
        return self.summary

    def get_context_for_prompt(self, include_summary: bool = True, include_history: bool = True, history_messages_window: Optional[int] = None) -> str:
        """
        Constructs a string representation of the context for prompting an LLM.
        Args:
            include_summary (bool): Whether to include the summary.
            include_history (bool): Whether to include the recent history.
            history_messages_window (Optional[int]): Number of recent messages to include from history.
        """
        parts = []
        if include_summary and self.summary:
            parts.append(f"[Summary]\n{self.summary}")
        
        if include_history and self.history:
            history_to_show = self.get_history_messages(window=history_messages_window)
            
            formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_to_show])
            if formatted_history:
                parts.append(f"[History]\n{formatted_history}")

        return "\n\n".join(parts)

    def update_environment_state(self, key: str, value: Any):
        """Updates a key in the environment_state."""
        self.environment_state[key] = value
        logger.debug(f"Environment state updated: {key} = {value}")

    def get_environment_state(self, key: Optional[str] = None) -> Any:
        """
        Gets a specific key from environment_state or the whole state dictionary.
        If key is provided and not found, returns None.
        """
        if key:
            return self.environment_state.get(key)
        return dict(self.environment_state) # Return a copy

    def update_slot(self, slot_name: str, value: Any):
        """Track filled conversation slots"""
        self.slot_values[slot_name] = value
        # Auto-remove from unresolved if filled
        self.unresolved_issues = [issue for issue in self.unresolved_issues 
                                if issue.get('slot') != slot_name]

    def add_unresolved(self, issue: str, slot: Optional[str] = None):
        """Track pending conversation threads (SINGLE IMPLEMENTATION)"""
        if not isinstance(issue, str) or len(issue.strip()) == 0:
            logger.error("Invalid unresolved issue format")
            return
            
        self.unresolved_issues.append({
            'description': issue,
            'slot': slot,
            'turn_number': len(self.history)
        })

    def register_intent(self, intent: str, confidence: float):
        """Record detected intents with metadata"""
        self.intent_history.append({
            'name': intent,
            'confidence': confidence,
            'turn': len(self.history),
            'timestamp': datetime.now().isoformat()
        })
        self.environment_state['last_intent'] = intent

    def detect_topic_shift(self, current_topic: str, threshold: float = 0.7) -> bool:
        from src.agents.perception.encoders.text_encoder import TextEncoder, load_config as load_perception_config
    
        config = load_perception_config()
        tokenizer = LanguageTokenizer()
        encoder = TextEncoder(config=config, tokenizer=tokenizer)
        encoder.eval()
    
        if not self.history:
            return False
    
        previous_topics = [msg['content'] for msg in self.history[-4:] if msg['role'] == 'user']
        if not previous_topics:
            return False
    
        current_vec = self._encode_text(current_topic, encoder)
        similarities = []
        for past in previous_topics:
            past_vec = self._encode_text(past, encoder)
            sim = F.cosine_similarity(current_vec, past_vec, dim=0).item()
            similarities.append(sim)
    
        return max(similarities) < threshold

    def _encode_text(self, text: str, encoder) -> torch.Tensor:
        """Helper to convert string to embedding using encoder"""
        cached = self.cache.get_embedding(text)
        if cached is not None:
            return cached
        if not encoder.tokenizer:
            raise ValueError("TextEncoder requires a tokenizer to embed text.")
        
        tokens = encoder.tokenizer.encode(text)["input_ids"]
        token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            output = encoder(token_tensor, style_id=0)
            if isinstance(output, tuple):
                # output is (hidden_states, hidden_states) — take the actual tensor
                encoded = output[0]  # shape: [batch_size, seq_len, embed_dim]
            else:
                encoded = output
            if isinstance(encoded, tuple):  # just in case encoded is again a tuple
                encoded = encoded[0]
            pooled = torch.mean(encoded, dim=1).squeeze(0)  # shape: [embed_dim]
    
        self.cache.add_embedding(text, pooled)
        return pooled

    @property
    def required_slots_filled(self) -> bool:
        return all(slot in self.slot_values for slot in self.required_slots)

    def get_missing_slots(self) -> List[str]:
        return [slot for slot in self.required_slots if slot not in self.slot_values]

    def update_user_profile(self, preferences: Dict):
        """Deep merge preferences"""
        existing = self.environment_state.get('user_preferences', {})
        existing.update(preferences)
        self.environment_state['user_preferences'] = existing

    def get_personalization_context(self) -> str:
        """Format personalization for prompts"""
        prefs = self.environment_state.get('user_preferences', {})
        return "\n".join([f"{k}: {v}" for k,v in prefs.items()])

    def save_state(self, file_path: Union[str, Path]):
        """Save context to JSON file"""
        state = {
            'history': self.history,
            'slots': self.slot_values,
            'intent_history': self.intent_history,
            'summary': self.summary
        }
        with open(file_path, 'w') as f:
            json.dump(state, f)

    @classmethod
    def load_state(cls, file_path: Union[str, Path]):
        """Load context from saved state"""
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        instance = cls()
        instance.history = state.get('history', [])
        instance.slot_values = state.get('slots', {})
        instance.intent_history = state.get('intent_history', [])
        instance.summary = state.get('summary', '')
        return instance

    def get_time_since_last_interaction(self) -> float:
        """Returns minutes since last message"""
        if not self.history:
            return 0.0
        last_msg_time = datetime.fromisoformat(self.history[-1].get('timestamp', datetime.now().isoformat()))
        return (datetime.now() - last_msg_time).total_seconds() / 60

    def is_follow_up(self, current_utterance: str) -> bool:
        patterns = self.follow_up_patterns_path 
        return any(re.search(pattern, current_utterance, re.IGNORECASE) 
               for pattern in patterns)

    def clear(self):
        """Clear the history, summary, and reset environment state and other tracking"""
        self.history = []
        self.summary = self.default_initial_summary or ""
        self.environment_state = self.dialogue_config.get(
            "initial_environment_state", 
            self.dialogue_config.get("default_initial_environment_state", 
                {"session_id": None, "user_preferences": {}, "last_intent": None}
            )
        )
        # Reset other state trackers
        self.slot_values = {}
        self.unresolved_issues = []
        self.intent_history = []   # Reset to empty list
        logger.info("DialogueContext cleared (full reset)")
 
    def serialize(self):
        return {
            "messages": self.messages,
            "slots": self.slots,
            "intent_history": self.intent_history,
            "environment_state": self.environment_state
        }
    
    def deserialize(self, data):
        self.messages = data.get("messages", [])
        self.slots = data.get("slots", {})
        self.intent_history = data.get("intent_history", [])
        self.environment_state = data.get("environment_state", {})

if __name__ == "__main__":
    print("\n=== Running Dialogue Context ===\n")
    printer.status("Init", "Dialogue Context initialized", "success")

    context = DialogueContext()

    print(context)

    print("\n* * * * * Phase 2 * * * * *\n")
    input="What is life about?"
    output="The meaning of life differse from person to person."
    printer.status("Init", context.add_turn(user_input=input, agent_response=output), "success")
    printer.status("Init", context._summarize(), "success")

    print("\n=== Finished Running Dialogue Context ===\n")
