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

from src.agents.language.utils.language_cache import LanguageCache
from logs.logger import get_logger

logger = get_logger("Dialogue Context")

CONFIG_PATH = "src/agents/language/configs/language_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class DialogueContext:
    def __init__(self, config):
        """
        Manages dialogue memory.

        Args:
            user_config_override (Optional[Dict]): User-specific configuration overrides.
                                                   These are merged with the base config from CONFIG_PATH.
                                                   Typically, this would be the 'dialogue_context_settings' part
                                                   of a larger agent configuration.
            summarizer_fn (Optional[Callable]): Custom summarizer function.
                                                Expected signature: (history_messages, current_summary) -> new_summary
        """
        self.config = config
        merged_cfg = get_merged_config()
        config = merged_cfg.get("dialogue_context_settings", {})
        self.history: List[Dict[str, str]] = []
        self.intent_history: List[Dict] = []
        self._initialize_history(config)

        self.summary: Optional[str] = config.get("initial_summary",
                                                 config.get("default_initial_summary", "The conversation has just begun."))
        self.environment_state: Dict[str, Any] = config.get("initial_environment_state",
                                                            config.get("default_initial_environment_state",
                                                                       {"session_id": None, "user_preferences": {}, "last_intent": None}))

        self.memory_limit = self.config.get("dialogue_context_settings", {}).get("memory_limit")
        self.enable_summarization = self.config.get("dialogue_context_settings", {}).get("enable_summarization")
        self.summarizer_fn: Optional[Callable[[List[Dict[str, str]], Optional[str]], str]] = {}

        # Slot/Entity Tracking System
        self.slot_values: Dict[str, Any] = {}
        self.unresolved_issues: List[Dict] = []
        self.required_slots = merged_cfg.get("required_slots", [])

        cache_cfg = merged_cfg.get("language_cache", {})
        self.cache = LanguageCache(config=cache_cfg)

        logger.info(f"DialogueContext initialized. Memory limit: {self.memory_limit}, Summarization: {self.enable_summarization}")

    def _initialize_history(self, config: Dict):
        initial_history_raw = config.get("initial_history", [])
        if not initial_history_raw: # If initial_history is explicitly empty or not provided, use default
             default_history_raw = config.get("default_initial_history", ["System: Hello! How can I assist you today?"])
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

    def add_turn(self, user_input: str, agent_response: str):
        """Adds a user input and agent response as a turn."""
        self.add_message("user", user_input)
        self.add_message("agent", agent_response) # Or assistant, bot, etc.

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
        """Track pending conversation threads"""
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

    def _encode_text(self, text: str, encoder) -> torch.Tensor:
        """Helper to convert string to embedding using encoder"""
        cached = self.cache.get_embedding(text)
        if cached is not None:
            return cached
        if not encoder.tokenizer:
            raise ValueError("TextEncoder requires a tokenizer to embed text.")
        
        tokens = encoder.tokenizer.encode(text)["input_ids"]
        token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
    
        with torch.no_grad():
            encoded, _ = encoder(token_tensor, style_id=0)
            pooled = torch.mean(encoded, dim=1).squeeze(0)  # Convert to 1D tensor [embed_dim]
    
        self.cache.add_embedding(text, pooled)
        return pooled

    def detect_topic_shift(self, current_topic: str, threshold: float = 0.7) -> bool:
        from src.agents.perception.encoders.text_encoder import TextEncoder, load_config as load_perception_config
        from src.agents.perception.modules.tokenizer import Tokenizer
    
        config = load_perception_config()
        tokenizer = Tokenizer(config)
        encoder = TextEncoder(config=config, tokenizer=tokenizer)
        encoder.eval()
    
        if not self.history:
            return False
    
        previous_topics = [msg['content'] for msg in self.history[-4:] if msg['role'] == 'user']
        if not previous_topics:
            return False
    
        current_vec = self._encode_text(current_topic, encoder)
    
        current_vec = self._encode_text(current_topic, encoder)
        similarities = []
        for past in previous_topics:
            past_vec = self._encode_text(past, encoder)
            sim = F.cosine_similarity(current_vec, past_vec, dim=0).item()
            similarities.append(sim)
    
        return max(similarities) < threshold

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
    def load_state(cls, config, file_path: Union[str, Path]):
        """Load context from saved state"""
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        instance = cls(config)
        instance.history = state.get('history', [])
        instance.slot_values = state.get('slots', {})
        instance.intent_history = state.get('intent_history', [])
        instance.summary = state.get('summary', '')
        return instance

    def add_unresolved(self, issue: str, slot: Optional[str] = None):
        if not isinstance(issue, str) or len(issue.strip()) == 0:
            logger.error("Invalid unresolved issue format")
            return

    def get_time_since_last_interaction(self) -> float:
        """Returns minutes since last message"""
        if not self.history:
            return 0.0
        last_msg_time = datetime.fromisoformat(self.history[-1].get('timestamp', datetime.now().isoformat()))
        return (datetime.now() - last_msg_time).total_seconds() / 60

    def is_follow_up(self, current_utterance: str) -> bool:
        patterns = self.config.get("dialogue_context_settings", {}).get("follow_up_patterns_path")
        return any(re.search(pattern, current_utterance, re.IGNORECASE) 
               for pattern in patterns)

    def clear(self):
        """Clear the history, summary, and optionally reset environment state to defaults."""
        self.history = []
        # Re-fetch default config for summary and env_state to reset them
        merged_cfg = get_merged_config(None) # No override, just base config
        config = merged_cfg.get("dialogue_context_settings", {})

        self.summary = config.get("initial_summary", 
                                  config.get("default_initial_summary", "The conversation has just begun."))
        self.environment_state = config.get("initial_environment_state", 
                                            config.get("default_initial_environment_state", 
                                                       {"session_id": None, "user_preferences": {}, "last_intent": None}))
        logger.info("DialogueContext cleared (history, summary, environment_state reset to defaults).")
 
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

    config =  {}

    context = DialogueContext(config=config)

    print(context)
    print("\n=== Finished Running Dialogue Context ===\n")

if __name__ == "__main__":
    print("=== Dialogue Context Interactive Test ===")
    
    # Load test configuration
    test_config = {
        "dialogue_context_settings": {
            "memory_limit": 3,
            "required_slots": ["destination", "travel_dates", "budget"],
            "follow_up_patterns_path": "src/agents/language/templates/follow_up_patterns.json",
            "enable_summarization": True
        }
    }

    # Initialize context
    context = DialogueContext(config=test_config)
    
    # Test 1: Basic conversation flow
    print("\n--- Test 1: Conversation History ---")
    context.add_turn("I want to plan a trip to Japan", "Great! When would you like to go?")
    context.add_turn("In spring next year", "Spring is lovely there. For how long?")
    print("History:")
    for msg in context.get_history_messages():
        print(f"{msg['role'].upper()}: {msg['content']}")

    # Test 2: Slot filling
    print("\n--- Test 2: Slot Management ---")
    context.update_slot("destination", "Japan")
    print(f"Filled slots: {context.slot_values}")
    print(f"Missing slots: {context.get_missing_slots()}")

    # Test 3: Follow-up detection
    print("\n--- Test 3: Follow-up Detection ---")
    follow_up_utterance = "Regarding our previous discussion about destinations..."
    is_follow = context.is_follow_up(follow_up_utterance)
    print(f"'{follow_up_utterance}'\nIs follow-up? {is_follow}")

    # Test 4: Topic shift detection
    print("\n--- Test 4: Topic Detection ---")
    new_topic = "What's the weather forecast for tomorrow?"
    is_shift = context.detect_topic_shift(new_topic)
    print(f"'{new_topic}'\nIs topic shift? {is_shift}")

    # Test 5: Summarization
    print("\n--- Test 5: Summarization ---")
    # Add mock summarizer
    context.summarizer_fn = lambda history, summary: f"{summary} [Summarized {len(history)} messages]"
    # Exceed memory limit
    context.add_turn("About 2 weeks", "What's your budget?")
    context.add_turn("Around $5000", "Great! Let's look at options...")
    print(f"Current summary: {context.get_summary()}")
    print(f"History length after summarization: {len(context.history)}")

    # Test 6: Persistence
    print("\n--- Test 6: Save/Load ---")
    context.save_state("src/agents/language/cache/test_session.json")
    loaded_context = DialogueContext.load_state(test_config, "src/agents/language/cache/test_session.json")
    print(f"Loaded history: {len(loaded_context.history)} messages")
    print(f"Loaded slots: {loaded_context.slot_values}")

    # Test 7: Clear functionality
    print("\n--- Test 7: Clear Context ---")
    context.clear()
    print(f"Summary after clear: {context.get_summary()}")
    print(f"Slots after clear: {context.slot_values}")

    print("\n=== All Tests Completed ===")
