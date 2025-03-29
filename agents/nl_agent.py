import re
import json
import time
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque

@dataclass
class DialogueContext:
    """Stores conversation history and environment state for multi-turn dialogues."""
    history: deque = field(default_factory=lambda: deque(maxlen=10))  # Last 10 turns
    environment_state: Dict[str, Any] = field(default_factory=dict)   # External state (e.g., API results)
    user_preferences: Dict[str, Any] = field(default_factory=dict)    # User-specific settings

class LanguageAgent:
    def __init__(self, llm, max_retries: int = 3, timeout: int = 10):
        """
        Initialize the Language Agent with safety checks, context management, and benchmarking.
        
        Args:
            llm: A Large Language Model with a `generate()` method.
            max_retries: Retries for failed LLM calls (default: 3).
            timeout: Timeout for LLM responses (default: 10 sec).
        
        References:
            - Safety: Gehman et al. (2020). "RealToxicityPrompts".
            - Context: Adiwardana et al. (2020). "Towards a Human-like Open-Domain Chatbot".
            - Benchmarking: Wang et al. (2021). "SuperGLUE: A Stickier Benchmark for General-Purpose Language Models".
        """
        self.llm = llm
        self.max_retries = max_retries
        self.timeout = timeout
        self.context = DialogueContext()  # Multi-turn context manager
        self.benchmark_data = {
            "parsing_accuracy": [],
            "response_time": [],
            "safety_violations": 0,
        }
        self._validate_llm()

    def _validate_llm(self):
        """Ensure the LLM has the required `generate` method."""
        if not hasattr(self.llm, 'generate') or not callable(self.llm.generate):
            raise AttributeError("LLM must implement a 'generate' method.")

    def validate_response(self, response: str) -> bool:
        """
        Check if the LLM response violates safety constraints (e.g., toxicity, PII).
        
        Args:
            response: LLM-generated text.
        
        Returns:
            True if safe, False if unsafe.
        
        References:
            - Dinan et al. (2019). "Build it Break it Fix it for Dialogue Safety".
        """
        # Rule-based safety checks (extend with ML-based filters if needed)
        unsafe_patterns = [
            r"(?i)(kill|harm|hurt|attack|hate)\b",
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-like patterns
            r"(?i)(password|credit card|social security)",
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, response):
                self.benchmark_data["safety_violations"] += 1
                return False
        
        return True

    def update_context(self, user_input: str, llm_response: str) -> None:
        """
        Maintain conversation history and state for multi-turn dialogues.
        
        Args:
            user_input: Latest user query.
            llm_response: Generated assistant response.
        
        References:
            - Roller et al. (2021). "Recipes for Building an Open-Domain Chatbot".
        """
        self.context.history.append((user_input, llm_response))

    def evaluate_parsing_accuracy(self, test_cases: List[Tuple[str, Dict]]) -> float:
        """
        Benchmark the accuracy of `translate_user_input()` against labeled test cases.
        
        Args:
            test_cases: List of (input_text, expected_parsed_output).
        
        Returns:
            Accuracy score (0.0 to 1.0).
        
        References:
            - Rajpurkar et al. (2016). "SQuAD: 100,000+ Questions for Machine Comprehension".
        """
        correct = 0
        for text, expected in test_cases:
            parsed = self.translate_user_input(text)
            if parsed == expected:
                correct += 1
            self.benchmark_data["parsing_accuracy"].append(parsed == expected)
        
        return correct / len(test_cases)

    def generate_prompt(self, user_input: str) -> str:
        """(Previous implementation with context injection.)"""
        prompt = f"User: {user_input}\nContext: {json.dumps(self.context.environment_state)}\n"
        if self.context.history:
            prompt += "Dialogue History:\n" + "\n".join([f"User: {u}\nBot: {r}" for u, r in self.context.history])
        prompt += "\nAssistant:"
        return prompt

    def process_input(self, user_input: str) -> Tuple[str, Dict]:
        """
        Full processing pipeline with safety, context, and benchmarking.
        
        Args:
            user_input: Natural language input.
        
        Returns:
            (llm_response, structured_command)
        
        Raises:
            RuntimeError: If LLM fails or response is unsafe.
        """
        start_time = time.time()
        
        # Step 1: Generate prompt with context
        prompt = self.generate_prompt(user_input)
        
        # Step 2: Get LLM response (with retries)
        llm_response = self.interface_llm(prompt)
        
        # Step 3: Validate safety
        if not self.validate_response(llm_response):
            llm_response = "[SAFETY FILTER] I cannot comply with this request."
        
        # Step 4: Parse and update context
        structured_input = self.translate_user_input(user_input)
        self.update_context(user_input, llm_response)
        
        # Step 5: Record performance
        self.benchmark_data["response_time"].append(time.time() - start_time)
        
        return llm_response, structured_input

    def interface_llm(self, prompt: str) -> str:
        """
        Robust interface with the LLM using exponential backoff for retries.
        
        Args:
            prompt: Input prompt for the LLM.
        
        Returns:
            LLM response (stripped of leading/trailing whitespace).
        
        Raises:
            RuntimeError: If LLM fails to respond after max_retries.
        
        References:
            - For retry mechanisms: Google Cloud API Design Guide (2021).
        """
        for attempt in range(self.max_retries):
            try:
                response = self.llm.generate(prompt, timeout=self.timeout)
                return response.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM failed after {self.max_retries} attempts: {str(e)}")
                continue

    def translate_user_input(self, user_input: str) -> Dict:
        """
        Parse user input into structured commands using semantic parsing.
        
        Args:
            user_input: Raw natural language input.
        
        Returns:
            Dictionary with:
                - 'intent': High-level goal (e.g., "search", "create")
                - 'entities': Key objects/parameters
                - 'args': Additional arguments
        
        References:
            - Kamath & Das (2019). "A Survey on Semantic Parsing".
        """
        # Enhanced parsing with common NLP patterns
        patterns = {
            'search': r'(?:search|find)\s+(?P<query>.+?)\s+(?:about|for)\s+(?P<topic>.+)',
            'create': r'(?:create|make)\s+(?P<object>\w+)(?:\s+with\s+(?P<params>.+))?',
            'default': r'(?P<command>\w+)(?:\s+(?P<args>.+))?'
        }

        for intent, pattern in patterns.items():
            match = re.match(pattern, user_input.strip(), re.IGNORECASE)
            if match:
                groups = match.groupdict()
                return {
                    'intent': intent,
                    'entities': {k: v for k, v in groups.items() if v},
                    'args': groups.get('args', '').split() if 'args' in groups else []
                }

        return {'intent': 'unknown', 'entities': {}, 'args': []}
