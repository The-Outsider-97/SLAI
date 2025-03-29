import re
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
from collections import deque

@dataclass
class DialogueContext:
    """Stores conversation history and environment state for multi-turn dialogues."""
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    environment_state: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)

class LanguageAgent:
    def __init__(self, llm, max_retries: int = 3, timeout: int = 15):
        self.llm = llm
        self.max_retries = max_retries
        self.timeout = timeout
        self.context = DialogueContext()
        self.benchmark_data = {
            "parsing_accuracy": [],
            "response_time": [],
            "safety_violations": 0,
        }
        self._validate_llm()

    def save_context(self, file_path: Union[str, Path]) -> None:
        """
        Save the current dialogue context to a file using pickle.
        
        Args:
            file_path: Path to save the context (e.g., "context.pkl").
        
        Example:
            >>> agent.save_context("chat_context.pkl")
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.context, f)

    def load_context(self, file_path: Union[str, Path]) -> None:
        """
        Load dialogue context from a pickle file.
        
        Args:
            file_path: Path to the saved context file.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            pickle.UnpicklingError: If the file is corrupted.
        
        Example:
            >>> agent.load_context("chat_context.pkl")
        """
        try:
            with open(file_path, 'rb') as f:
                self.context = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Context file not found: {file_path}")
        except pickle.UnpicklingError:
            raise pickle.UnpicklingError(f"Failed to load context from {file_path}")

    def _validate_llm(self) -> None:
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
        
        Notes:
            Extend with ML-based classifiers (e.g., Hugging Face's `detoxify`) for production.
        """
        unsafe_patterns = [
            r"(?i)\b(kill|harm|hurt|attack|hate)\b",
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-like patterns
            r"(?i)\b(password|credit\s*card|social\s*security)\b",
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, response):
                self.benchmark_data["safety_violations"] += 1
                return False
        return True

    def update_context(self, user_input: str, llm_response: str) -> None:
        """Update dialogue history and environment state."""
        self.context.history.append((user_input, llm_response))

    def evaluate_parsing_accuracy(self, test_cases: List[Tuple[str, Dict]]) -> float:
        """
        Benchmark parsing accuracy against labeled test cases.
        
        Args:
            test_cases: List of (input_text, expected_parsed_output).
        
        Returns:
            Accuracy score (0.0 to 1.0).
        """
        correct = 0
        for text, expected in test_cases:
            parsed = self.translate_user_input(text)
            if parsed == expected:
                correct += 1
            self.benchmark_data["parsing_accuracy"].append(parsed == expected)
        return correct / len(test_cases)

    def generate_prompt(self, user_input: str) -> str:
        """Generate a prompt with context and history."""
        prompt = f"User: {user_input}\nContext: {json.dumps(self.context.environment_state)}\n"
        if self.context.history:
            prompt += "Dialogue History:\n" + "\n".join([f"User: {u}\nBot: {r}" for u, r in self.context.history])
        prompt += "\nAssistant:"
        return prompt

    def process_input(self, user_input: str) -> Tuple[str, Dict]:
        """
        End-to-end processing pipeline.
        
        Returns:
            Tuple of (LLM response, structured command).
        
        Raises:
            RuntimeError: If LLM fails or response is unsafe.
        """
        start_time = time.time()
        
        # Step 1: Generate prompt
        prompt = self.generate_prompt(user_input)
        
        # Step 2: Get LLM response
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
        """Robust LLM interface with retries and timeout."""
        for attempt in range(self.max_retries):
            try:
                response = self.llm.generate(prompt, timeout=self.timeout)
                return response.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM failed after {self.max_retries} attempts: {str(e)}")
                continue

    def translate_user_input(self, user_input: str) -> Dict:
        """Parse user input into structured commands."""
        patterns = {
            'search': r'(?:search|find)\s+(?P<query>.+?)(?:\s+(?:about|for)\s+(?P<topic>.+))?',
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
