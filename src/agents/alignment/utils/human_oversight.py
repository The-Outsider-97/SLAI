
import time
import threading

from typing import Dict, List

class HumanOversightTimeout(Exception):
    """Raised when a human oversight request times out."""

    def __init__(self, message="Human oversight response timed out.", timeout_seconds=None):
        if timeout_seconds:
            message += f" Timeout was set to {timeout_seconds} seconds."
        super().__init__(message)
        self.timeout_seconds = timeout_seconds

class HumanOversightInterface:
    """
    Interface for managing human oversight in the alignment system.

    Supports:
    - Manual approval or rejection of model decisions.
    - Injecting human preferences in ambiguous cases.
    - Timeout and fallback handling for unattended requests.
    """

    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
        self.response = None
        self._lock = threading.Lock()

    def request_approval(self, context: Dict) -> bool:
        """
        Request human approval for a model decision.

        Parameters:
        - context (Dict): Information needed for human judgment
                          (e.g., alignment score, input text, predicted preference)

        Returns:
        - bool: True if approved, False if rejected

        Raises:
        - HumanOversightTimeout: if no response is received in time
        """
        print("\n[HUMAN OVERSIGHT REQUIRED]")
        print("Context:", context)
        print("Please type 'approve' or 'reject':")

        def wait_for_input():
            self.response = input().strip().lower()

        input_thread = threading.Thread(target=wait_for_input)
        input_thread.start()
        input_thread.join(timeout=self.timeout_seconds)

        if self.response is None:
            raise HumanOversightTimeout(timeout_seconds=self.timeout_seconds)

        return self.response == "approve"

    def inject_preference(self, options: List[str]) -> str:
        """
        Prompt human to select the most aligned option from a list.

        Parameters:
        - options (List[str]): Options to choose from

        Returns:
        - str: The selected option
        """
        print("\n[HUMAN PREFERENCE INJECTION]")
        print("Choose the most ethically appropriate response:")
        for idx, option in enumerate(options):
            print(f"{idx}: {option}")

        while True:
            try:
                selected = int(input("Enter the number of your choice: "))
                if 0 <= selected < len(options):
                    return options[selected]
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a valid number.")
