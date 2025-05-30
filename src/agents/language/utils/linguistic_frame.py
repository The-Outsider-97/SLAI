
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class SpeechActType(Enum):
    """Speech act classification based on Searle's taxonomy"""
    ASSERTIVE = "Representatives"  # Commit speaker to truth (e.g., assertions)
    DIRECTIVE = "Directives"       # Attempt to get listener to do something (e.g., requests)
    COMMISSIVE = "Commissives"     # Commit speaker to future action (e.g., promises)
    EXPRESSIVE = "Expressives"     # Express psychological state (e.g., apologies)
    DECLARATION = "Declarations"   # Change reality through utterance (e.g., "You're fired")

@dataclass
class LinguisticFrame:
    """Structured representation of language acts (inspired by Speech Act Theory)"""
    intent: str
    entities: Dict[str, Any] # Changed from str to Any to match usage
    sentiment: float  # Range [-1, 1]
    modality: str  # From Nuyts (2005) modality taxonomy
    confidence: float  # [0, 1]
    act_type: SpeechActType
    propositional_content: Optional[str] = None  # What the speech act is about
    illocutionary_force: Optional[str] = None   # Speaker's intended purpose (e.g., "request", "warn")
    perlocutionary_effect: Optional[str] = None # Intended listener impact (e.g., "persuade")

    # Validation
    def __post_init__(self):
        self.confidence = min(1.0, max(0.0, self.confidence))
        self.sentiment = min(1.0, max(-1.0, self.sentiment))
