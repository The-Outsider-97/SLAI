
import json

from typing import Optional, Any

from src.agents.language.utils.linguistic_frame import LinguisticFrame
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Language Error")
printer = PrettyPrinter

class NLGFillingError(Exception):
    """Raised when template filling fails due to missing or malformed entities."""
    def __init__(self, message: str, intent: str = None, template: str = None, entity_data: dict = None):
        self.intent = intent
        self.template = template
        self.entity_data = entity_data
        super().__init__(f"{message} | intent: {intent} | template: {template} | entity_data: {entity_data}")


class NLGValidationError(Exception):
    """Raised when the generated response does not meet validation criteria."""
    def __init__(self, message: str, response_text: str = None, expected_format: str = None):
        self.response_text = response_text
        self.expected_format = expected_format
        super().__init__(f"{message} | response_text: {response_text} | expected_format: {expected_format}")


class TemplateNotFoundError(Exception):
    """Raised when a template for the given intent is missing."""
    def __init__(self, message: str, intent: str = None, templates: dict = None):
        self.intent = intent
        self.templates = templates
        super().__init__(f"{message} | intent: {intent} | available_templates: {list(templates.keys()) if templates else 'None'}")

class NLGGenerationError(Exception):
    """Raised when neural language generation fails"""
    def __init__(self, message: str, 
                 prompt: Optional[str] = None, 
                 frame: Optional[LinguisticFrame] = None, 
                 context: Optional[Any] = None,
                 error_type: Optional[str] = None,
                 original_exception: Optional[Exception] = None):
        
        self.message = message
        self.prompt = prompt
        self.frame = frame
        self.context = context
        self.error_type = error_type
        self.original_exception = original_exception
        
        # Extract frame details for logging
        frame_details = {}
        if frame:
            frame_details = {
                "intent": frame.intent,
                "act_type": frame.act_type.value if frame.act_type else None,
                "sentiment": frame.sentiment,
                "modality": frame.modality
            }
        
        # Context summary
        context_summary = None
        if context and hasattr(context, 'get_context_summary'):
            try:
                context_summary = context.get_context_summary()
            except Exception:
                context_summary = "Unable to retrieve context summary"
        
        # Build detailed error message
        details = [
            f"Error Type: {error_type or 'Unspecified'}",
            f"Original Exception: {type(original_exception).__name__ if original_exception else 'None'}",
            f"Prompt Length: {len(prompt) if prompt else 0} chars",
            f"Frame Details: {json.dumps(frame_details)}",
            f"Context Summary: {context_summary or 'None'}"
        ]
        
        super().__init__(f"NLG Generation Error: {message} | Details: {' | '.join(details)}")
