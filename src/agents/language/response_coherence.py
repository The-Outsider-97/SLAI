from typing import List
from src.agents.language.grammar_processor import GrammarProcessor
from src.agents.language.resource_loader import ResourceLoader
from src.agents.language.nlp_utils import CoreferenceResolver
from src.agents.language_agent import DialogueContext


class ResponseCoherence:
    """Evaluate coherence between generated response and prior conversation context."""

    def __init__(self):
        self.structured_wordlist = ResourceLoader.get_structured_wordlist()
        self.grammar = GrammarProcessor(structured_wordlist=self.structured_wordlist)
        self.coref_resolver = CoreferenceResolver()

    def validate(self, response: str, context: DialogueContext) -> bool:
        
        """Check response relevance to context using lexical and syntactic overlap"""
        # Optionally resolve coreference in history
        resolved_history = [self.coref_resolver.resolve(turn, context.history) for turn in context.history]

        # Tokenize and POS tag
        response_tagged = self.grammar._pos_tag(response)
        history_tagged = [self.grammar._pos_tag(turn) for turn in resolved_history]

        # Extract content words (NOUN, PROPN, VERB, ADJ)
        content_tags = {"NOUN", "PROPN", "VERB", "ADJ"}
        response_content = {word for word, tag in response_tagged if tag in content_tags}
        history_content = {word for turn in history_tagged for word, tag in turn if tag in content_tags}

        # Compute relevance ratio
        overlap = response_content & history_content
        return len(overlap) / (len(response_content) + 1e-5) > 0.25  # Threshold tunable

    def coherence_score(self, response: str, context: DialogueContext) -> float:
        """Return a coherence score between 0 and 1 based on lexical & syntactic consistency"""
        resolved_history = [self.coref_resolver.resolve(turn, context.history) for turn in context.history]
        response_tagged = self.grammar._pos_tag(response)
        history_tagged = [self.grammar._pos_tag(turn) for turn in resolved_history]

        response_words = {word for word, _ in response_tagged}
        history_words = {word for turn in history_tagged for word, _ in turn}
        return len(response_words & history_words) / (len(response_words) + 1e-5)
