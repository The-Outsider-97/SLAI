"""Pure-Python checks for degenerate generated text (loops, echoing, repeated replies, templated filler).

Dependency-free on purpose (no torch, no config): it is imported by the NLG engine on the response path and by
training/evaluation code.  Every function is deterministic.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

__all__ = [
    "degenerate_reasons",
    "find_templated_sentences",
    "generation_health",
    "has_repetition_loop",
    "restore_sentence_case",
]

_WORD = re.compile(r"[\w']+", re.UNICODE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_STANDALONE_I = re.compile(r"\bi(?=(?:'(?:m|ll|d|ve|s)\b)|\b)")
_SENTENCE_START = re.compile(r"(^|[.!?]\s+)([a-z])")


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def _norm(text: str) -> str:
    return " ".join(_words(text))


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def has_repetition_loop(text: str, *, unigram_repeats: int = 4, ngram_repeats: int = 3) -> bool:
    """True when a word repeats ``unigram_repeats`` times in a row, or a 2..4-word phrase repeats ``ngram_repeats`` times in a row."""
    words = _words(text)
    for n in (1, 2, 3, 4):
        need = unigram_repeats if n == 1 else ngram_repeats
        if len(words) < n * need:
            continue
        for start in range(0, len(words) - n * need + 1):
            gram = words[start:start + n]
            if all(words[start + k * n:start + (k + 1) * n] == gram for k in range(1, need)):
                return True
    return False


def restore_sentence_case(text: str) -> str:
    """Capitalise sentence starts and the pronoun "I" (incl. I'm/I'll/I'd/I've) in lower-cased text; wording is never changed."""
    if not text:
        return text
    out = _STANDALONE_I.sub("I", text)
    return _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), out)


def degenerate_reasons(response: str, *, user_text: str = "", previous_assistant: Iterable[str] = (),
                       near_copy: float = 0.9) -> List[str]:
    """Reasons a candidate reply should not be shown: ``empty``, ``repetition_loop``, ``echoes_user``,
    ``repeats_previous_assistant`` (exact or near-copy by word-set overlap >= ``near_copy``), ``low_lexical_diversity``."""
    reasons: List[str] = []
    words = _words(response)
    if not words:
        return ["empty"]
    if has_repetition_loop(response):
        reasons.append("repetition_loop")
    if len(words) >= 8 and len(set(words)) / len(words) < 0.35 and "repetition_loop" not in reasons:
        reasons.append("low_lexical_diversity")
    user_words = _words(user_text)
    if user_words and (words == user_words or (len(user_words) >= 3 and _jaccard(words, user_words) >= near_copy and len(words) <= len(user_words) + 2)):
        reasons.append("echoes_user")
    for prev in previous_assistant:
        prev_words = _words(prev)
        if prev_words and (words == prev_words or (len(prev_words) >= 3 and _jaccard(words, prev_words) >= near_copy)):
            reasons.append("repeats_previous_assistant")
            break
    return reasons


def find_templated_sentences(texts: Sequence[str], *, min_count: int = 20, min_fraction: float = 0.08,
                             min_words: int = 4) -> Dict[str, int]:
    """Sentences that recur across many *different* replies (filler). Returns {normalised sentence: number of replies containing it}.

    A sentence is reported when it appears in at least ``min_count`` replies and in at least ``min_fraction`` of all replies.
    Per-reply repeats count once, so one reply cannot make a sentence "templated" on its own."""
    total = len(texts)
    if total == 0:
        return {}
    counts: Counter = Counter()
    for text in texts:
        seen = {_norm(s) for s in _SENTENCE_SPLIT.split(text or "")}
        counts.update(s for s in seen if len(s.split()) >= min_words)
    floor = max(min_count, min_fraction * total)
    return {s: c for s, c in counts.items() if c >= floor}


def generation_health(prompts: Sequence[str], responses: Sequence[str]) -> Dict[str, float]:
    """Aggregate health of a batch of generations. ``penalty`` is in [0, 1] (0 = healthy); ``healthy`` is penalty < 0.25."""
    n = len(responses)
    if n == 0:
        return {"n": 0, "penalty": 1.0, "healthy": False, "loop_rate": 0.0, "echo_rate": 0.0, "empty_rate": 1.0, "unique_ratio": 0.0}
    loops = echoes = empties = 0
    for i, resp in enumerate(responses):
        prompt = prompts[i] if i < len(prompts) else ""
        reasons = degenerate_reasons(resp, user_text=prompt)
        loops += "repetition_loop" in reasons or "low_lexical_diversity" in reasons
        echoes += "echoes_user" in reasons
        empties += "empty" in reasons
    unique_ratio = len({_norm(r) for r in responses}) / n
    loop_rate, echo_rate, empty_rate = loops / n, echoes / n, empties / n
    penalty = min(1.0, loop_rate + echo_rate + empty_rate + 0.5 * (1.0 - unique_ratio))
    return {"n": n, "penalty": round(penalty, 6), "healthy": penalty < 0.25, "loop_rate": loop_rate, "echo_rate": echo_rate,
            "empty_rate": empty_rate, "unique_ratio": unique_ratio}
