"""Torch-free response-quality utilities shared by LANTRA training and inference.

One implementation is used by every component that has to decide whether a
conversational response is healthy:

* the teacher builder (``src/training/supervised_llm_builder.py``) rejects
  boilerplate and near-duplicate responses before they become training data;
* the trainer (``train_lantra.py``) audits supervised data for templated filler
  and scores generation health so checkpoints that collapse onto one canned
  answer are not selected;
* the NLG engine rejects looping / echoing / repeated LANTRA candidates at run
  time.

Nothing here imports torch, numpy or any SLAI runtime, so it is safe to import
from lightweight scripts and from tests.

Design note: filler is detected *statistically*, never by a hard-coded phrase
list. A sentence that is appended after many unrelated answers ("Start with the
part that matters most to you.") has a distinctive signature: it occurs many
times as a non-initial sentence and follows many *different* preceding
sentences. Legitimate greeting continuations ("Hi there! What can I help you
with?") occur often too, but always after the same few openers, so they are not
flagged.
"""
from __future__ import annotations

import collections
import math
import re

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W\d_]+)?", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def words(text: Any) -> List[str]:
    """Unicode alphanumeric spans; this is not language-specific word segmentation."""
    return _WORD_RE.findall(str(text or "").lower())


def normalize(text: Any) -> str:
    return " ".join(words(text))


def split_sentences(text: Any) -> List[str]:
    value = " ".join(str(text or "").split())
    if not value:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(value) if part.strip()]


def sentence_key(sentence: str) -> str:
    return normalize(sentence)


# ---------------------------------------------------------------------------
# Corpus-level template (filler) detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TemplateFinding:
    sentence: str
    count: int
    share: float
    distinct_contexts: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence": self.sentence,
            "count": self.count,
            "share": round(self.share, 4),
            "distinct_contexts": self.distinct_contexts,
        }


def find_templated_sentences(
    responses: Iterable[str],
    *,
    min_count: int = 8,
    max_share: float = 0.02,
    min_contexts: int = 6,
    min_words: int = 3,
) -> Dict[str, TemplateFinding]:
    """Find sentences that are appended to many unrelated responses.

    A non-initial sentence is reported when it

    * occurs at least ``min_count`` times and in at least ``max_share`` of all
      responses, and
    * follows at least ``min_contexts`` *different* preceding sentences.

    Returns ``{sentence_key: TemplateFinding}``.
    """
    corpus = [split_sentences(item) for item in responses]
    total = len(corpus)
    if total == 0:
        return {}
    counts: collections.Counter[str] = collections.Counter()
    contexts: Dict[str, set[str]] = collections.defaultdict(set)
    display: Dict[str, str] = {}
    for sentences in corpus:
        seen_in_response: set[str] = set()
        for index, sentence in enumerate(sentences):
            if index == 0:
                continue
            key = sentence_key(sentence)
            if len(key.split()) < min_words or key in seen_in_response:
                continue
            seen_in_response.add(key)
            counts[key] += 1
            contexts[key].add(sentence_key(sentences[index - 1]))
            display.setdefault(key, sentence)
    findings: Dict[str, TemplateFinding] = {}
    for key, count in counts.items():
        share = count / total
        if count >= min_count and share >= max_share and len(contexts[key]) >= min_contexts:
            findings[key] = TemplateFinding(display[key], count, share, len(contexts[key]))
    return findings


def strip_templated_sentences(text: str, templated: Iterable[str]) -> str:
    """Remove flagged non-initial sentences. Returns ``text`` unchanged if nothing remains."""
    banned = set(templated)
    if not banned:
        return text
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return text
    kept = [sentences[0]] + [s for s in sentences[1:] if sentence_key(s) not in banned]
    return " ".join(kept) if kept else text


def contains_templated(text: str, templated: Iterable[str]) -> Optional[str]:
    banned = set(templated)
    if not banned:
        return None
    for index, sentence in enumerate(split_sentences(text)):
        if index > 0 and sentence_key(sentence) in banned:
            return sentence
    return None


# ---------------------------------------------------------------------------
# Single-response degeneracy
# ---------------------------------------------------------------------------
def repetition_ratio(tokens: Sequence[str]) -> float:
    """Fraction of tokens that repeat an earlier token (0 = all unique)."""
    if not tokens:
        return 0.0
    return 1.0 - len(set(tokens)) / len(tokens)


def has_repetition_loop(text: Any) -> bool:
    """True for ``meta meta meta ...`` style loops and repeated n-grams."""
    tokens = words(text)
    if len(tokens) < 4:
        return False
    run = 1
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if current == previous else 1
        if run >= 4:
            return True
    if len(tokens) >= 8 and repetition_ratio(tokens) >= 0.7:
        return True
    for size in (2, 3, 4):
        if len(tokens) < size * 3:
            continue
        grams = collections.Counter(tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1))
        if grams and max(grams.values()) >= 3:
            return True
    return False


def token_jaccard(a: Any, b: Any) -> float:
    left, right = set(words(a)), set(words(b))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def degenerate_reasons(
    response: str,
    *,
    user_text: str = "",
    previous_assistant: Sequence[str] = (),
    templated: Iterable[str] = (),
    echo_threshold: float = 0.9,
    repeat_threshold: float = 0.8,
) -> List[str]:
    """Reasons a candidate response must not be shown / trained on (empty = healthy)."""
    reasons: List[str] = []
    if not words(response):
        return ["empty"]
    if has_repetition_loop(response):
        reasons.append("repetition_loop")
    if user_text and len(words(user_text)) >= 3 and token_jaccard(response, user_text) >= echo_threshold:
        reasons.append("echoes_user")
    for previous in previous_assistant:
        if previous and token_jaccard(response, previous) >= repeat_threshold:
            reasons.append("repeats_previous_assistant")
            break
    flagged = contains_templated(response, templated)
    if flagged:
        reasons.append(f"template_filler:{flagged}")
    return reasons


# ---------------------------------------------------------------------------
# Generation health (used for checkpoint selection and probes)
# ---------------------------------------------------------------------------
def _distinct_n(outputs: Sequence[str], n: int) -> float:
    grams: List[Tuple[str, ...]] = []
    for output in outputs:
        tokens = words(output)
        grams.extend(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return len(set(grams)) / len(grams) if grams else 0.0


def generation_health(
    prompts: Sequence[str],
    outputs: Sequence[str],
    *,
    templated: Iterable[str] = (),
) -> Dict[str, Any]:
    """Summarise whether a model gives distinct, non-degenerate, input-sensitive answers.

    ``penalty`` is in ``[0, 1]`` (0 = healthy). It is deliberately dominated by
    *collapse* signals - one canned answer for many prompts, loops, empties -
    because those are exactly the failures that token-level validation loss cannot see.
    """
    templated = set(templated)
    count = len(outputs)
    if count == 0:
        return {"examples": 0, "penalty": 1.0}
    normalized = [normalize(item) for item in outputs]
    empty = sum(1 for item in normalized if not item)
    loops = sum(1 for item in outputs if has_repetition_loop(item))
    echoes = sum(
        1 for prompt, output in zip(prompts, outputs)
        if len(words(prompt)) >= 3 and token_jaccard(prompt, output) >= 0.9
    )
    filler = sum(1 for item in outputs if contains_templated(item, templated))
    most_common = collections.Counter(normalized).most_common(1)[0][1]
    unique_ratio = len(set(normalized)) / count
    top_share = most_common / count
    # With only a handful of prompts a repeated greeting is legitimate; require
    # a minimum sample before treating duplicates as collapse.
    collapse = max(0.0, 1.0 - unique_ratio) if count >= 8 else 0.0
    penalty = (
        0.40 * collapse
        + 0.25 * (loops / count)
        + 0.20 * (empty / count)
        + 0.10 * (filler / count)
        + 0.05 * (echoes / count)
    )
    return {
        "examples": count,
        "unique_ratio": round(unique_ratio, 4),
        "top_answer_share": round(top_share, 4),
        "empty_rate": round(empty / count, 4),
        "loop_rate": round(loops / count, 4),
        "echo_rate": round(echoes / count, 4),
        "filler_rate": round(filler / count, 4),
        "distinct_1": round(_distinct_n(outputs, 1), 4),
        "distinct_2": round(_distinct_n(outputs, 2), 4),
        "mean_words": round(sum(len(words(item)) for item in outputs) / count, 2),
        "penalty": round(min(1.0, penalty), 4),
    }


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
_I_CONTRACTION_RE = re.compile(r"\bi(?=(?:'(?:m|ll|ve|d))?\b)")


def restore_sentence_case(text: str) -> str:
    """Re-capitalise sentence starts and the pronoun *I*.

    LANTRA's tokenizer lower-cases its input, so decoded text is all lower case.
    This is presentation only and never changes wording.
    """
    value = " ".join(str(text or "").split())
    if not value:
        return value
    value = _I_CONTRACTION_RE.sub("I", value)
    # Preserve supplied technical/proper-name casing (pH, iPhone). A lowercased
    # model output cannot recover such casing without lexical evidence.
    mixed_case_starts = {
        match.start() for match in re.finditer(r"[^\W\d_]+", value)
        if any(char.isupper() for char in match.group())
        and any(char.islower() for char in match.group())
    }
    out: List[str] = []
    capitalize_next = True
    for index, char in enumerate(value):
        if capitalize_next and char.isalpha():
            out.append(char if index in mixed_case_starts else char.upper())
            capitalize_next = False
        else:
            out.append(char)
        if char in ".!?":
            capitalize_next = True
        elif char.isalnum():
            capitalize_next = False
    return "".join(out)


__all__ = [
    "TemplateFinding",
    "contains_templated",
    "degenerate_reasons",
    "find_templated_sentences",
    "generation_health",
    "has_repetition_loop",
    "normalize",
    "repetition_ratio",
    "restore_sentence_case",
    "sentence_key",
    "split_sentences",
    "strip_templated_sentences",
    "token_jaccard",
    "words",
]
