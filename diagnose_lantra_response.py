from __future__ import annotations

import argparse
import json

from src.agents.language.lantra_runtime import LantraRuntime
from src.agents.language.nlg_engine import NLGEngine
from src.agents.language.utils.linguistic_frame import LinguisticFrame, SpeechActType


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose LANTRA -> NLG response flow.")
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    message = args.message.strip()
    runtime = LantraRuntime()
    nlg = NLGEngine()

    encoded = runtime.tokenizer.encode(message, add_special_tokens=True)
    decoded = runtime.tokenizer.decode(
        encoded["input_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()
    print(f"TOKENIZER ROUNDTRIP: {decoded!r}")

    intent = nlg._match_intent_by_trigger(message)
    frame = LinguisticFrame(
        intent=intent,
        entities={},
        sentiment=0.0,
        modality="epistemic",
        confidence=1.0,
        act_type=SpeechActType.ASSERTIVE,
        propositional_content=message,
    )

    context = {
        "history": [{"role": "user", "content": message}],
        "summary": "",
        "relevant_context": message,
        "slots": {},
        "environment": {},
        "unresolved_issues": [],
    }

    candidate = runtime.nlg_generate("", frame, context).strip()
    relevance = None

    if candidate:
        try:
            ranked = runtime.rerank(message, [candidate], top_k=1)
            if ranked.candidates:
                relevance = float(ranked.candidates[0].score)
        except Exception:
            pass

    result = nlg.generate_detailed(
        frame,
        context=context,
        neural_candidate=candidate or None,
        neural_relevance=relevance,
    )

    print(json.dumps({
        "mode": result.generation_mode,
        "fallback_used": result.fallback_used,
        "response": result.text,
        "lantra_candidate": candidate,
        "lantra_relevance": relevance,
        "issues": [
            issue.to_dict() if hasattr(issue, "to_dict") else str(issue)
            for issue in result.issues
        ],
        "attempts": [attempt.to_dict() for attempt in result.attempts],
        "model_calls": 1,
    }, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
