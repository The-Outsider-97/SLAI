# LANTRA supervised corpus v3

This bundle replaces the old `synthetic_seed_v2` corpus.

## What changed
- Preserves the existing seven LANTRA task contracts.
- Removes the shared 30-concept technical glossary pattern that dominated the old corpus.
- `dialogue` teaches greetings, small talk, capabilities, clarification, sentiment, planning, general questions, and multi-turn assistance.
- `generation` teaches rewriting, explanation, comparison, drafting, extraction, structuring, and reasoning.
- `classification` uses one coherent taxonomy: `user_request_intent_v1`.
- `translation`, `summarization`, `embedding`, and `reranking` use task-appropriate examples.
- Variants from the same source group are kept in the same split.

## Training
Archive the old supervised seed bundle first, then place these files in `data/processed/lantra/supervised/`.

Use the raw-pretraining checkpoint as the starting point so the old synthetic supervised behavior is not carried forward:

```powershell
py -m train_lantra `
  --init-from src\agents\language\checkpoints\lantra\lantra_raw_pretrain_best.pt `
  --data data\processed\lantra\supervised `
  --raw-pretrain-epochs 0 `
  --no-glove-bootstrap `
  --require-all-supervised-tasks
```

This is a corrected, internally consistent starter corpus. It is intentionally not described as deployment-scale; expand it with licensed and independently curated data after validating this baseline.
