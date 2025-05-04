# Embedding Models Overview

| Size    | 100d (BPE / GloVe)                              | 200d (BPE / GloVe)                              | 300d (BPE / GloVe)                                |
|---------|------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| 5k      | Lightweight, prototyping, mobile tasks         | Lightweight, better semantics, small tasks     | Lightweight, rich semantics, quick prototypes  |
| 10k     | Small vocab, fast pipelines, basic NLP         | Small vocab, improved coverage, solid NLP      | Small vocab, high-fidelity similarity tasks    |
| 20k     | Mid-size vocab, chatbots, entity tasks         | Mid-size vocab, sentiment, QA tasks            | Mid-size, fine-grained similarity, classification |
| 30k     | Large vocab, document tasks, summarization     | Large vocab, complex NLP, QA pipelines         | Large vocab, best for semantic similarity, QA  |
| 40k     | Broad vocab, research models, summarization    | Broad vocab, summarization, information retrieval | Broad vocab, top-end summarization, deep QA    |
| 50k     | Full vocab, research-grade models, heavy NLP   | Full vocab, advanced research tasks            | Full vocab, highest semantic detail, research NLP |


---

### Suggested Use Cases

- **100d** → Fast inference, lightweight devices, simple tasks  
- **200d** → Balanced depth and size, versatile general-purpose use  
- **300d** → Deep semantic tasks, research, fine-grained text understanding, top-tier models

---
