# RAG Evaluation Harness

A small, dependency-free harness that **measures** the HealthAI RAG pipeline against
a labeled test set — instead of judging quality by feel.

It deliberately separates two kinds of metrics:

| Stage | Needs the LLM? | Cost | What it answers |
| --- | --- | --- | --- |
| **Retrieval** | No (local embeddings + vector store) | Free / fast | *Did we fetch the right documents?* |
| **Generation** | Yes | Tokens | *Did the model answer well and refuse when it should?* |
| **LLM judge** | Yes | Tokens | *Is the answer faithful to the context and relevant?* |

Because retrieval quality is the foundation of any RAG system and costs nothing to
measure, the default run evaluates retrieval only.

## Run it

```bash
# Free, fast — retrieval metrics only
python -m eval.run

# Also generate answers and score them (calls the LLM)
python -m eval.run --with-generation

# Add reference-free LLM-judged faithfulness & relevance
python -m eval.run --judge

# Sweep retrieval depth
python -m eval.run --k 5
```

Requires the vector store to be ingested (run the app once, or `POST /ingest`) and
`OPENROUTER_API_KEY` set in `.env` (the retrieval-only run reads config but never
calls the API).

Each run prints a console report and writes `eval/report.json` and `eval/report.md`.

## Metrics

**Retrieval** (over answerable cases)
- **Hit@k** — fraction of questions where at least one expected source document was
  retrieved in the top *k*. The headline retrieval number.
- **Recall@k** — fraction of *all* expected sources retrieved (for multi-source cases).
- **MRR** — mean reciprocal rank of the first correct source; rewards ranking the
  right document higher.

**Generation**
- **Answer rate** — share of answerable questions the assistant actually answered
  (did *not* incorrectly abstain).
- **Keyword coverage** — share of expected answer keywords present (a lightweight,
  LLM-free correctness signal).
- **Abstention accuracy** — for out-of-scope questions, how often the assistant
  correctly refused with the fallback message. Tests that the relevance floor and
  prompt guardrails actually hold.
- **Avg latency** — wall-clock seconds per answered question.

**LLM judge** (reference-free, RAGAS-style)
- **Faithfulness** — every claim in the answer is supported by the retrieved context.
- **Relevance** — the answer addresses the question (a correct refusal counts).

## The test set

[`testset.json`](testset.json) holds labeled cases: a question, the source
document(s) that *should* be retrieved, optional answer keywords, and a
`should_answer` flag (set `false` for out-of-scope questions the assistant must
refuse). Add cases as you add documents — more labels make the metrics sharper.

## Why this matters

This harness makes quality changes **comparable**: tweak the chunk size, the
embedding model, or add a reranker, re-run, and see the numbers move. That is the
difference between "it seems better" and "Hit@k went from 0.78 → 0.94."
