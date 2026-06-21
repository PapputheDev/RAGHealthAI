# HealthAI — Healthcare AI Assistant (RAG + OpenRouter)

[![CI](https://github.com/PapputheDev/RAGHealthAI/actions/workflows/ci.yml/badge.svg)](https://github.com/PapputheDev/RAGHealthAI/actions/workflows/ci.yml)

A compact, interview-ready Healthcare AI Assistant demonstrating:

- Synthetic healthcare knowledge base ingestion (TXT → chunks → embeddings)
- Persistent local vector search using ChromaDB
- Retrieval-Augmented Generation (RAG) using OpenRouter (`meta-llama/llama-3.3-70b-instruct`)
- A simple “agent router” that sends scheduling-related questions to a mock appointment tool

This repository uses **synthetic healthcare content only** (see `data/`). It is intended for software demonstration and does **not** provide medical advice.

## Dataset Details

The knowledge base consists of synthetic healthcare documents:

- appointment_scheduling_policy.txt
- telehealth_policy.txt
- medication_refill_policy.txt
- insurance_eligibility_faq.txt
- patient_discharge_instructions.txt
- hipaa_privacy_guidelines.txt

These documents were manually created for demonstration purposes and do not contain any real patient data, PHI, or confidential healthcare information.

## Project Overview

The goal is to show an end-to-end RAG pipeline with a clean, modular Python codebase:

- **Ingestion**: read `.txt` docs → chunk → embed → store in ChromaDB
- **Retrieval**: embed query → top-k semantic search
- **Generation**: build a strict “context-only” prompt → call OpenRouter Llama → return answer + sources + confidence

## Technology Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| **LLM** | OpenRouter `meta-llama/llama-3.3-70b-instruct` | Free-tier OpenAI-compatible API; strong reasoning for policy Q&A |
| **Embedding model** | `BAAI/bge-small-en-v1.5` (SentenceTransformers) | Fast, small, normalized vectors; state-of-the-art retrieval benchmarks for its size |
| **Vector database** | ChromaDB (persistent) | Zero-infra local setup; survives restarts; good enough for prototype-scale datasets |
| **Text splitter** | LangChain `RecursiveCharacterTextSplitter` | Sentence-boundary-aware; configurable overlap preserves context across chunk edges |
| **Framework** | FastAPI | Async-ready, auto-generates OpenAPI docs, Pydantic validation built-in |

## Architecture

High-level flow:

```mermaid
flowchart LR
   U[User/API Client] -->|POST /ask| API[FastAPI]
   API --> A[Agent Router]

   A -->|Scheduling keywords| T[Mock Appointment Tool]
   A -->|Otherwise| R[RAG Orchestrator]

   subgraph Ingestion
      D[TXT docs in data/] --> S[RecursiveCharacterTextSplitter]
      S --> E[SentenceTransformers embeddings]
      E --> C[(ChromaDB persistent store)]
   end

   R -->|Top 3| C
   R --> P[Healthcare RAG Prompt]
   P --> L[OpenRouter Llama 3.3 70B]
   L --> API
   T --> API
```

Key modules:

- `app/config.py` — typed settings via env vars + `.env` loading
- `app/embeddings.py` — SentenceTransformers singleton (`BAAI/bge-small-en-v1.5`)
- `app/vector_store.py` — persistent ChromaDB wrapper (collection `healthcare_docs`)
- `app/ingest.py` — ingestion pipeline (500/100 chunking)
- `app/prompts.py` — strict healthcare RAG prompt (context-only, refusal rules)
- `app/llm.py` — OpenRouter client with retries/timeouts
- `app/rag.py` — retrieve → prompt → generate → return answer/sources/confidence
- `app/agent.py` — routes appointment questions to a mock tool; otherwise to RAG
- `app/main.py` — FastAPI API surface

## Prompt Engineering

The system uses a strict context-only healthcare prompt. The full prompt template is:

```
You are a Healthcare Policy & Information Assistant.

You must follow these rules strictly:

1) Use only the provided CONTEXT to answer.
    - Do NOT use outside knowledge.
    - Do NOT guess or invent details.
    - If the CONTEXT does not contain enough information to answer, reply with exactly:
      I could not find this information in the provided documents.

2) Safety restrictions (refuse these requests):
    - Diagnosis: Do not diagnose or assess the likelihood of a medical condition.
    - Prescriptions: Do not prescribe, recommend specific prescription drugs, or provide dosage instructions.
    - Emergency guidance: If the user describes an emergency, advise them to seek urgent/emergency care.

3) Source-backed answers:
    - Cite sources from the CONTEXT.
    - Use bracketed citations at the end of the relevant sentence.

4) Style:
    - Be concise and professional.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
```

Key design decisions:
- The model is explicitly forbidden from using outside knowledge, preventing hallucination.
- The exact fallback phrase is hardcoded in both the prompt and `prompts.py` for consistency.
- Diagnosis/prescription refusals protect against unsafe medical advice.
- Inline citations (`[source: filename]`) are enforced, making answers auditable.

## Agent / Tool Workflow

The `agent.py` module implements a lightweight router:

```
User question
    │
    ▼
Does the question contain scheduling keywords?
("appointment", "book", "schedule", "available slot", …)
    │
    ├── YES → appointment_tool(department, modality)
    │          Extracts department (cardiology, dermatology, …) and modality
    │          (video/in_person) from the question text, then returns
    │          synthetic available slots.
    │
    └── NO  → RAG pipeline
               Retrieve top-3 chunks from ChromaDB → build context →
               format prompt → call OpenRouter Llama → return answer +
               sources + confidence
```

The `route` field in every `/ask` response shows which path was taken.

## Sample Questions and Responses

**Policy question (RAG path):**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can a patient request a medication refill through telehealth?"}'
```

```json
{
  "answer": "Yes, patients can request medication refills through telehealth if the medication is already prescribed and does not require an in-person evaluation. [source: telehealth_policy.txt]",
  "sources": [
    {"document": "telehealth_policy.txt", "chunk": "Medication refill requests may be reviewed during telehealth visits..."},
    {"document": "medication_refill_policy.txt", "chunk": "Refill requests submitted via telehealth are processed within 72 hours..."}
  ],
  "confidence": "high",
  "route": "rag"
}
```

**Scheduling question (appointment tool path):**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I book a cardiology appointment via video call?"}'
```

```json
{
  "answer": "Here are the next available appointment slots (synthetic).\n\nAvailable slots:\n- 2026-06-04T09:00 (20 min, video)\n- 2026-06-04T11:00 (20 min, video)\n- 2026-06-04T13:00 (20 min, video)",
  "sources": [],
  "confidence": "high",
  "route": "appointment_tool"
}
```

**Unknown question (no relevant documents):**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the hospital cafeteria menu?"}'
```

```json
{
  "answer": "I could not find this information in the provided documents.",
  "sources": [],
  "confidence": "low",
  "route": "rag"
}
```

## Setup

### 1) Python environment

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate:

- Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2) Environment variables

Copy `.env.example` to `.env` and set values.

```bash
copy .env.example .env
```

## Environment Variables

All configuration is loaded from `.env` via `app/config.py` (Pydantic Settings). No secrets are hardcoded in source code.

### Setup

```bash
copy .env.example .env    # Windows
cp .env.example .env      # Linux/macOS
# then open .env and paste your OPENROUTER_API_KEY
```

### Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | **Yes** | — | OpenRouter API key for LLM calls. Get one free at openrouter.ai |
| `MODEL_NAME` | No | `meta-llama/llama-3.3-70b-instruct` | OpenRouter model ID to use for generation |
| `CHUNK_SIZE` | No | `800` | Max characters per document chunk during ingestion |
| `CHUNK_OVERLAP` | No | `200` | Characters of overlap between consecutive chunks (preserves cross-chunk context) |
| `CHROMA_DB_PATH` | No | `./chroma_db` | Directory where ChromaDB persists its vector index. Docker overrides this to `/app/chroma_db` |
| `LOG_LEVEL` | No | `INFO` | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Logging & Error Handling

The application includes:

- Structured application logging using Python logging module
- Request-level logging for ingestion and question-answering workflows
- Validation of API inputs through Pydantic models
- Graceful handling of:
  - Missing documents
  - Empty vector database
  - OpenRouter API failures
  - Invalid requests
  - Missing environment variables

Errors are returned with meaningful HTTP status codes and messages.

### Security notes

- `.env` is listed in `.gitignore` — never commit it.
- `.env.example` is the safe template (no real keys) — commit that instead.
- The `config.py` module validates all variables at startup; a missing `OPENROUTER_API_KEY` causes an immediate descriptive error rather than a silent runtime failure.

## Ingestion Workflow

Ingestion reads all `.txt` files under `data/`, chunks them using `RecursiveCharacterTextSplitter` with values from `.env` (defaults: `chunk_size=800`, `chunk_overlap=200`). Then it embeds chunks using `BAAI/bge-small-en-v1.5` and stores them in the persistent ChromaDB collection `healthcare_docs`.

Run ingestion:

```bash
python .\app\ingest.py
```

You can also call ingestion via the API:

```bash
curl -X POST http://localhost:8000/ingest
```

## RAG Workflow

For non-scheduling questions:

1. Retrieve **top 3** chunks from ChromaDB
2. Build a context block with source labels
3. Format the strict healthcare prompt:
    - answers **only** from provided context
    - refuses hallucinations
    - refuses diagnosis and prescriptions
    - requires source-backed statements
    - returns the exact fallback message when context is insufficient
4. Call OpenRouter Llama and return:
    - `answer`
    - `sources` (source filenames)
    - `confidence` (derived from retrieval distance)

## API

Start the API server:

```bash
uvicorn app.main:app --reload
```

Swagger UI:

- (http://127.0.0.1:8000/)

### Endpoints

- `GET /health`
- `POST /ingest`
- `POST /ask`

### API Examples

Health check:

```bash
curl http://localhost:8000/health
```

Ingest documents:

```bash
curl -X POST http://localhost:8000/ingest
```

Ask a scheduling question (routes to mock appointment tool):

```bash
curl -X POST http://localhost:8000/ask \
   -H "Content-Type: application/json" \
   -d "{\"question\": \"Can I schedule a doctor visit next week?\"}"
```

Ask a policy question (routes to RAG + OpenRouter):

```bash
curl -X POST http://localhost:8000/ask \
   -H "Content-Type: application/json" \
   -d "{\"question\": \"What is the no-show fee and how late can I arrive?\"}"
```

## Web UI

A chat-style frontend is served automatically at `http://localhost:8000/` when the API is running.

Features:
- Chat interface with user/AI message bubbles
- Example question chips (click to fill the input)
- **Ingest Docs** button runs `/ingest` and shows a toast with chunk stats
- Each AI response shows a **route badge** (RAG or Appointment Tool) and a **confidence badge** (High/Medium/Low)
- Sources are shown with document name and excerpt beneath each answer
- Typing animation while waiting for a response

No separate build step — the UI is a single HTML file (`static/index.html`) served by FastAPI.

## Local Smoke Tests

OpenRouter connectivity test (prints one response):

```bash
python .\tests\openrouter_smoke.py
```

Retrieval test (no LLM calls; prints top 3 chunks + scores + sources):

```bash
python .\tests\retrieval_smoke.py
```

## Docker Usage

This repo includes a production-friendly container setup:

- Python 3.11 slim image
- Installs `requirements.txt`
- Runs `uvicorn` with multiple workers
- Persists ChromaDB data via a named volume

Build and run:

```bash
docker compose up --build
```

Persistent vector DB:

- A named Docker volume is mounted to `/app/chroma_db`.

## Future Improvements

Engineering / reliability:

- Add CI (lint, type-check, tests), and structured logging (JSON)
- Add request IDs + tracing (OpenTelemetry)
- Add rate limiting and authentication (API key/JWT)

RAG quality:

- Store richer `SourceReference` details (chunk_index + score) end-to-end
- Better confidence calibration and threshold-based “insufficient context” behavior
- Add hybrid search (BM25 + embeddings) and metadata filters
- Add evaluation harness (golden Q/A set + regression tests)

Safety & compliance (real-world):

- Formal HIPAA risk assessment and privacy/security controls
- Separate PHI from vector indexes, and implement retention + access policies
- Human-in-the-loop escalation and clinically reviewed content

Product:

- Replace the mock appointment tool with a real scheduling integration
- Add conversation memory with strict PHI-safe handling

