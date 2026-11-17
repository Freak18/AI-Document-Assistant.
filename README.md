# AI Tool

AI Tool is a lightweight PDF-grounded Q&A product with a clean user interface, a Python backend, and a retrieval pipeline built around PostgreSQL + pgvector. The app answers questions from uploaded documents, supports semantic caching, routes operational tasks into deterministic tools, and exposes a minimal admin overview for debugging and monitoring.

---

## Project summary

The current implementation is a full-stack AI workflow made of four main pieces:

1. Frontend: React + Vite + Tailwind-based UI for upload, chat, and admin mode.
2. Backend API: FastAPI service that exposes upload, chat, docs, logs, metrics, and cache-clear endpoints.
3. Retrieval and orchestration: LangGraph workflow with an RAG path, tool-based Deep Agent, and guardrails.
4. Persistence: PostgreSQL + pgvector for documents, chunks, order metadata, semantic cache, and request telemetry.

---

## Architecture at a glance

```mermaid
flowchart TD
    U[User uploads PDF or asks a question] --> API[FastAPI /upload + /chat]
    API --> G[Gateway]
    G --> R[Semantic cache lookup]
    R --> Guard[Guardrails]
    Guard --> W[LangGraph workflow]

    W --> S[store.search()]
    S --> V[pgvector cosine retrieval]
    V --> LLM[Gateway.complete()]

    W --> D{Intent classification}
    D -->|math / order| A[Deep Agent tool-calling loop]
    D -->|summary / fact| LLM
    LLM -->|rag_miss_retry| A
    LLM -->|answered| Eval[Background evaluator]
    A --> Eval
    Eval --> DB[(PostgreSQL + pgvector)]

    subgraph Data
        DB --> Docs[documents]
        DB --> Chunks[document_chunks]
        DB --> Orders[orders]
        DB --> Usage[llm_usage]
        DB --> Logs[workflow_logs]
        DB --> Cache[cache_entries]
    end
```

---

## Libraries and what they are used for

### Frontend
- React 19: application UI and state management.
- Vite 8: fast local dev server and production build tool.
- TailwindCSS 4 with @tailwindcss/vite: styling system for the clean white / gray / pastel UI.
- lucide-react: icons used across the chat and admin views.

### Backend
- FastAPI: API layer and request handling.
- Uvicorn: ASGI server that serves the backend.
- python-multipart: parsing uploaded PDF files from multipart request bodies.
- SQLAlchemy 2.x: ORM layer and database sessions.
- psycopg[binary]: PostgreSQL driver.
- pgvector: vector storage and similarity search on document embeddings.
- Alembic: migration and schema management.
- PyPDF2: PDF page extraction.
- numpy: embedding vectors and numerical math, plus in-memory fallback retrieval.
- deterministic bag‑of‑hashes embedding: lightweight, reproducible vector generation without external models.
- langchain-core: PromptTemplate and tool utilities.
- langchain-text-splitters: recursive chunking for text windows.
- langgraph: state machine orchestration for workflow routing.
- openai: OpenAI-compatible client used to call Gemini and OpenRouter endpoints (both completions and tool-calling).
- httpx: general HTTP usage, plus compatibility with provider APIs.

### Supporting runtime pieces
- pgvector.sqlalchemy.Vector: vector column mapping for document chunks.
- AST + operator modules: safe evaluation used in the calculator tool.
- re and regex: prompt classification, security screening, and summary detection.

---

## How the system works

### 1. PDF upload and ingestion
When a PDF is uploaded through the FastAPI upload route:

1. The file is validated in `backend/main.py`.
2. It must be a `.pdf`, not empty, and under the size cap.
3. The PDF is written to `backend/uploads/<filename>`.
4. Any previously indexed document is cleared with `store.clear_documents()`.
5. The semantic cache is also flushed with `gateway.clear_cache()`.
6. `store.ingest_pdf()` parses the file with PyPDF2 and extracts each page's text.
7. The extracted text is combined and split into overlapping windows (`1200` characters, `200` overlap) using `RecursiveCharacterTextSplitter`.
8. Each chunk is embedded using `embed_text()` (deterministic 384-dim bag-of-hashes embedding, keeping Vercel deployments lightweight).
9. The resulting chunks are stored as `Document` and `DocumentChunk` rows in PostgreSQL, using the pgvector column type.
10. The newest upload becomes the active document source of truth for retrieval.

This ensures stale file content does not remain mixed into future answers.

### 2. Semantic caching
The gateway keeps a lightweight in-memory cache and persists some entries to the `cache_entries` table.

- The lookup function computes a normalized prompt embedding and compares it against cached prompts.
- The score is a hybrid of:
  - cosine similarity between embeddings
  - Jaccard similarity on word sets
- If the score exceeds a threshold, the same answer is returned without re-running the full workflow.

This is how repeated or near-identical prompts are served very quickly.

### 3. Guardrails
The request passes through two layers before it reaches the RAG workflow:

- Regex guardrails in `backend/guardrails.py`
  - blocks profanity
  - blocks prompt-injection and jailbreak patterns
- LLM-as-a-judge fallback
  - a secondary model check decides whether the prompt is safe to continue
  - if blocked, the app returns a deterministic guardrail message

This keeps unsafe prompts from reaching document retrieval or the tool layer.

### 4. Routing logic in LangGraph
The workflow is implemented as a `StateGraph` in `backend/workflow.py`.

The node flow is:

1. `gateway` (skips cache on multi-turn follow-ups)
2. `guardrail` (regex always runs; LLM-as-a-judge runs when `ENABLE_LLM_JUDGE=true`)
3. `rag_solver`
4. `llm` (with `k=8` chunks + recent conversation history) or `agent`
5. If `llm` returns no match (`rag_miss_retry`), escalates to `agent` for multi-query tool search
6. end

The classification logic is intentionally simple but practical:

- `order` queries trigger the Deep Agent path
- `math` queries trigger the Deep Agent path
- `summary` / overview prompts are routed through document retrieval and summarized from the uploaded file
- normal factual questions go through document retrieval and answer generation

### 5. Retrieval and ranking
`store.search()` runs semantic retrieval over the active document (`k=8` in RAG). The app ranks hits with a domain-agnostic hybrid formula based on:

- pgvector cosine similarity (`0.45`)
- lexical overlap between the question and chunk text (`0.55`)
- recency weighting to prefer the newest uploaded document

Only the current upload is treated as active, which keeps answers anchored to the most recent document rather than stale indexed content.

### 6. LLM generation path
When a prompt is determined to be document-grounded, the app builds a retrieval prompt using `PromptTemplate` from `langchain_core`.

The model call is centralized in `backend/gateway.py`:

- it resolves the current provider from config
- if Gemini is configured, it calls the Gemini-compatible OpenAI endpoint
- if primary provider fails and OpenRouter is configured, the app retries through the fallback provider
- if both fail, the app falls back to a local deterministic answer generator
- token and cost estimates are recorded

This keeps the project configurable without scattering LLM calls across modules.

### 7. Deep Agent for operational & multi-query tasks
`DeepAgent` (`backend/agent.py`) runs an OpenAI-compatible tool-calling loop (`run_async()`) with three tools, falling back to deterministic `run()` in offline/`local-fallback` mode:

- `search_document`: searches the active document with rephrased or alternative queries (triggered on `rag_miss_retry` or multi-step questions)
- `calculator`: safe AST-based arithmetic evaluator
- `order_lookup`: queries the `orders` table in PostgreSQL, or falls back to a built-in sample order map

This makes the project behave differently for:

- document facts: RAG grounded in uploaded content
- operational tasks: deterministic tool execution with no hallucinated logic

### 8. Background evaluator
After the model answer is returned, the app queues `run_evaluator()` in the background.

The evaluator:

- builds a rubric prompt
- asks the gateway to score directness and quality
- parses the numeric score from the response
- stores the evaluation result in the `evaluations` table

This is a lightweight feedback loop rather than an interrupting user-facing step.

---

## Storage and data model

The database is defined in `backend/models.py` and includes:

- `documents`: file metadata and upload state
- `document_chunks`: chunk text and embedding vectors
- `orders`: operational lookup catalog
- `llm_usage`: token, cost, and provider telemetry
- `workflow_logs`: node-level execution history
- `evaluations`: background answer-quality scores
- `conversations`: conversation-level metadata
- `messages`: per-message chat records
- `cache_entries`: persisted semantic cache entries

The app also keeps an in-memory log buffer via `backend/trace_log.py` for quick UI status updates and front-end diagnostics.

---

## Request lifecycle

A normal question flow looks like this:

1. User sends a message from the frontend.
2. FastAPI `/chat` endpoint receives the request.
3. `run_workflow()` is called.
4. Gateway checks the semantic cache.
5. Guardrails validate the prompt.
6. The workflow classifies the question.
7. If it is a document question, retrieval runs on the active document.
8. The top relevant chunks are assembled into a context string.
9. The LLM answer is generated through the gateway.
10. If the query is math or order-related, the Deep Agent may return a deterministic answer instead.
11. The result is sent back to the frontend along with node state, trace info, and metrics.
12. A background evaluation is queued for quality scoring.

---

## Startup and runtime behavior

The app is launched by the project root `start.sh` script, which is responsible for:

- validating Node.js and the Python virtual environment
- detecting local or remote PostgreSQL (e.g., Neon)
- bootstrapping database schema & vector extensions via Alembic
- running the architectural verification test suite
- starting the FastAPI backend on port 8000 and polling `/health`
- starting the Vite React frontend on port 5173
- opening your default browser automatically
- managing clean shutdown on Ctrl+C

### Quick Launch Commands

```bash
# Standard single-click launch (runs all 44 architectural verification tests)
./start.sh

# Fast startup (skips test suite for instant boot)
./start.sh --fast

# Force reinstallation of requirements
./start.sh --install
```

---

## Current product behavior

The current app is intentionally designed around a minimal, premium UX:

- User mode: clean chat and upload experience
- Admin mode: diagnostic flow / logs view
- Light background palette with white, gray, and soft pastel accents
- PDF upload is treated as a document grounding workflow rather than a generic chatbot response
- The newest uploaded file is always the active document source of truth

---

## Architecture summary

The app is not a generic LLM wrapper. It is a controlled document-grounded workflow with:

- deterministic storage and retrieval
- semantic caching for repeated prompts
- guardrails for safe interaction
- a strict RAG path for uploaded documents
- tool-based fallback for math and operational data queries
- lightweight monitoring and evaluation built into the stack

This makes it suitable for real-world Q&A over uploaded PDFs while keeping operational behavior transparent, controlled, and debuggable.
