# AI Tool - Full Stack Architecture Guide

This document describes the current implementation of the project, not the older planning draft. It reflects the actual libraries, runtime mechanisms, and request-processing flow in the codebase as it exists today.

---

## 1. System overview

The app is a minimal but functional AI workflow built around three main ideas:

1. Users upload a PDF and ask questions against it.
2. The backend parses, chunks, embeds, and stores the document in PostgreSQL + pgvector.
3. LangGraph decides whether the query should be answered from retrieval, from a deterministic tool, or from a guardrail-blocked path.

The actual stack is:

- Frontend: React 19 + Vite + TailwindCSS
- Backend: FastAPI + Uvicorn
- Orchestration: LangGraph + langchain-core
- Retrieval: PostgreSQL + pgvector + SQLAlchemy
- Embeddings: deterministic 384-dim bag-of-hashes (`blake2b` token + bigram hashing, Vercel-lightweight)
- LLM access: OpenAI-compatible client to Gemini/OpenRouter
- Safety: regex filters + an LLM-as-a-judge gate
- Tooling: safe calculator + order lookup

---

## 2. Current runtime stack

### Frontend

The frontend is intentionally minimal and clean:

- React 19 for component rendering and state
- Vite 8 for local dev and build pipeline
- TailwindCSS 4 for the premium white / gray / pastel UI
- lucide-react for icons
- App state is owned in `frontend/src/App.jsx`, with upload flow and mode switching handled there

### Backend

The backend is structured as a FastAPI app with a modular architecture:

- `backend/main.py`: HTTP API surface (`/health`, `/chat`, `/upload`, `/documents`, `/metrics`, `/logs`, `/clear-cache`)
- `backend/config.py`: configuration and provider selection
- `backend/database.py`: SQLAlchemy engine, DB bootstrap, health checks
- `backend/models.py`: ORM tables for documents, chunks, cache, usage, logs, and conversations
- `backend/gateway.py`: centralized LLM access and cache logic
- `backend/guardrails.py`: input screening and security judge
- `backend/workflow.py`: LangGraph orchestration and routing
- `backend/agent.py`: LLM tool-calling Deep Agent (`search_document`, `calculator`, `order_lookup`) with deterministic fallback
- `backend/vector_store.py`: PDF text extraction, chunking (`1200`-char / `200`-overlap), bag-of-hashes embedding, indexing, and retrieval

### Supporting libraries

- SQLAlchemy 2.x for ORM/session logic
- psycopg[binary] for PostgreSQL access
- pgvector for vector storage and similarity search
- Alembic for migrations and schema bootstrap
- PyPDF2 for parsing uploaded PDFs
- numpy for vector math and in-memory fallback search
- langchain-core and langgraph for prompt templates, `@tool` wrappers, and state-machine orchestration
- langchain-text-splitters for chunk splitting
- openai Python SDK (`AsyncOpenAI`) used through OpenAI-compatible endpoints for Gemini and OpenRouter (completions + tool calling)
- httpx used in the ecosystem and compatibility layer

---

## 3. How the request lifecycle works

### Step A: Startup and DB bootstrap
When the app starts, `backend/main.py` runs its lifespan hook:

1. `ensure_database_ready()` checks whether Postgres is available.
2. It creates the `vector` extension if needed.
3. It verifies required tables exist.
4. If they do not, it runs Alembic migrations and then creates metadata.
5. It seeds a minimal order catalog if the DB is otherwise empty.
6. It logs the gateway status and database status.

This ensures that the app can boot in a clean environment without manual schema setup.

### Step B: Uploading a PDF
The `/upload` endpoint does the following:

1. Validates that the uploaded file is a PDF and not empty.
2. Saves the raw file to `backend/uploads/<filename>`.
3. Clears old indexed documents with `store.clear_documents()`.
4. Clears cached answers with `gateway.clear_cache()`.
5. Calls `store.ingest_pdf(data, filename)`.
6. Extracts raw text page by page using PyPDF2.
7. Concatenates and chunks it using `RecursiveCharacterTextSplitter`.
8. Embeds each chunk with `embed_text()`.
9. Persists Document and DocumentChunk records to PostgreSQL.
10. Sets the newest upload as the active document source of truth.

This is the core ingestion pipeline and does not call the LLM for chunking or indexing.

### Step C: Query handling
When the user sends a chat request, `/chat` calls `run_workflow()`.

The workflow state starts with:

- `gateway`: idle
- `guardrail`: idle
- `rag`: idle
- `llm`: idle
- `agent`: idle
- `eval`: idle

Then the graph executes in this order:

1. `gateway_node` (skips cache on multi-turn follow-ups)
2. `guardrail_node` (regex always runs; LLM judge runs when `ENABLE_LLM_JUDGE=true`)
3. `rag_solver_node`
4. either `llm_node` (with `k=8` chunks + recent conversation history) or `agent_node`
5. if `llm_node` finds no match (`rag_miss_retry`), escalates to `agent_node`
6. end

---

## 4. Gateway and semantic cache

The gateway is the single LLM access layer. All model calls go through `gateway.complete()` rather than bypassing to other modules.

The gateway does four important things:

1. Chooses the configured provider from `config.py`
2. Calls Gemini/OpenRouter through an OpenAI-compatible API
3. Falls back to the alternate provider on transient failures
4. Tracks token usage, approximate cost, and request metrics

The semantic cache is implemented in `gateway.lookup()` and `gateway.remember()`.

It combines:

- cosine similarity between the current prompt embedding and cached prompt embedding
- Jaccard similarity of the word sets

The score is combined as:

- 65% embedding similarity
- 35% lexical overlap

If the score passes the threshold, the app returns the cached answer immediately.

This prevents repeated prompts from re-running the same retrieval and generation flow unnecessarily.

---

## 5. Guardrails

Safety is enforced by `backend/guardrails.py`.

### Regex layer
This checks the prompt for:

- profanity
- jailbreak / prompt-injection phrases
- attempts to reveal system instructions
- malicious or bypassing language

### LLM judge layer
If regex passes, a judge prompt is sent to the gateway with purpose `judge`.

The model is asked to return exactly `ALLOW` or `BLOCK`.

If blocked, the app returns a fixed security message and stops the workflow before retrieval begins.

This is a practical gate: raw regex is cheap, and the model-based judgment handles edge cases without making the app overly noisy.

---

## 6. Retrieval and vector search

The vector store is implemented in `backend/vector_store.py`.

### Embedding strategy

`embed_text()` generates a normalized 384-dim deterministic bag-of-hashes vector (`blake2b` over unigrams and bigrams), avoiding heavy local ML dependencies such as `sentence-transformers` so the backend deploys cleanly on Vercel.

### Chunking strategy

The app uses `RecursiveCharacterTextSplitter` with:

- `1200`-character window (`WINDOW_WORDS = 1200`)
- `200`-character overlap (`OVERLAP_WORDS = 200`)

This keeps related clauses and headings together across nearby sections of any document.

### Ranking strategy

Retrieval is domain-agnostic mixed-rank:

- vector similarity from pgvector (`0.45`)
- lexical word overlap between the query and chunk text (`0.55`)
- recency boost so newest upload wins

The active document is the newest uploaded source. Older documents are intentionally not allowed to dominate the answer set.

---

## 7. Workflow classification

The app uses `workflow.py` to decide how to handle each request.

### Intent routes

The request is classified by regex patterns:

- `order`: asks for shipment / order status / operational records
- `math`: arithmetic or percent calculations
- `summary`: overview, summarize, document description, main idea questions
- `fact`: direct question about the uploaded document

### Behavior

- `order` and `math` route directly to the Deep Agent
- document summary / overview prompts can still answer from the uploaded context instead of being rejected as no-match
- normal fact questions go through RAG

This distinction is important because the app is designed to answer grounded document questions while still supporting operational tool workflows.

---

## 8. Deep Agent and tools

The Deep Agent is implemented in `backend/agent.py`. When an external LLM is configured, `DeepAgent.run_async()` runs an OpenAI-compatible tool-calling loop (up to 4 iterations) over `search_document`, `calculator`, and `order_lookup`. In `local-fallback` mode, it falls back to deterministic `DeepAgent.run()`.

### Document search (`search_document`)
Allows `DeepAgent` to re-query the active document with alternative terms or synonyms when initial single-shot RAG returns no match (`rag_miss_retry`) or when a multi-step query needs document facts combined with math.

### Calculator
The calculator uses Python’s `ast` module to parse an expression and safely evaluate only numeric operations.

Allowed operations:

- addition
- subtraction
- multiplication
- division
- modulo
- exponent

This avoids arbitrary code execution while still supporting arithmetic in the app.

### Order lookup
The order tool performs this logic:

1. looks for an `ORD-...` pattern in the user prompt
2. checks the PostgreSQL `orders` table
3. falls back to an in-memory order catalog if the DB is unavailable
4. returns a concise answer such as status + item + ETA

This is a deterministic tool path, not a generative one.

---

## 9. Evaluation and monitoring

After a successful answer, the app launches a background evaluator.

The evaluator prompt asks the model to rate the answer on a 0.0 to 1.0 scale for directness and factual alignment. The result is stored in the `evaluations` table and also reflected in the in-memory metrics object.

Telemetry includes:

- total requests
- cache hits
- cache hit rate
- tokens used
- estimated cost
- latest evaluation score

The app also persists workflow events to `workflow_logs` and in-memory logs for the frontend console.

---

## 10. Persistence layout

The database is centered on PostgreSQL + pgvector and stores:

- `documents`: metadata and ingestion state
- `document_chunks`: embeddings and chunk text
- `orders`: sample order data for operational lookups
- `llm_usage`: token and cost telemetry
- `workflow_logs`: node traces and execution events
- `evaluations`: background quality scoring output
- `conversations`: conversation metadata
- `messages`: chat records
- `cache_entries`: persisted semantic-cache entries

This makes the app practical for both local debugging and later production-style observability.

---

## 11. Actual processing path in one sentence

The application does this in sequence: upload PDF → extract text → chunk text → embed chunks → store in pgvector → guardrail request → look up cache → classify intent → retrieve relevant chunks → either answer from document context or use a deterministic tool → score the answer in the background.

This is the current architecture of the project and is the version the app is designed around today.
