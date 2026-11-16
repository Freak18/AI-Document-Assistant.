# Changes: PostgreSQL + pgvector and Unified LLM Provider Architecture

## 1. Objective

Update the existing **Enterprise AI Architecture Simulator** without
rebuilding it.

The current application already includes a React frontend, FastAPI
backend, LangGraph orchestration, RAG with FAISS, Deep Agent tool
execution, guardrails, an LLM gateway, telemetry, and background
evaluation.

This change introduces:

1.  PostgreSQL as the persistent application database.
2.  pgvector as the vector storage and similarity-search layer,
    replacing FAISS.
3.  A centralized LLM configuration and provider manager.
4.  Gemini as the preferred provider and OpenRouter as the fallback
    provider.
5.  A rule that all application LLM calls must pass through the LLM
    Gateway.
6.  Persistence for documents, chunks, conversations, messages, usage,
    evaluations, workflow logs, cache entries, and orders.

Preserve existing behavior and the current React UI wherever possible.
Do not remove working functionality unless it is explicitly replaced
below.

------------------------------------------------------------------------

## 2. Target Architecture

``` text
React + Vite
    |
    v
FastAPI
    |
    v
LLM Gateway
    |-- Provider selection and runtime fallback
    |-- Semantic cache
    |-- Token and cost tracking
    |-- Request/response telemetry
    |
    v
Guardrails
    |-- Regex screening
    |-- LLM-as-a-Judge through Gateway
    |
    v
LangGraph Orchestrator
    |-- RAG path
    |     |-- PostgreSQL + pgvector retrieval
    |     |-- Context construction
    |     |-- LLM generation through Gateway
    |
    |-- Deep Agent path
          |-- Calculator tool
          |-- PostgreSQL order lookup tool
          |-- LLM reasoning/tool selection through Gateway
    |
    v
Final Answer
    |
    v
Background Evaluator through Gateway

PostgreSQL:
    - Application and conversation data
    - Document metadata and extracted chunks
    - pgvector embeddings
    - LLM usage and cost records
    - Evaluations and workflow logs
    - Persistent semantic cache
    - Operational order records
```

### Architectural principles

-   LangGraph must not depend directly on FAISS or pgvector. It should
    call a retrieval service/interface.
-   Application code must not instantiate provider-specific LLM clients
    directly.
-   All LLM calls, including guardrail judging, RAG generation, agent
    reasoning, routing/classification when LLM-based, and evaluation,
    must use the LLM Gateway.
-   Embeddings are a separate capability from chat/completion models.
-   Preserve deterministic tools such as calculator and exact order
    lookup; do not invoke an LLM when deterministic execution is
    sufficient.
-   Use environment variables for secrets. Never commit API keys or
    database passwords.

------------------------------------------------------------------------

## 3. PostgreSQL and pgvector

### 3.1 Local development setup

Add a Docker Compose service using a pgvector-enabled PostgreSQL image.

Example:

``` yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: ai-workbench-db
    environment:
      POSTGRES_DB: ai_workbench
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d ai_workbench"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  postgres_data:
```

Use a local `.env` file for `POSTGRES_PASSWORD`; do not hardcode
production credentials. Ensure `.env` is in `.gitignore`.

Enable the extension during database initialization or through a
migration:

``` sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3.2 Python dependencies

Add the required packages to the backend dependency file:

``` text
sqlalchemy
psycopg[binary]
pgvector
alembic
langchain-openai
```

Keep existing dependencies that are still used by the application.
Remove `faiss-cpu` only after the FAISS implementation and tests have
been fully replaced.

Use compatible, pinned versions in the project's dependency management
process rather than allowing uncontrolled upgrades.

### 3.3 Database connection

Create a dedicated database module, for example `backend/database.py`,
responsible for:

-   Reading `DATABASE_URL` from environment configuration.
-   Creating the SQLAlchemy engine.
-   Creating the session factory.
-   Providing a FastAPI dependency for database sessions.
-   Managing connection lifecycle and clean shutdown.
-   Supporting a connection pool appropriate for the deployment
    environment.

Example environment variable:

``` env
DATABASE_URL=postgresql+psycopg://admin:YOUR_PASSWORD@localhost:5432/ai_workbench
```

Do not put credentials directly in source code.

### 3.4 Database models and tables

Create SQLAlchemy models and Alembic migrations for the following
logical entities.

  -----------------------------------------------------------------------
  Table                               Important fields / purpose
  ----------------------------------- -----------------------------------
  `documents`                         ID, filename, storage key/path,
                                      content type, file size, status,
                                      title, created timestamp

  `document_chunks`                   ID, document ID, chunk index, chunk
                                      text, embedding vector, metadata,
                                      created timestamp

  `conversations`                     ID, optional user/session
                                      identifier, title, created and
                                      updated timestamps

  `messages`                          ID, conversation ID, role, content,
                                      provider, model, created timestamp

  `llm_usage`                         ID, request ID, provider, model,
                                      input/output/total tokens,
                                      estimated cost, cache-hit flag,
                                      latency, timestamp

  `evaluations`                       ID, conversation/message/request
                                      reference, directness score,
                                      factual-alignment score, evaluator
                                      feedback, status, timestamp

  `workflow_logs`                     ID, request/conversation reference,
                                      node name, status, log message,
                                      duration, timestamp

  `cache_entries`                     ID, normalized prompt or cache key,
                                      response, embedding if used,
                                      provider/model metadata, expiry and
                                      timestamps

  `orders`                            ID, order number, status, and mock
                                      operational fields used by the
                                      order lookup tool
  -----------------------------------------------------------------------

Requirements:

-   Use UUIDs or another consistent primary-key strategy.
-   Add foreign keys and indexes where appropriate.
-   Add timestamps for auditability.
-   Use JSON/JSONB for flexible metadata only where it is useful; keep
    core searchable fields relational.
-   Store original PDFs in local/object storage, not as large binary
    values in ordinary metadata rows. Store their storage key/path in
    `documents`.
-   Store extracted text and its embedding together in
    `document_chunks`.
-   Ensure deleting a document has a defined behavior for its chunks.
-   Add indexes for common lookup fields such as document ID,
    conversation ID, order number, request ID, and timestamps.

### 3.5 Embedding model and vector dimension

Keep the embedding provider/model configurable independently from the
chat LLM provider.

The vector column dimension must exactly match the chosen embedding
model. Do not assume a dimension without checking the model actually
configured in the existing code.

For example, `sentence-transformers/all-MiniLM-L6-v2` produces
384-dimensional embeddings. If this model is selected, define the
pgvector column as `Vector(384)`. If another model is used, configure
the corresponding dimension and migrate the schema accordingly.

Requirements:

-   Use the same embedding model for document chunks and query
    embeddings.
-   Normalize vectors consistently if the selected similarity metric and
    implementation require it.
-   Record the embedding model/version in document or chunk metadata so
    future model changes can be managed safely.
-   Do not silently mix embeddings from different models in the same
    vector index.
-   Keep embedding generation local if the current implementation
    already uses local normalized embeddings and that remains suitable.

### 3.6 Vector retrieval

Replace FAISS search with PostgreSQL pgvector search.

Use cosine distance where appropriate. A representative query is:

``` sql
SELECT
    id,
    document_id,
    chunk_text,
    embedding <=> :query_embedding AS distance
FROM document_chunks
WHERE document_id = :document_id
ORDER BY embedding <=> :query_embedding
LIMIT 5;
```

The document filter is optional; retrieval should support searching
across all indexed documents when the user has not selected a specific
document.

Requirements:

-   Return the chunk text, document reference, similarity/distance, and
    metadata needed by RAG.
-   Apply metadata filters where required.
-   Make `top_k` configurable.
-   Convert distance to a clearly defined similarity score only when
    needed; do not label raw cosine distance as similarity.
-   Add an HNSW or IVFFlat index when appropriate for expected data
    volume. Start with a correct exact-search implementation and
    add/benchmark an approximate index as a separate optimization.
-   Keep retrieval behind a service/interface so the orchestration graph
    is independent of the storage implementation.

### 3.7 Alembic migrations

Set up Alembic and use migrations to:

-   Create the pgvector extension.
-   Create tables, foreign keys, and indexes.
-   Create the vector column with the configured dimension.
-   Add or modify indexes safely.

Do not rely on `Base.metadata.create_all()` as the production schema
migration strategy.

------------------------------------------------------------------------

## 4. PDF Ingestion and RAG Changes

Update the existing vector-store/document ingestion implementation.

### Required ingestion flow

1.  Receive the PDF through the existing upload endpoint.
2.  Validate file type and size.
3.  Save the original file to configured local/object storage.
4.  Create a `documents` row with an ingestion status.
5.  Extract text using the existing PDF parser.
6.  Preserve the existing overlapping-window chunking behavior:
    approximately 300 words per chunk with a 50-word overlap.
7.  Generate embeddings using the configured embedding model.
8.  Insert chunk text, document ID, chunk index, metadata, and embedding
    into `document_chunks`.
9.  Mark the document as indexed.
10. Return document metadata and ingestion statistics to the frontend.

### Important ingestion requirements

-   Preserve the existing deterministic, zero-LLM-call ingestion
    behavior. Do not use Gemini or OpenRouter to parse, chunk, title, or
    embed documents unless a separately approved feature requires it.
-   Preserve deterministic heading/title extraction if already
    implemented.
-   Use database transactions for related metadata/chunk writes.
-   Avoid duplicate chunks if the same document is reprocessed. Define
    whether ingestion replaces existing chunks or creates a new document
    version.
-   Handle failures and update document status with an actionable error.
-   Do not report successful indexing until database writes are
    committed.
-   Keep the existing user-facing ingestion summary, including chunk
    count and the fact that ingestion used zero LLM generation tokens.

### RAG query flow

1.  Receive the user query.
2.  Generate its embedding with the same embedding model used during
    ingestion.
3.  Retrieve relevant chunks from pgvector.
4.  Apply configured retrieval thresholds and `top_k`.
5.  Pass retrieved context and the user query to the LLM Gateway.
6.  Generate an answer grounded in the retrieved context.
7.  Return answer, source document/chunk references, retrieval scores,
    and workflow logs.
8.  Persist conversation/message and relevant telemetry.

Do not claim that a query is grounded if retrieval returns no usable
context. Preserve the existing escalation behavior for unresolved or
operational queries.

------------------------------------------------------------------------

## 5. Unified LLM Configuration

Create a centralized configuration module, such as `backend/config.py`.

Use the following provider-selection logic as the requested initial
behavior:

``` python
import os

GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if GEMINI_API_KEY:
    LLM_PROVIDER = "gemini"
    LLM_API_KEY = GEMINI_API_KEY
    LLM_BASE_URL = (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    DEFAULT_MODEL = "gemini-3.5-flash-lite"

elif OPENROUTER_API_KEY:
    LLM_PROVIDER = "openrouter"
    LLM_API_KEY = OPENROUTER_API_KEY
    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openrouter/free"

else:
    raise ValueError(
        "No LLM API key configured. Set GEMINI_API_KEY, "
        "GOOGLE_API_KEY, or OPENROUTER_API_KEY."
    )
```

Important:

-   Keep the requested provider priority: Gemini first, OpenRouter
    second.
-   Verify that the configured model ID is currently supported by the
    selected provider before deployment. Treat model names as
    configuration, not as guaranteed permanent identifiers.
-   Do not log API keys.
-   Validate missing/invalid configuration at startup and provide a
    clear error.
-   Support optional model overrides through environment variables if
    useful, without changing the default priority.

Suggested `.env.example`:

``` env
GEMINI_API_KEY=
GOOGLE_API_KEY=
OPENROUTER_API_KEY=

# Optional overrides
GEMINI_MODEL=gemini-3.5-flash-lite
OPENROUTER_MODEL=openrouter/free

DATABASE_URL=postgresql+psycopg://admin:password@localhost:5432/ai_workbench
POSTGRES_PASSWORD=change-me

# Embedding configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

The example values are development defaults only. Confirm the selected
model and its vector dimension before using them.

------------------------------------------------------------------------

## 6. LLM Client and Provider Manager

Use a common OpenAI-compatible client integration for Gemini and
OpenRouter, for example LangChain's `ChatOpenAI` configured with the
provider's base URL, API key, and model.

Create a dedicated module such as `backend/llm/llm_client.py` or
`backend/llm/provider_manager.py`.

Responsibilities:

-   Construct clients from centralized configuration.
-   Expose a consistent interface for chat/completion calls.
-   Support model and temperature configuration.
-   Support structured output/tool calling only where the selected
    provider/model supports it.
-   Keep provider-specific headers or metadata isolated inside the
    provider manager.
-   Expose provider/model identity for telemetry.
-   Avoid creating independent provider clients throughout the
    application.

Do not assume every model exposed through OpenRouter's `openrouter/free`
route supports identical context sizes, tool calling, structured output,
or response formats. Check capabilities and handle unsupported features
explicitly.

------------------------------------------------------------------------

## 7. LLM Gateway: Single Entry Point for All LLM Calls

The LLM Gateway remains the mandatory entry point for all LLM requests.

Update `gateway.py` to provide a consistent interface for:

-   RAG answer generation.
-   Guardrail LLM-as-a-Judge.
-   LLM-based routing/classification, if used.
-   Deep Agent reasoning and tool selection.
-   Background answer evaluation.
-   Any future LLM-based functionality.

No other module should call Gemini/OpenRouter directly or instantiate an
LLM client directly.

### Gateway responsibilities

1.  Accept a request with task type, messages/prompt, model settings,
    and optional metadata.
2.  Apply cache lookup where caching is safe for that task.
3.  Select the configured primary provider.
4.  Invoke the provider through the unified client.
5.  Retry transient errors according to bounded retry/backoff policy.
6.  If the primary provider has a retryable failure and a fallback key
    is configured, attempt OpenRouter.
7.  Normalize the response and usage metadata.
8.  Record provider, model, token counts, latency, estimated cost, cache
    status, and outcome.
9.  Persist usage/telemetry records to PostgreSQL.
10. Return a consistent result to the calling component.

### Runtime fallback behavior

The requested provider priority is:

-   Primary: Gemini when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is
    present.
-   Otherwise primary: OpenRouter when `OPENROUTER_API_KEY` is present.

Additionally, implement runtime fallback:

-   If Gemini is primary and OpenRouter is configured, retry eligible
    transient Gemini failures through OpenRouter.
-   Do not fallback on malformed requests, invalid input, or
    authentication/authorization failures that are unlikely to succeed
    with the same credentials.
-   Handle rate limits, timeouts, transient server errors, and provider
    unavailability with bounded retries.
-   Do not create an infinite retry loop.
-   Record both the failed primary attempt and successful fallback
    attempt.
-   If both providers fail, return a controlled application error and
    log the failure without exposing secrets.
-   If Gemini is not configured and OpenRouter is primary, do not
    attempt Gemini fallback.

### Request/task configuration

Different tasks may need different settings:

-   RAG answer: normal response generation.
-   Guardrail judge: concise structured classification.
-   Evaluator: scoring and short rationale.
-   Agent: tool calling where supported.

Use task-specific model/temperature/token settings, but route all calls
through the same gateway. Avoid using semantic cache for tasks where
caching could create unsafe or incorrect decisions, such as
user-specific guardrail judgments, unless the cache key and policy
explicitly make it safe.

### Semantic cache

Preserve the existing semantic cache behavior and its configured
similarity logic. Move persistence to PostgreSQL `cache_entries` if
appropriate.

-   Keep the existing cache-hit behavior and threshold unless a
    deliberate change is approved.
-   Record cache hits and misses.
-   Cache only responses that are safe to reuse.
-   Include relevant context/document versions, task type,
    model/configuration, and other necessary scope in the cache key to
    avoid returning stale or cross-user responses.
-   Ensure a cache hit does not incorrectly skip security checks. In
    particular, do not bypass guardrails for a new user request simply
    because a similar prompt exists in cache.
-   If the current README's claim that cache hits bypass both generation
    and the LLM judge is retained, explicitly validate that this is safe
    for the request/session policy before preserving it. Security
    screening must not be weakened.

------------------------------------------------------------------------

## 8. Update Existing AI Components

### `guardrails.py`

-   Keep the existing regex screening.
-   Send LLM-as-a-Judge requests through the Gateway.
-   Do not instantiate Gemini or OpenRouter clients here.
-   Persist relevant decision metadata if required.
-   Fail safely if the judge is unavailable; define whether the request
    is blocked, deferred, or handled under a documented policy.
-   Do not let cached responses bypass mandatory security checks.

### `workflow.py`

-   Keep the existing LangGraph architecture and routing behavior.
-   Keep RAG-first resolution and Deep Agent escalation.
-   Use a retrieval service abstraction rather than importing
    FAISS/pgvector implementation details.
-   Ensure any LLM-based router/classifier calls go through the Gateway.
-   Keep workflow state explicit and serializable where practical.
-   Return node status and logs for the existing Admin Mode
    visualization.

### `agent.py`

-   Preserve calculator and order lookup tools.
-   Make order lookup query PostgreSQL instead of relying only on a
    Python dictionary.
-   Keep the calculator deterministic and safe (AST-based evaluation or
    the existing safe implementation).
-   Route agent reasoning and model-driven tool selection through the
    Gateway.
-   Confirm tool-calling compatibility for the selected provider/model.
-   If a provider/model does not support the required tool-calling
    behavior, return a controlled error or use a documented compatible
    model/configuration. Do not silently pretend the tool was executed.

### `evaluator.py`

-   Keep evaluation asynchronous/background as currently designed.
-   Route evaluator LLM calls through the Gateway.
-   Persist scores, feedback, provider/model, and evaluation status.
-   Ensure evaluation failure does not fail or delay the
    already-delivered user response.

### `gateway.py`

-   Centralize provider calls, runtime fallback, semantic cache,
    token/cost tracking, and usage persistence.
-   Avoid duplicate telemetry writes when retries occur; distinguish
    request-level summary from provider-attempt records if needed.

### `main.py`

-   Initialize database dependencies and services.
-   Keep existing `/chat`, `/upload`, and `/health` endpoints.
-   Add or update endpoints only when required by the existing frontend
    or Admin Mode.
-   Ensure sessions are closed correctly and long-running/background
    work does not hold request-scoped database sessions indefinitely.
-   Update health checks to report API and database status without
    exposing secrets.

### `vector_store.py`

-   Replace FAISS-specific implementation with pgvector-backed
    storage/retrieval, or rename/refactor it into a database-backed
    vector repository.
-   Preserve a clean interface for ingestion and retrieval.
-   Do not change chunking behavior unless explicitly required.

------------------------------------------------------------------------

## 9. Frontend and Admin Mode

Preserve the existing User Mode and Admin Mode UI.

Update the Admin Mode telemetry to show, where available:

-   Active provider and model.
-   Primary/fallback provider attempts.
-   Provider failure and fallback events.
-   LLM request count.
-   Input/output/total tokens.
-   Estimated cost.
-   Cache hits and misses.
-   Database connection/health.
-   Number of indexed documents and chunks.
-   Retrieval top-k and returned similarity/distance values.
-   Workflow node execution logs.
-   Evaluation scores and status.

The UI must display actual backend values, not fabricated/demo values.
Clearly label estimated or mocked cost calculations.

No redesign is required unless the existing UI needs a small change to
display these additional fields.

------------------------------------------------------------------------

## 10. Environment and Deployment

### Local development

-   Run PostgreSQL + pgvector in Docker Compose.
-   Run FastAPI locally.
-   Run React/Vite locally.
-   Keep secrets in `backend/.env` or another ignored local environment
    file.
-   Provide `.env.example` with empty secret values and documented
    configuration.

### Deployment

The React frontend may remain deployed on Vercel. The FastAPI backend
and PostgreSQL database must be deployed to services that support the
required runtime and database connectivity.

For production:

-   Use managed PostgreSQL with pgvector support.
-   Configure TLS/SSL for database connections as required by the
    provider.
-   Use a connection pool suitable for serverless or container
    deployment.
-   Store uploaded PDFs in persistent object storage; do not rely on
    ephemeral application filesystem storage.
-   Configure database backups and retention.
-   Keep provider API keys and database credentials in deployment
    secrets.
-   Do not expose database credentials or LLM keys to the React
    frontend.
-   Restrict CORS to approved frontend origins.

------------------------------------------------------------------------

## 11. Testing and Acceptance Criteria

Implement or update automated tests. The existing architecture
verification suite should continue to run and should be expanded for the
new database/provider behavior.

### Database and vector tests

-   PostgreSQL connection succeeds.
-   pgvector extension is available.
-   Alembic migrations create all required tables and indexes.
-   Document metadata persists.
-   Chunks and embeddings persist.
-   Vector retrieval returns the nearest expected chunks.
-   Data remains available after backend restart.
-   Deleting/reprocessing a document follows the defined lifecycle.
-   Database errors are handled without reporting false ingestion
    success.

### LLM provider tests

-   Gemini is selected when `GEMINI_API_KEY` is set.
-   `GOOGLE_API_KEY` is accepted when `GEMINI_API_KEY` is absent.
-   OpenRouter is selected when no Gemini key is set and an OpenRouter
    key exists.
-   Startup fails clearly if no provider key is configured.
-   Runtime fallback from Gemini to OpenRouter occurs for eligible
    transient errors.
-   Invalid request/authentication errors do not trigger inappropriate
    fallback.
-   Both-provider failure returns a controlled error.
-   API keys never appear in logs or API responses.
-   Provider/model and usage metadata are recorded.

Use mocked provider clients in automated tests; do not require real paid
API calls for the test suite.

### Gateway coverage

Verify that all LLM call sites use the Gateway:

-   Guardrail judge.
-   RAG answer generation.
-   LLM-based routing/classification.
-   Deep Agent reasoning/tool selection.
-   Background evaluator.

Add tests or code-review checks to prevent direct provider calls from
being introduced elsewhere.

### Existing behavior regression tests

-   Regex guardrails still block configured malicious prompts.
-   RAG still uses overlapping 300-word chunks with 50-word overlap.
-   PDF ingestion still uses zero LLM generation calls.
-   RAG-first routing remains intact.
-   Operational/math queries can use the Deep Agent.
-   Calculator results remain deterministic.
-   Order lookup reads the database.
-   Background evaluation does not delay the user-facing response.
-   Admin Mode receives real workflow logs and telemetry.
-   Existing API contracts remain compatible with the frontend, or
    frontend changes are made in the same change.

### Definition of done

The work is complete when:

1.  PostgreSQL + pgvector replaces FAISS for vector persistence and
    retrieval.
2.  Application data listed in this specification is persisted in
    PostgreSQL.
3.  PDF ingestion and retrieval work after a backend restart.
4.  Gemini/OpenRouter configuration follows the specified priority.
5.  Runtime fallback works for eligible transient provider failures.
6.  All LLM calls pass through the LLM Gateway.
7.  Usage, cost estimates, provider identity, cache status, and
    evaluation data are persisted.
8.  Existing LangGraph and Deep Agent behavior remains functional.
9.  Existing frontend User Mode and Admin Mode continue to work.
10. Automated tests pass, including the existing architecture checks.

------------------------------------------------------------------------

## 12. Suggested Implementation Order

Implement in small, verifiable stages:

1.  Inspect the current source code and identify actual FAISS, LLM,
    cache, and telemetry call sites.
2.  Add Docker Compose PostgreSQL + pgvector and verify connectivity.
3.  Add SQLAlchemy models, database session management, and Alembic
    migrations.
4.  Implement document and chunk persistence.
5.  Implement pgvector retrieval behind a retrieval interface.
6.  Replace FAISS usage and run RAG regression tests.
7.  Add centralized LLM configuration and the unified provider client.
8.  Update the LLM Gateway to own every LLM call.
9.  Add bounded Gemini-to-OpenRouter runtime fallback.
10. Move cache, usage, logs, evaluations, conversations, messages, and
    orders to PostgreSQL as specified.
11. Update Admin Mode telemetry.
12. Run the full automated test suite and document local/deployment
    setup.

Before changing code, inspect the existing implementation and reuse its
working components. Do not assume a feature is missing merely because it
is not described in a source file; verify the actual code. Avoid
unrelated refactoring.

## 13. Important Notes

-   The README describes the current implemented behavior; preserve it
    unless this change explicitly replaces it.
-   The original build specification describes the intended
    architecture; use it as context, not as a reason to rebuild the
    project.
-   This document specifies the requested changes. It does not claim
    that the changes have already been implemented.
-   Confirm the currently supported Gemini model ID and OpenRouter
    model/route before deployment. The configured model names may need
    updating as provider catalogs change.
-   Confirm the existing embedding model and vector dimension from the
    source code before creating the database schema.
