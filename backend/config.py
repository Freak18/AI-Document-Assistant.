from __future__ import annotations

import os
from pathlib import Path

_env_path = Path(__file__).with_name(".env")
if _env_path.exists():
    for _raw in _env_path.read_text().splitlines():
        _line = _raw.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _val = _line.split("=", 1)
        os.environ.setdefault(_key.strip(), _val.strip().strip('"').strip("'"))

GEMINI_API_KEY: str = (
    os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
).strip()

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()

if GEMINI_API_KEY:
    LLM_PROVIDER = "gemini"
    LLM_API_KEY = GEMINI_API_KEY
    LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
elif OPENROUTER_API_KEY:
    LLM_PROVIDER = "openrouter"
    LLM_API_KEY = OPENROUTER_API_KEY
    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
else:
    LLM_PROVIDER = "local-fallback"
    LLM_API_KEY = ""
    LLM_BASE_URL = ""
    DEFAULT_MODEL = "local-fallback"

FALLBACK_PROVIDER: str = "openrouter" if (LLM_PROVIDER == "gemini" and OPENROUTER_API_KEY) else ""
FALLBACK_API_KEY: str = OPENROUTER_API_KEY if FALLBACK_PROVIDER == "openrouter" else ""
FALLBACK_BASE_URL: str = "https://openrouter.ai/api/v1" if FALLBACK_PROVIDER == "openrouter" else ""
FALLBACK_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

DATABASE_URL: str = os.getenv("DATABASE_URL", "")

LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BACKOFF_SECONDS: float = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "0.75"))
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIMENSION", os.getenv("PGVECTOR_DIM", "384")))

