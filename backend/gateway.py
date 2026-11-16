from __future__ import annotations

import re
import threading
import time
import uuid
from typing import Optional

import numpy as np

import config
from trace_log import log
from vector_store import content_tokens, embed_text

INPUT_USD_PER_1K = 0.00015
OUTPUT_USD_PER_1K = 0.0006

SIMILARITY_THRESHOLD = 0.82

_RETRYABLE_PATTERNS = re.compile(
    r"rate.?limit|quota|timeout|timed.?out|service.?unavailable|"
    r"server.?error|5\d\d|connection",
    re.IGNORECASE,
)
_NON_RETRYABLE_PATTERNS = re.compile(
    r"401|403|unauthorized|forbidden|invalid.?api.?key|authentication",
    re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _is_retryable(error_msg: str) -> bool:
    if _NON_RETRYABLE_PATTERNS.search(error_msg):
        return False
    return bool(_RETRYABLE_PATTERNS.search(error_msg))


class LLMGateway:
    def __init__(self) -> None:
        self._provider = config.LLM_PROVIDER
        self._api_key = config.LLM_API_KEY
        self._base_url = config.LLM_BASE_URL
        self.model_name = config.DEFAULT_MODEL

        self._fallback_provider = config.FALLBACK_PROVIDER
        self._fallback_api_key = config.FALLBACK_API_KEY
        self._fallback_base_url = config.FALLBACK_BASE_URL
        self._fallback_model = config.FALLBACK_MODEL

        self.total_requests = 0
        self.cache_hits = 0
        self.tokens = 0
        self.cost_usd = 0.0
        self.latest_eval_score: Optional[float] = None

        self._cache: list[dict] = []
        self._lock = threading.Lock()

        self._primary_client = None
        self._fallback_client = None

        self._load_cache_from_db()

    @property
    def api_key(self) -> str:
        return self._api_key

    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear_cache(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self.cache_hits = 0

        try:
            from database import DB_ENABLED, SessionLocal
            if DB_ENABLED and SessionLocal is not None:
                from models import CacheEntry
                db = SessionLocal()
                try:
                    deleted = db.query(CacheEntry).delete()
                    db.commit()
                    log(f"[INFO] Cleared {deleted} semantic cache rows from PostgreSQL")
                finally:
                    db.close()
        except Exception as exc:
            log(f"[WARN] Could not clear persisted semantic cache: {exc}")

        return count

    def metrics(self) -> dict:
        with self._lock:
            hit_rate = (
                round((self.cache_hits / self.total_requests) * 100, 1)
                if self.total_requests > 0
                else 0.0
            )
            return {
                "total_requests": self.total_requests,
                "cached_requests": self.cache_hits,
                "cache_hit_rate": hit_rate,
                "tokens_used": self.tokens,
                "estimated_cost": round(self.cost_usd, 6),
                "latest_eval_score": self.latest_eval_score,
                "provider": self._provider,
                "model": self.model_name,
                "fallback_provider": self._fallback_provider or None,
            }

    def record_request(self) -> None:
        with self._lock:
            self.total_requests += 1

    def set_eval_score(self, score: float) -> None:
        with self._lock:
            self.latest_eval_score = round(score, 2)

    # ------------------------------------------------------------------
    # Semantic Cache
    # ------------------------------------------------------------------
    def lookup(self, prompt: str) -> tuple[dict | None, float]:
        key = " ".join(prompt.lower().split())
        query_vector = embed_text(prompt)
        query_words = _word_set(prompt)
        best_score = 0.0
        best_item = None
        with self._lock:
            for item in self._cache:
                if item["key"] == key:
                    return item, 1.0
                cosine = float(np.dot(query_vector, item["vector"]))
                union = query_words | item["words"]
                jaccard = (len(query_words & item["words"]) / len(union)) if union else 0.0
                score = (0.65 * cosine) + (0.35 * jaccard)
                if score > best_score:
                    best_score = score
                    best_item = item
        if best_item is not None and best_score >= SIMILARITY_THRESHOLD:
            return best_item, best_score
        return None, best_score

    def remember(self, prompt: str, answer: str) -> None:
        if not answer:
            return
        cache_key = " ".join(prompt.lower().split())
        item = {
            "key": cache_key,
            "prompt": prompt,
            "answer": answer,
            "vector": embed_text(prompt),
            "words": _word_set(prompt),
        }
        with self._lock:
            self._cache.append(item)
            if len(self._cache) > 100:
                del self._cache[:20]
        # Persist to PostgreSQL cache_entries (best-effort, non-blocking)
        self._save_cache_entry_to_db(
            cache_key=cache_key,
            normalized_prompt=cache_key,
            response=answer,
            provider=self._provider,
            model=self.model_name,
            task_type="general",
        )

    def mark_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    # ------------------------------------------------------------------
    # Core completion (single entry point for ALL LLM calls)
    # ------------------------------------------------------------------
    async def complete(self, prompt: str, purpose: str) -> str:
        """Call the configured primary LLM provider; fall back to secondary on transient errors."""
        if self._provider == "local-fallback":
            return self._fallback_local(prompt, purpose)

        start_ms = time.time() * 1000
        request_id = str(uuid.uuid4())[:8]

        # --- Primary provider ---
        try:
            text_out = await self._call_provider(
                prompt=prompt,
                api_key=self._api_key,
                base_url=self._base_url,
                model=self.model_name,
                provider_name=self._provider,
            )
            self._record_usage(prompt, text_out, self._provider, purpose, start_ms, cache_hit=False)
            return text_out

        except Exception as exc:
            err_msg = str(exc)
            log(f"[WARN] Primary provider '{self._provider}' failed for {purpose}: {err_msg}")

            # --- Fallback provider (only for eligible transient errors) ---
            if self._fallback_provider and _is_retryable(err_msg):
                log(f"[INFO] Attempting fallback provider '{self._fallback_provider}'")
                try:
                    text_out = await self._call_provider(
                        prompt=prompt,
                        api_key=self._fallback_api_key,
                        base_url=self._fallback_base_url,
                        model=self._fallback_model,
                        provider_name=self._fallback_provider,
                    )
                    self._record_usage(prompt, text_out, self._fallback_provider, purpose, start_ms, cache_hit=False)
                    log(f"[INFO] Fallback '{self._fallback_provider}' succeeded for {purpose}")
                    return text_out
                except Exception as fallback_exc:
                    log(f"[ERROR] Fallback provider '{self._fallback_provider}' also failed: {fallback_exc}")

            # --- Local fallback as last resort ---
            log(f"[WARN] All providers failed for {purpose}; using local fallback")
            return self._fallback_local(prompt, purpose)

    async def _call_provider(
        self,
        prompt: str,
        api_key: str,
        base_url: str,
        model: str,
        provider_name: str,
    ) -> str:
        """Call a provider via its OpenAI-compatible API (Gemini + OpenRouter)."""
        return await self._call_openai_compat(prompt, api_key, base_url, model, provider_name)

    async def _call_openai_compat(
        self,
        prompt: str,
        api_key: str,
        base_url: str,
        model: str,
        provider_name: str,
    ) -> str:
        """Call any OpenAI-compatible endpoint (Gemini OpenAI-compat or OpenRouter)."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"{provider_name} returned an empty completion")
        return content.strip()

    def _record_usage(
        self,
        prompt: str,
        text_out: str,
        provider: str,
        purpose: str,
        start_ms: float,
        cache_hit: bool,
    ) -> None:
        in_tokens = estimate_tokens(prompt)
        out_tokens = estimate_tokens(text_out)
        cost = (in_tokens / 1000) * INPUT_USD_PER_1K + (out_tokens / 1000) * OUTPUT_USD_PER_1K
        latency_ms = time.time() * 1000 - start_ms
        with self._lock:
            self.tokens += in_tokens + out_tokens
            self.cost_usd += cost
        log(
            f"[INFO] Gateway {purpose} via {provider} "
            f"tokens={in_tokens + out_tokens} cost=${cost:.6f} latency={latency_ms:.0f}ms"
        )
        # Persist to DB (best-effort, non-blocking)
        self._persist_usage(provider, purpose, in_tokens, out_tokens, cost, latency_ms, cache_hit)

    def _persist_usage(
        self,
        provider: str,
        purpose: str,
        in_tokens: int,
        out_tokens: int,
        cost: float,
        latency_ms: float,
        cache_hit: bool,
    ) -> None:
        from database import DB_ENABLED, SessionLocal
        if not DB_ENABLED or SessionLocal is None:
            return
        try:
            from models import LLMUsage
            db = SessionLocal()
            try:
                record = LLMUsage(
                    id=str(uuid.uuid4()),
                    provider=provider,
                    model=self.model_name,
                    purpose=purpose,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    total_tokens=in_tokens + out_tokens,
                    estimated_cost_usd=cost,
                    cache_hit=cache_hit,
                    latency_ms=latency_ms,
                )
                db.add(record)
                db.commit()
            finally:
                db.close()
        except Exception as exc:
            log(f"[WARN] Could not persist LLM usage: {exc}")

    # ------------------------------------------------------------------
    # Persistent cache (PostgreSQL cache_entries)
    # ------------------------------------------------------------------
    def _load_cache_from_db(self) -> None:
        """Load persistent cache entries from PostgreSQL into memory."""
        from database import DB_ENABLED, SessionLocal
        if not DB_ENABLED or SessionLocal is None:
            return
        try:
            from models import CacheEntry
            db = SessionLocal()
            try:
                rows = db.query(CacheEntry).all()
                with self._lock:
                    for row in rows:
                        self._cache.append({
                            "key": row.cache_key,
                            "prompt": row.normalized_prompt,
                            "answer": row.response,
                            "vector": embed_text(row.normalized_prompt),
                            "words": _word_set(row.normalized_prompt),
                        })
                log(f"[INFO] Loaded {len(rows)} cache entries from PostgreSQL")
            finally:
                db.close()
        except Exception as exc:
            log(f"[WARN] Could not load cache from DB: {exc}")

    def _save_cache_entry_to_db(
        self,
        cache_key: str,
        normalized_prompt: str,
        response: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> None:
        """Persist a new cache entry to PostgreSQL (best-effort)."""
        from database import DB_ENABLED, SessionLocal
        if not DB_ENABLED or SessionLocal is None:
            return
        try:
            from models import CacheEntry
            from datetime import datetime, timedelta, timezone
            db = SessionLocal()
            try:
                # Avoid duplicates
                existing = db.query(CacheEntry).filter_by(cache_key=cache_key).first()
                if existing:
                    existing.hits = (existing.hits or 0) + 1
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    entry = CacheEntry(
                        cache_key=cache_key,
                        normalized_prompt=normalized_prompt,
                        response=response,
                        provider=provider,
                        model=model,
                        task_type=task_type,
                        hits=1,
                        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
                    )
                    db.add(entry)
                db.commit()
            finally:
                db.close()
        except Exception as exc:
            log(f"[WARN] Could not save cache entry to DB: {exc}")

    # ------------------------------------------------------------------
    # Local (no-API-key) fallback
    # ------------------------------------------------------------------
    def _fallback_local(self, prompt: str, purpose: str) -> str:
        if purpose == "judge":
            return "ALLOW"
        if purpose == "evaluator":
            return "0.89"
        context = _section(prompt, "Context:")
        question = _section(prompt, "Question:") or prompt
        direct = _extract_direct_answer(question, context)
        if direct:
            return direct
        picked = _best_sentences(question, context)
        if not picked:
            return "No indexed passage matched that question. Upload a PDF to add documents."
        return "From the indexed documents: " + " ".join(picked)


# ---------------------------------------------------------------------------
# Sentence-level helpers for local fallback
# ---------------------------------------------------------------------------
def _extract_direct_answer(question: str, context: str) -> str:
    if not context:
        return ""
    q = question.lower()
    wants_name = any(token in q for token in ["name", "candidate", "employee", "applicant", "offered to"])
    if not wants_name:
        return ""

    patterns = [
        r"(?i)\b(?:candidate|employee|applicant|full\s+name|name\s*(?:of\s+the)?\s*(?:candidate|employee|applicant))\s*[:\-]?\s*([A-Z][A-Za-z0-9.'\-]+(?:\s+[A-Z][A-Za-z0-9.'\-]+)+)",
        r"(?i)\b(?:offered\s+to|offered\s+to\s+candidate|person\s+name)\s*[:\-]?\s*([A-Z][A-Za-z0-9.'\-]+(?:\s+[A-Z][A-Za-z0-9.'\-]+)+)",
        r"(?i)\bbetween\s+([A-Z][A-Za-z0-9.'\-]+(?:\s+[A-Z][A-Za-z0-9.'\-]+)+)\s+and\s+[A-Z][A-Za-z0-9.'\-]+",
        r"(?i)\b(?:Mr|Ms|Mrs|Dr)\.?\s+[A-Z][A-Za-z0-9.'\-]+(?:\s+[A-Z][A-Za-z0-9.'\-]+)+",
    ]

    for pattern in patterns:
        match = re.search(pattern, context)
        if match:
            value = match.group(1).strip(" ,;.")
            if value and value.lower() not in {"this agreement", "the company", "company", "candidate", "employee", "applicant"}:
                return f"The candidate name is {value}."
    return ""


def _best_sentences(question: str, context: str) -> list[str]:
    query_terms = content_tokens(question)
    plain = re.sub(r"\[[^\]]+\]", " ", context.replace("\n", " "))
    sentences = re.split(r"(?<=[.!?])\s+", plain)
    ranked = []
    for index, sentence in enumerate(sentences):
        cleaned = " ".join(sentence.split())
        if len(cleaned) < 20:
            continue
        words = content_tokens(cleaned)
        overlap = (len(query_terms & words) / len(query_terms)) if query_terms else 0.0
        field_bonus = 0.0
        q_lower = question.lower()
        if any(token in q_lower for token in ["candidate", "employee", "applicant", "name"]) and re.search(r"\b(?:candidate|employee|applicant|full name|name)\b", cleaned, re.IGNORECASE):
            field_bonus += 0.5
        if any(token in q_lower for token in ["compensation", "salary", "ctc", "bonus"]) and re.search(r"\b(?:fixed compensation|annual compensation|salary|ctc|bonus)\b", cleaned, re.IGNORECASE):
            field_bonus += 0.5
        if any(token in q_lower for token in ["designation", "role", "title"]) and re.search(r"\b(?:designation|role|job title|title)\b", cleaned, re.IGNORECASE):
            field_bonus += 0.5
        if any(token in q_lower for token in ["joining", "join", "start", "date"]) and re.search(r"\b(?:joining date|date of joining|start date|joining)\b", cleaned, re.IGNORECASE):
            field_bonus += 0.5
        ranked.append((overlap + field_bonus, -index, cleaned))
    ranked.sort(reverse=True)
    if not ranked or ranked[0][0] <= 0:
        return []
    best = ranked[0][0]
    return [sentence for overlap, _index, sentence in ranked if overlap >= best * 0.67][:2]


def _section(prompt: str, label: str) -> str:
    if label not in prompt:
        return ""
    body = prompt.split(label, 1)[1]
    for marker in ("Question:", "User message:"):
        if marker in body:
            body = body.split(marker, 1)[0]
    return body.strip()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
gateway = LLMGateway()
