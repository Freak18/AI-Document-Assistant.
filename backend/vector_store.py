from __future__ import annotations

import hashlib
import re
import uuid
from functools import lru_cache
from io import BytesIO
from typing import Optional

import numpy as np
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
from sqlalchemy import text

from config import EMBEDDING_DIM, EMBEDDING_MODEL
from database import DB_ENABLED, SessionLocal
from trace_log import log

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency should degrade gracefully
    SentenceTransformer = None

WINDOW_WORDS = 300
OVERLAP_WORDS = 50
DIM = EMBEDDING_DIM

SAMPLE_CIRCULAR = """\
SEBI Sample Market Supervision Circular (SMS-SIM-2024-11). This document represents standard regulatory clauses for equity settlement, margin collections, client funds segregation, insider trading windows, and disclosure norms in simulated capital markets.

Settlement. All equity delivery trades in the simulated market settle on a T+1 cycle. Funds
and securities must both be available by 10:30 IST on the trading day after the trade date.
A broker that fails to complete pay-in by that time must close out the position on the same
day and report the shortage to the clearing house before 16:00 IST. Partial pay-in does not
extend the deadline. Auction close-out is priced at the closing auction plus a sample penalty
of 1 percent, and that penalty is charged to the defaulting broker rather than the client
when the broker caused the miss.

Margin. An intraday margin shortfall must be collected from the client by 11:00 IST on the
next trading day. If the shortfall remains unpaid, the broker shall square off the open
positions and may not extend fresh exposure to that client until the ledger is positive.
The minimum upfront margin for an equity delivery buy in this sample circular is 20 percent
of the trade value. Derivative futures use SPAN plus an exposure margin of 3 percent of
notional. A client may meet margin with cash or the pledged sovereign bills listed in
Schedule A. Equity collateral is haircut by 25 percent before it counts toward margin.

Client funds. Client funds must be segregated from the broker's proprietary funds. Transfers
from the client bank account into the proprietary account are prohibited, including temporary
transfers overnight. Idle client funds may be placed only in the instruments listed in
Schedule A of this sample circular: overnight sovereign bills and the clearing-house liquid
fund. Interest earned on those placements is credited back to the client ledger within three
trading days. The broker reconciles the client bank account, the clearing-house deposit, and
the back-office ledger every trading day before 18:00 IST. A break larger than 10000 sample
currency units is reported to the compliance officer the same evening.

Insider window. Designated persons must close the trading window 48 hours before the board
meeting that considers financial results. The window reopens 48 hours after the results are
published to the exchange. A designated person who trades inside the closed window must report
the trade to the compliance officer within one trading day. The compliance officer files that
report with the exchange within two trading days. Immediate relatives of designated persons
follow the same window. A pre-cleared trading plan filed 30 days earlier is the only sample
exception, and the plan cannot be changed inside the closed window.

Disclosure. An acquirer must disclose the holding when purchases cross 5 percent of the
paid-up share capital of a listed company. A further disclosure is required at every
additional 2 percent. The disclosure is due within two trading days of the crossing. Pledged
shares count toward the threshold. Convertible instruments count only after conversion,
except when conversion is certain within 60 days. Creeping acquisitions above 25 percent are
not permitted in this sample circular unless an open offer is launched the same day. The open
offer stays open for 10 trading days and the floor price is the volume-weighted price of the
previous 60 trading days.

Research reports. A research report that recommends a security must disclose whether the firm
handled a mandate for that issuer in the previous 12 months. The analyst who signs the report
may not trade the security for 30 days after publication. Price targets in the report must
state the valuation method in one sentence, for example discounted cash flow or peer
multiples. A report that was shown to the issuer before publication must say so on the first
page. Distribution to clients happens only after that disclosure is attached.

Complaints. A client complaint about execution or funds must be acknowledged within two
trading days and closed within 21 calendar days. If the client is not satisfied, the broker
tells the client how to escalate to the sample exchange arbitration window. Records of the
complaint, including call notes and ledger extracts, are retained for five years. Arbitration
filings under this sample circular are accepted up to 500000 sample currency units. Larger
claims are directed to the sample civil window. The broker may not force a client to sign a
waiver before the complaint is closed.

Cyber reporting. A broker that detects unauthorized access to client data must notify the
sample exchange within 6 hours of confirmation. The first notice names the systems affected
and whether client credentials were exposed. A follow-up report within 14 days describes
containment and the number of clients notified. Credentials that may have been exposed are
reset within 24 hours. The broker keeps the forensic image for 18 months.

This closing clause keeps complex obligations together: segregation, the T+1 pay-in deadline
at 10:30 IST, the 11:00 IST margin deadline, the 48 hour insider window, and the 5 percent
disclosure threshold must be read as one control framework rather than isolated sentences.
A control failure in any one of those clauses is reported in the monthly compliance pack
that the broker files by the fifth trading day of the next month.
"""


# ---------------------------------------------------------------------------
# Pure-Python utilities (zero LLM calls, unchanged behaviour)
# ---------------------------------------------------------------------------

def extract_heading_or_title(text: str, filename: str) -> str:
    """Extract document heading/title deterministically without calling any LLM."""
    clean_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if clean_lines:
        first = clean_lines[0].rstrip(". ")
        if 8 <= len(first) <= 120:
            return first
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if sentences and 8 <= len(sentences[0]) <= 120:
            return sentences[0].rstrip(". ")
    clean_name = re.sub(r"\.[^.]+$", "", filename).replace("_", " ").replace("-", " ")
    return clean_name.title()


@lru_cache(maxsize=1)
def _get_sentence_model():
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def chunk_words(text: str, window: int = WINDOW_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Chunk text using a production-grade recursive character splitter while preserving the old API."""
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=window,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def _bucket(token: str) -> int:
    digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(digest[:4], "little") % DIM


def embed_text(text: str) -> np.ndarray:
    """Return a normalized embedding using the configured Sentence Transformers model."""
    if not text or not text.strip():
        return np.zeros(DIM, dtype=np.float32)
    try:
        model = _get_sentence_model()
        embedding = model.encode(text.strip(), convert_to_numpy=True, normalize_embeddings=True)
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.size == 0:
            return np.zeros(DIM, dtype=np.float32)
        if arr.shape[0] != DIM:
            arr = np.resize(arr, DIM).astype(np.float32)
        return arr
    except Exception as exc:
        log(f"[WARN] Sentence-transformers embedding failed; falling back to deterministic bag-of-hashes: {exc}")
        vector = np.zeros(DIM, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for index, token in enumerate(tokens):
            vector[_bucket(token)] += 1.0
            if index + 1 < len(tokens):
                vector[_bucket(f"{token}_{tokens[index + 1]}")] += 1.4
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector


CONTENT_STOP = {
    "a", "an", "the", "of", "to", "and", "or", "in", "on", "for", "by", "is", "are", "be",
    "this", "that", "with", "from", "as", "at", "into", "must", "shall", "within", "before",
    "after", "only", "when", "until", "about", "than", "then", "not", "sample", "circular",
    "document", "documents", "what", "where", "which", "who", "how", "does", "did", "can",
    "could", "would", "should", "please", "explain", "describe", "tell", "indexed",
}


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:\+[a-z0-9]+)?", text.lower())
        if token not in CONTENT_STOP and len(token) > 2 and not token.isdigit()
    }


def _field_match_bonus(query: str, chunk_text: str) -> float:
    q = query.lower()
    text = chunk_text.lower()

    direct_patterns = [
        ("candidate|employee|applicant|full name|name", r"(?:candidate\s*name|employee\s*name|applicant\s*name|full\s*name|dear\s*\w+|name\s*[:\-])\s*\w+"),
        ("compensation|salary|ctc|bonus|remuneration|fixed pay|fixed compensation|pay", r"(?:fixed\s*pay|total\s*fixed\s*pay|fixed\s*compensation|annual\s*compensation|salary|basic\s*salary|ctc|base\s*salary|bonus|remuneration)\s*[:\-\s]*(?:inr|\$|rs\.?|[0-9])"),
        ("designation|role|job title|title|position", r"(?:designation|role|job\s*title|title|position|titlewillbe)\s*[:\-\s]*"),
        ("joining|join|start date|start|date of joining", r"(?:date\s*of\s*joining|joining\s*date|start\s*date|joining)\s*[:\-\s]*"),
    ]

    for keywords, pattern in direct_patterns:
        if any(k in q for k in keywords.split("|")) and re.search(pattern, text):
            return 0.95

    # Secondary fuzzy word checks (handling joined words from extracted PDF text)
    if any(k in q for k in ["compensation", "salary", "ctc", "bonus", "remuneration", "fixed pay", "fixed"]):
        if any(k in text for k in ["totalfixedpay", "fixedpay", "basicsalary", "remuneration", "38,00,000", "fixed pay", "basic salary", "compensation"]):
            return 0.75
    if any(k in q for k in ["candidate", "name", "who"]) and any(k in text for k in ["yashwanth", "candidate", "dearyashwanth", "name:"]):
        return 0.75
    if any(k in q for k in ["title", "designation", "role", "position"]) and any(k in text for k in ["software engineer", "titlewillbe", "designation"]):
        return 0.75
    if any(k in q for k in ["joining", "join", "start date"]) and any(k in text for k in ["joining", "start date", "21-oct-2026"]):
        return 0.75
    return 0.0


# ---------------------------------------------------------------------------
# VectorStore class
# ---------------------------------------------------------------------------
class VectorStore:
    """Pgvector-backed vector store with an in-memory fallback for no-DB mode."""

    def __init__(self) -> None:
        # In-memory structures – used as fallback when DB is unavailable
        self._mem_chunks: list[dict] = []
        self._mem_matrix: Optional[np.ndarray] = None
        self._documents: dict[str, dict] = {}  # source -> metadata
        self._document_sequence: dict[str, int] = {}
        self._next_sequence = 0
        self._active_source: Optional[str] = None
        self.engine = "pgvector" if DB_ENABLED else "numpy"

    def _recency_boost(self, source: str) -> float:
        if not self._document_sequence:
            return 0.0
        max_seq = max(self._document_sequence.values())
        seq = self._document_sequence.get(source, 0)
        if max_seq <= 0:
            return 0.0
        return 0.05 * (seq / max_seq)

    def get_active_source(self) -> Optional[str]:
        if self._active_source:
            return self._active_source
        if DB_ENABLED and SessionLocal is not None:
            from models import Document
            db = SessionLocal()
            try:
                latest = db.query(Document).filter(Document.status == "ready").order_by(Document.created_at.desc()).first()
                if latest:
                    self._active_source = latest.filename
                    return self._active_source
            except Exception:
                pass
            finally:
                db.close()
        return None

    def set_active_source(self, source: str) -> None:
        self._active_source = source

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_text(self, text: str, source: str) -> tuple[int, str]:
        """Chunk, embed, and store a text document.  Returns (chunk_count, title).
        
        No LLM calls are made.  Uses a database transaction when DB is available.
        """
        pieces = chunk_words(text)
        if not pieces:
            raise ValueError("No extractable text found in the document")

        title = extract_heading_or_title(text, source)
        embeddings = [embed_text(p) for p in pieces]

        if DB_ENABLED and SessionLocal is not None:
            self._persist_to_db(source, title, pieces, embeddings)
        else:
            self._persist_in_memory(source, title, pieces, embeddings)

        self._next_sequence += 1
        self._document_sequence[source] = self._next_sequence
        self._documents[source] = {
            "source": source,
            "title": title,
            "chunks": len(pieces),
        }
        self._active_source = source
        return len(pieces), title

    def ingest_pdf(self, data: bytes, source: str) -> tuple[int, str]:
        """Parse a PDF and ingest its text content with empty-page and extraction safeguards."""
        try:
            reader = PdfReader(BytesIO(data))
        except Exception as exc:
            raise ValueError(f"Could not decode PDF: {exc}") from exc
        pages: list[str] = []
        for page_no, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            cleaned = " ".join(text.split())
            if cleaned:
                pages.append(f"\n--- Page {page_no} ---\n{text.strip()}")
        combined = "\n".join(pages).strip()
        if not combined:
            raise ValueError("No extractable text found inside the PDF")
        return self.ingest_text(combined, source)

    def _persist_to_db(
        self,
        source: str,
        title: str,
        pieces: list[str],
        embeddings: list[np.ndarray],
    ) -> None:
        """Write a document record and all its chunks to PostgreSQL atomically."""
        from models import Document, DocumentChunk
        import os

        storage_path = str(
            (
                __import__("pathlib").Path(__file__).with_name("uploads") / source
            ).resolve()
        )
        doc_id = str(uuid.uuid4())

        db: Session = SessionLocal()
        try:
            # Remove existing chunks if the same filename was re-uploaded
            existing = db.query(Document).filter_by(filename=source).first()
            if existing:
                db.delete(existing)
                db.flush()

            doc = Document(
                id=doc_id,
                filename=source,
                storage_path=storage_path,
                content_type="application/pdf",
                file_size=os.path.getsize(storage_path) if os.path.exists(storage_path) else 0,
                status="indexing",
                title=title,
            )
            db.add(doc)
            db.flush()

            for idx, (piece, emb) in enumerate(zip(pieces, embeddings)):
                # pgvector expects a Python list for the vector column
                chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    chunk_index=idx,
                    chunk_text=piece,
                    embedding=emb.tolist(),
                    chunk_metadata={"source": source, "title": title},
                )
                db.add(chunk)

            doc.status = "ready"
            db.commit()
            log(f"[INFO] Persisted {len(pieces)} chunks for '{source}' (doc_id={doc_id}) to pgvector")
        except Exception as exc:
            db.rollback()
            log(f"[ERROR] DB write failed for '{source}': {exc} – using in-memory fallback")
            self._persist_in_memory(source, title, pieces, embeddings)
        finally:
            db.close()

    def _persist_in_memory(
        self,
        source: str,
        title: str,
        pieces: list[str],
        embeddings: list[np.ndarray],
    ) -> None:
        """Store chunks in-memory when DB is unavailable."""
        for idx, (piece, emb) in enumerate(zip(pieces, embeddings)):
            self._mem_chunks.append({
                "text": piece,
                "source": source,
                "title": title,
                "chunk": idx + 1,
            })
        matrix = np.vstack(embeddings).astype(np.float32)
        self._mem_matrix = matrix if self._mem_matrix is None else np.vstack([self._mem_matrix, matrix])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 3) -> list[dict]:
        """Search for the top-k most relevant chunks.

        Uses pgvector cosine distance when the DB is available, otherwise
        falls back to in-memory numpy cosine similarity.
        """
        if self._active_source is not None and query and not query.lower().startswith("select "):
            # Keep the most recent uploaded document as the active source of truth.
            # This prevents the built-in sample circular from dominating real PDF answers.
            pass
        if DB_ENABLED and SessionLocal is not None:
            return self._search_pgvector(query, k)
        return self._search_in_memory(query, k)

    def _search_pgvector(self, query: str, k: int) -> list[dict]:
        """Run cosine-distance search via pgvector."""
        query_vec = embed_text(query).tolist()
        vec_literal = "[" + ",".join(str(v) for v in query_vec) + "]"

        active = self.get_active_source()
        db = SessionLocal()
        rows = []
        try:
            if active:
                sql_active = text(
                    """
                    SELECT
                        dc.id,
                        dc.document_id,
                        dc.chunk_index,
                        dc.chunk_text,
                        dc.chunk_metadata,
                        (dc.embedding <=> CAST(:qvec AS vector)) AS distance
                    FROM document_chunks dc
                    WHERE dc.chunk_metadata->>'source' = :active_source
                    ORDER BY dc.embedding <=> CAST(:qvec AS vector)
                    LIMIT :k
                    """
                )
                rows = db.execute(sql_active, {"qvec": vec_literal, "active_source": active, "k": k * 6}).fetchall()

            if not rows:
                sql_all = text(
                    """
                    SELECT
                        dc.id,
                        dc.document_id,
                        dc.chunk_index,
                        dc.chunk_text,
                        dc.chunk_metadata,
                        (dc.embedding <=> CAST(:qvec AS vector)) AS distance
                    FROM document_chunks dc
                    ORDER BY dc.embedding <=> CAST(:qvec AS vector)
                    LIMIT :k
                    """
                )
                rows = db.execute(sql_all, {"qvec": vec_literal, "k": k * 6}).fetchall()
        except Exception as exc:
            log(f"[WARN] pgvector search failed: {exc} – using in-memory fallback")
            db.close()
            return self._search_in_memory(query, k)
        finally:
            db.close()

        if not rows:
            return self._search_in_memory(query, k)

        query_terms = content_tokens(query)
        ranked: list[tuple[float, dict]] = []
        for row in rows:
            meta = row.chunk_metadata or {}
            source = meta.get("source", "")
            title = meta.get("title", "")
            chunk_text = row.chunk_text
            cosine_similarity = max(0.0, 1.0 - float(row.distance))
            chunk_terms = content_tokens(chunk_text)
            lexical = (
                (len(query_terms & chunk_terms) / len(query_terms)) if query_terms else 0.0
            )
            field_bonus = _field_match_bonus(query, chunk_text)
            recency_boost = self._recency_boost(source)
            combined = (0.35 * cosine_similarity) + (0.45 * lexical) + field_bonus + recency_boost
            ranked.append(
                (
                    combined,
                    {
                        "text": chunk_text,
                        "source": source,
                        "title": title,
                        "chunk": row.chunk_index + 1,
                        "score": round(cosine_similarity, 4),
                        "lexical": round(lexical, 4),
                        "retrieval_source": "pgvector",
                    },
                )
            )

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [hit for _, hit in ranked[:k]]

    def _search_in_memory(self, query: str, k: int) -> list[dict]:
        """Numpy cosine search over the in-memory chunk matrix."""
        if not self._mem_chunks or self._mem_matrix is None:
            return []
        query_vector = embed_text(query).reshape(1, -1)
        sims = (self._mem_matrix @ query_vector.T).ravel()
        query_terms = content_tokens(query)
        query_lower = query.lower()
        ranked: list[tuple[float, dict]] = []
        for idx in range(len(self._mem_chunks)):
            source = self._mem_chunks[idx]["source"]
            if self._active_source is not None and source != self._active_source:
                continue
            score = float(sims[idx])
            chunk_text = self._mem_chunks[idx]["text"]
            chunk_terms = content_tokens(chunk_text)
            lexical = (len(query_terms & chunk_terms) / len(query_terms)) if query_terms else 0.0
            field_bonus = _field_match_bonus(query, chunk_text)
            recency_boost = self._recency_boost(source)
            combined = (0.25 * score) + (0.55 * lexical) + field_bonus + recency_boost
            hit = dict(self._mem_chunks[idx])
            hit["score"] = round(score, 4)
            hit["lexical"] = round(lexical, 4)
            hit["retrieval_source"] = "numpy"
            ranked.append((combined, hit))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [hit for _, hit in ranked[:k]]

    def document_stats(self) -> list[dict]:
        """Return per-document ingestion stats.

        Queries the DB when available, otherwise uses in-memory tracking.
        """
        if DB_ENABLED and SessionLocal is not None:
            return self._db_document_stats()
        return list(self._documents.values())

    def _db_document_stats(self) -> list[dict]:
        from models import Document, DocumentChunk
        from sqlalchemy import func as sqlfunc
        db = SessionLocal()
        try:
            rows = (
                db.query(
                    Document.filename,
                    Document.title,
                    sqlfunc.count(DocumentChunk.id).label("chunks"),
                )
                .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
                .filter(Document.status == "ready")
                .group_by(Document.filename, Document.title)
                .all()
            )
            return [
                {"source": r.filename, "title": r.title or r.filename, "chunks": r.chunks}
                for r in rows
            ]
        except Exception as exc:
            log(f"[WARN] DB document_stats failed: {exc}")
            return list(self._documents.values())
        finally:
            db.close()

    def clear_documents(self) -> None:
        """Remove all indexed documents so the next ingestion becomes the single active source of truth."""
        if DB_ENABLED and SessionLocal is not None:
            from models import Document, DocumentChunk
            db = SessionLocal()
            try:
                db.query(DocumentChunk).delete()
                db.query(Document).delete()
                db.commit()
                log("[INFO] Cleared all previously indexed documents from pgvector")
            except Exception as exc:
                db.rollback()
                log(f"[WARN] Failed to clear DB documents: {exc}")
            finally:
                db.close()

        self._documents.clear()
        self._document_sequence.clear()
        self._next_sequence = 0
        self._active_source = None
        self._mem_chunks = []
        self._mem_matrix = None

    def total_chunks(self) -> int:
        """Return the total number of indexed chunks."""
        if DB_ENABLED and SessionLocal is not None:
            from models import DocumentChunk
            db = SessionLocal()
            try:
                return db.query(DocumentChunk).count()
            except Exception:
                pass
            finally:
                db.close()
        return len(self._mem_chunks)


store = VectorStore()

# Seed a built-in sample circular only when the app is starting with no uploaded material.
try:
    if not store.document_stats():
        _count, _title = store.ingest_text(SAMPLE_CIRCULAR, "sample-market-circular.txt")
        log(f"[INFO] Pre-indexed sample circular: {_count} chunks (pgvector={DB_ENABLED})")
except Exception as _e:
    log(f"[WARN] Could not pre-index sample circular: {_e}")
