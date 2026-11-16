from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Index, Integer, String, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from database import Base

try:
    from pgvector.sqlalchemy import Vector as PgVector
    _VECTOR_AVAILABLE = True
except ImportError:
    _VECTOR_AVAILABLE = False
    PgVector = None

from config import EMBEDDING_DIM


def _uuid() -> str:
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    content_type = Column(String, nullable=False, default="application/pdf")
    file_size = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="uploaded")
    title = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index("ix_documents_status", "status"),
        Index("ix_documents_filename", "filename"),
    )


def _make_embedding_column():
    if _VECTOR_AVAILABLE and PgVector is not None:
        return Column(PgVector(EMBEDDING_DIM), nullable=True)
    return Column(String, nullable=True)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    document_id = Column(
        UUID(as_uuid=False),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(String, nullable=False)
    embedding = _make_embedding_column()
    chunk_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_index", "document_id", "chunk_index"),
    )


class Order(Base):
    __tablename__ = "orders"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    order_number = Column(String, nullable=False, unique=True)
    status = Column(String, nullable=False)
    item = Column(String, nullable=False)
    eta = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_orders_order_number", "order_number"),
    )


# ---------------------------------------------------------------------------
# LLM Usage / Telemetry
# ---------------------------------------------------------------------------
class LLMUsage(Base):
    __tablename__ = "llm_usage"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    request_id = Column(String, nullable=True)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    purpose = Column(String, nullable=True)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    estimated_cost_usd = Column(Float, nullable=False, default=0.0)
    cache_hit = Column(Boolean, nullable=False, default=False)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_llm_usage_created_at", "created_at"),
        Index("ix_llm_usage_provider", "provider"),
    )


# ---------------------------------------------------------------------------
# Workflow Logs
# ---------------------------------------------------------------------------
class WorkflowLog(Base):
    __tablename__ = "workflow_logs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    request_id = Column(String, nullable=True)
    node_name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    log_message = Column(String, nullable=True)
    duration_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_workflow_logs_request_id", "request_id"),
    )


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------
class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    request_id = Column(String, nullable=True)
    directness_score = Column(Float, nullable=True)
    factual_alignment_score = Column(Float, nullable=True)
    evaluator_feedback = Column(String, nullable=True)
    status = Column(String, nullable=False, default="pending")
    provider = Column(String, nullable=True)
    model = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_evaluations_request_id", "request_id"),
    )


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index("ix_conversations_user_id", "user_id"),
        Index("ix_conversations_session_id", "session_id"),
        Index("ix_conversations_updated_at", "updated_at"),
    )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    conversation_id = Column(
        UUID(as_uuid=False),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String, nullable=False)  # user | assistant | system
    content = Column(String, nullable=False)
    provider = Column(String, nullable=True)
    model = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
        Index("ix_messages_created_at", "created_at"),
    )


# ---------------------------------------------------------------------------
# Cache Entries (persistent semantic cache)
# ---------------------------------------------------------------------------
class CacheEntry(Base):
    __tablename__ = "cache_entries"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    cache_key = Column(String, nullable=False, unique=True)
    normalized_prompt = Column(String, nullable=False)
    response = Column(String, nullable=False)
    embedding = _make_embedding_column()
    provider = Column(String, nullable=True)
    model = Column(String, nullable=True)
    task_type = Column(String, nullable=True)
    hits = Column(Integer, nullable=False, default=0)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index("ix_cache_entries_cache_key", "cache_key"),
        Index("ix_cache_entries_expires_at", "expires_at"),
    )
