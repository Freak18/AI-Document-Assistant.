from __future__ import annotations

from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("storage_path", sa.String(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=False, server_default="application/pdf"),
        sa.Column("file_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(), nullable=False, server_default="uploaded"),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_documents_status", "documents", ["status"])
    op.create_index("ix_documents_filename", "documents", ["filename"])

    op.create_table(
        "document_chunks",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "document_id", UUID(as_uuid=False),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.String(), nullable=False),

        sa.Column("embedding", sa.Text().with_variant(
            sa.text(f"vector({EMBEDDING_DIM})"), "postgresql"
        ), nullable=True),
        sa.Column("chunk_metadata", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_chunks_index", "document_chunks", ["document_id", "chunk_index"])
    op.execute(
        f"CREATE INDEX ix_chunks_embedding_hnsw ON document_chunks "
        f"USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "orders",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("order_number", sa.String(), nullable=False, unique=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("item", sa.String(), nullable=False),
        sa.Column("eta", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_orders_order_number", "orders", ["order_number"])

    # Seed sample orders
    op.bulk_insert(
        sa.table(
            "orders",
            sa.column("id", sa.String),
            sa.column("order_number", sa.String),
            sa.column("status", sa.String),
            sa.column("item", sa.String),
            sa.column("eta", sa.String),
        ),
        [
            {"id": str(uuid.uuid4()), "order_number": "ORD-1001",
             "status": "Shipped", "item": "Equity delivery basket", "eta": "12 Nov 2026"},
            {"id": str(uuid.uuid4()), "order_number": "ORD-1002",
             "status": "Pending KYC", "item": "Demat account opening", "eta": "on hold"},
            {"id": str(uuid.uuid4()), "order_number": "ORD-2044",
             "status": "Settled", "item": "Listed corporate bond", "eta": "settled T+1"},
            {"id": str(uuid.uuid4()), "order_number": "ORD-3180",
             "status": "Rejected", "item": "Margin funding request", "eta": "not applicable"},
        ],
    )

    op.create_table(
        "llm_usage",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("request_id", sa.String(), nullable=True),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("purpose", sa.String(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_llm_usage_created_at", "llm_usage", ["created_at"])
    op.create_index("ix_llm_usage_provider", "llm_usage", ["provider"])

    op.create_table(
        "workflow_logs",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("request_id", sa.String(), nullable=True),
        sa.Column("node_name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("log_message", sa.String(), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_workflow_logs_request_id", "workflow_logs", ["request_id"])

    op.create_table(
        "evaluations",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("request_id", sa.String(), nullable=True),
        sa.Column("directness_score", sa.Float(), nullable=True),
        sa.Column("factual_alignment_score", sa.Float(), nullable=True),
        sa.Column("evaluator_feedback", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_evaluations_request_id", "evaluations", ["request_id"])

    op.create_table(
        "conversations",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
    op.create_index("ix_conversations_session_id", "conversations", ["session_id"])
    op.create_index("ix_conversations_updated_at", "conversations", ["updated_at"])

    op.create_table(
        "messages",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "conversation_id",
            UUID(as_uuid=False),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])

    op.create_table(
        "cache_entries",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("cache_key", sa.String(), nullable=False, unique=True),
        sa.Column("normalized_prompt", sa.String(), nullable=False),
        sa.Column("response", sa.String(), nullable=False),
        sa.Column(
            "embedding",
            sa.Text().with_variant(sa.text("vector(384)"), "postgresql"),
            nullable=True,
        ),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("task_type", sa.String(), nullable=True),
        sa.Column("hits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_cache_entries_cache_key", "cache_entries", ["cache_key"])
    op.create_index("ix_cache_entries_expires_at", "cache_entries", ["expires_at"])


def downgrade() -> None:
    op.drop_table("cache_entries")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("evaluations")
    op.drop_table("workflow_logs")
    op.drop_table("llm_usage")
    op.drop_table("orders")
    op.drop_index("ix_chunks_embedding_hnsw", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector")

