from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from config import DATABASE_URL

Base = declarative_base()

if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        echo=bool(os.getenv("SQLALCHEMY_ECHO")),
        pool_size=5,
        max_overflow=10,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    DB_ENABLED = True
else:
    engine = None
    SessionLocal = None
    DB_ENABLED = False


def ensure_database_ready() -> bool:
    """Create the required tables and seed baseline data when the database is empty."""
    if not DB_ENABLED or engine is None or SessionLocal is None:
        return False

    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        required_tables = {
            "documents",
            "document_chunks",
            "orders",
            "llm_usage",
            "workflow_logs",
            "evaluations",
            "conversations",
            "messages",
            "cache_entries",
        }

        if not required_tables.issubset(existing_tables):
            import models  # noqa: F401
            try:
                subprocess.run(
                    [sys.executable, "-m", "alembic", "upgrade", "head"],
                    cwd=str(Path(__file__).resolve().parent),
                    env={**os.environ, "DATABASE_URL": DATABASE_URL},
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] Alembic bootstrap failed: {exc.stderr or exc}")
            try:
                Base.metadata.create_all(bind=engine)
            except Exception:
                pass

        inspector = inspect(engine)
        if not required_tables.issubset(set(inspector.get_table_names())):
            return False

        # Seed a minimal order catalog if the database is otherwise empty.
        from models import Order

        with SessionLocal() as session:
            if session.query(Order).count() == 0:
                seed_rows = [
                    {
                        "id": "0f17a0e1-6dcb-4f4a-8f7e-f8f5b8b65c11",
                        "order_number": "ORD-1001",
                        "status": "Shipped",
                        "item": "Equity delivery basket",
                        "eta": "12 Nov 2026",
                    },
                    {
                        "id": "15f6a9f5-8a6b-469f-8d27-5f2f0dc1f7ff",
                        "order_number": "ORD-1002",
                        "status": "Pending KYC",
                        "item": "Demat account opening",
                        "eta": "on hold",
                    },
                    {
                        "id": "2784a4f0-d12b-4d35-9e85-fc26897f3ec8",
                        "order_number": "ORD-2044",
                        "status": "Settled",
                        "item": "Listed corporate bond",
                        "eta": "settled T+1",
                    },
                    {
                        "id": "35e52f4a-4c45-4429-b14d-743d6251b1f2",
                        "order_number": "ORD-3180",
                        "status": "Rejected",
                        "item": "Margin funding request",
                        "eta": "not applicable",
                    },
                ]
                for row in seed_rows:
                    session.merge(Order(**row))
                session.commit()

        return True
    except Exception as exc:
        print(f"[ERROR] Database bootstrap failed: {exc}")
        return False


def get_db() -> Generator:
    if not DB_ENABLED or SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def db_health() -> dict:
    if not DB_ENABLED or engine is None:
        return {"status": "disabled", "detail": "DATABASE_URL not configured"}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
