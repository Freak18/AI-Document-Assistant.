"""Alembic environment configuration."""
from __future__ import annotations

import os

from alembic import context

# Read DATABASE_URL from environment (set by startup script via backend/.env)
database_url = os.environ.get("DATABASE_URL")
if database_url:
    context.config.set_main_option("sqlalchemy.url", database_url)
