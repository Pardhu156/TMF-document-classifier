"""SQLAlchemy connection helpers for PostgreSQL persistence."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from src.config import DatabaseConfig
from src.database.models import Base


def create_db_engine(config: DatabaseConfig | None = None):
    """Create a SQLAlchemy engine from environment-backed config."""
    config = config or DatabaseConfig()
    if not config.sqlalchemy_url:
        raise ValueError("PostgreSQL is not configured. Set POSTGRES_* variables or DATABASE_URL.")
    return create_engine(config.sqlalchemy_url, pool_pre_ping=True)


def create_tables(engine) -> None:
    """Create all Stage 4 tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    _ensure_document_metadata_rbac_columns(engine)


def _ensure_document_metadata_rbac_columns(engine) -> None:
    """Add Stage 7.1+ RBAC metadata columns to existing databases."""
    inspector = inspect(engine)
    if "document_metadata" not in inspector.get_table_names():
        return
    existing_columns = {column["name"] for column in inspector.get_columns("document_metadata")}
    statements = []
    if "access_level" not in existing_columns:
        statements.append("ALTER TABLE document_metadata ADD COLUMN access_level VARCHAR(32) DEFAULT 'User' NOT NULL")
    if "owner_id" not in existing_columns:
        statements.append("ALTER TABLE document_metadata ADD COLUMN owner_id VARCHAR(255)")
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
        connection.execute(text("UPDATE document_metadata SET access_level = 'User' WHERE access_level IS NULL"))


def create_session_factory(config: DatabaseConfig | None = None) -> sessionmaker:
    """Return a configured SQLAlchemy session factory."""
    engine = create_db_engine(config)
    create_tables(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def session_scope(session_factory: sessionmaker) -> Iterator[Session]:
    """Provide a transactional scope around DB work."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
