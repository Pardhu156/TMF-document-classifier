"""SQLAlchemy connection helpers for PostgreSQL persistence."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
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
