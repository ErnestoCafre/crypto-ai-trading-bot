"""SQLAlchemy DeclarativeBase and lazily-built engine / session factory.

Nothing connects or reads settings at import time: the engine is created on
first use, so importing models (tests, Alembic autogenerate, the dashboard)
never needs a reachable database or a valid DATABASE_URL.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime
from functools import lru_cache
from typing import Any

from sqlalchemy import DateTime
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from trading_bot.config import get_settings


class Base(DeclarativeBase):
    # Every timestamp is timezone-aware, matching the migrations
    # (DateTime(timezone=True) / TIMESTAMPTZ).
    type_annotation_map = {datetime: DateTime(timezone=True)}


def build_url(url: str | None = None) -> str:
    """Convert postgresql:// to postgresql+asyncpg:// if needed."""
    url = url or get_settings().DATABASE_URL
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    return create_async_engine(
        build_url(),
        echo=get_settings().DATABASE_ECHO,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_session_factory()() as session:
        yield session


def __getattr__(name: str) -> Any:
    # Backwards-compatible lazy module attributes.
    if name == "engine":
        return get_engine()
    if name == "async_session_factory":
        return get_session_factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
