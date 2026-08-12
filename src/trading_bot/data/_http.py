"""Shared HTTP utilities for async fetchers."""

from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

import httpx
from loguru import logger

P = ParamSpec("P")
T = TypeVar("T")


def create_http_client(**kwargs: Any) -> httpx.AsyncClient:
    """Create a pre-configured httpx.AsyncClient."""
    defaults: dict[str, Any] = {
        "timeout": httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0),
        "follow_redirects": True,
        "headers": {"User-Agent": "trading-bot/1.0"},
    }
    defaults.update(kwargs)
    return httpx.AsyncClient(**defaults)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Callable[
    [Callable[P, Coroutine[Any, Any, T]]],
    Callable[P, Coroutine[Any, Any, T]],
]:
    """Decorator: retry on 429 / 5xx with exponential backoff + jitter."""

    def decorator(
        func: Callable[P, Coroutine[Any, Any, T]],
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status == 429 or status >= 500:
                        last_exc = exc
                        if attempt == max_retries:
                            break
                        delay = _compute_delay(
                            attempt, base_delay, exc.response.headers
                        )
                        logger.warning(
                            "HTTP {} from {} — retry {}/{} in {:.1f}s",
                            status,
                            exc.request.url,
                            attempt + 1,
                            max_retries,
                            delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
                except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)  # noqa: S311
                    logger.warning(
                        "Network error ({}) — retry {}/{} in {:.1f}s",
                        type(exc).__name__,
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


def _compute_delay(attempt: int, base_delay: float, headers: httpx.Headers) -> float:
    """Compute backoff delay, respecting Retry-After header."""
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return float(retry_after) + random.uniform(0, 1)  # noqa: S311
        except ValueError:
            pass
    delay: float = base_delay * (2**attempt) + random.uniform(0, 1)  # noqa: S311
    return delay
