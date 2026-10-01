# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:0.12.19 AS uv

# --- Build: resolve the locked environment into /app/.venv ---
FROM docker.io/library/python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Optional extras, e.g. --build-arg EXTRAS="--extra sentiment" for
# CryptoBERT (pulls torch, several GB).
ARG EXTRAS=""

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $EXTRAS

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev $EXTRAS

# --- Runtime ---
FROM docker.io/library/python:${PYTHON_VERSION}-slim-bookworm

# libgomp1: OpenMP runtime required by LightGBM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 bot \
    && useradd --system --uid 10001 --gid bot --home-dir /app \
       --shell /usr/sbin/nologin bot

WORKDIR /app
COPY --from=builder --chown=bot:bot /app /app
RUN mkdir -p var/state var/logs var/cache var/models \
    && chown -R bot:bot var

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER bot

# Healthy while trading cycles keep completing (see trading_bot/healthcheck.py).
HEALTHCHECK --interval=5m --timeout=30s --start-period=15m --retries=2 \
    CMD ["python", "-m", "trading_bot.healthcheck"]

CMD ["trading-bot"]
