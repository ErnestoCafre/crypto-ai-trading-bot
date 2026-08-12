"""Module entry point: allows `python -m trading_bot`.

The packaged entry point is the `trading-bot` console script (see
[project.scripts] in pyproject.toml), which calls the same function.
"""

from __future__ import annotations

from trading_bot.cli import run

if __name__ == "__main__":
    run()
