"""Dev entry point: allows `python .` from the repo root.

The packaged entry point is the `trading-bot` console script (see
[project.scripts] in pyproject.toml), which calls the same function.
"""

from __future__ import annotations

from core.cli import run

if __name__ == "__main__":
    run()
