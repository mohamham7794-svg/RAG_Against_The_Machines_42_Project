"""Entrypoint for ``uv run python -m src <command> [options]``."""

from __future__ import annotations

import fire

from src.cli import RagCLI


def main() -> None:
    """Expose :class:`~src.cli.RagCLI` as a Python Fire CLI."""
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()
