from __future__ import annotations

import argparse
import logging
import sys

from qa_agent.logging_config import configure_logging

logger = logging.getLogger("qa_agent.run")

_MCP_HINT = (
    "QA exploration runs must be started from Cursor via MCP "
    "(tools: qa_cursor_start / qa_cursor_step on ml-vis-qa-agent). "
    "Cursor MCP sampling provides vision — no external LLM API."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QA exploration agent (MCP-only — see --help)",
        epilog=_MCP_HINT,
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Emit structured JSON logs to stderr",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose, json_logs=args.json_logs)
    logger.error(_MCP_HINT)
    print(_MCP_HINT, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
