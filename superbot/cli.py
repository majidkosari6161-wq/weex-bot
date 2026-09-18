"""
SuperBot CLI.

Unified entry point for backtest, label, train, wfo, stability, report, parity,
live, and status commands.

Usage:
    python -m superbot <command> [options]
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback

from superbot.cli_utils import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="superbot",
        description="SuperBot — backtest + ML + WFO trading system",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--version", action="version", version="SuperBot 0.1.0")

    sub = parser.add_subparsers(dest="command", required=True)

    # Register subcommands
    from superbot.commands import backtest, status
    backtest.add_parser(sub)
    status.add_parser(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level, log_file=args.log_file)

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 2
    except ValueError as e:
        logger.error("Invalid argument: %s", e)
        return 2
    except Exception as e:
        logger.error("Unhandled error: %r", e)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())