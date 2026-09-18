"""superbot backtest — run a single backtest over a window."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from superbot.backtest.data_loader import load_market_data
from superbot.backtest.engine import BacktestEngine
from superbot.cli_utils import load_config, print_kv

logger = logging.getLogger(__name__)


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("backtest", help="Run a single backtest")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--start", default=None, help="UTC ISO8601 start bar")
    p.add_argument("--end", default=None, help="UTC ISO8601 end bar")
    p.add_argument("--out-dir", default=None, help="Output dir (default: superbot_runs/<run_id>)")
    p.add_argument("--no-report", action="store_true", help="Skip report generation")
    p.add_argument("--no-ledger", action="store_true", help="Skip ledger append")
    p.add_argument("--set", action="append", default=[], help="Config overrides: key.subkey=value")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, overrides=args.set)

    md = load_market_data(cfg["paths"]["data_csv"], use_cache=True)
    logger.info("Loaded %s", md.summary())

    engine = BacktestEngine(cfg, md)
    result = engine.run(start_utc=args.start, end_utc=args.end)

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["runs_dir"]) / result.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics from BacktestResult directly
    print()
    print("=" * 60)
    print(f"Backtest result — {result.run_id}")
    print("=" * 60)
    print_kv({
        "Bars": result.n_signals,
        "Trades opened": result.n_trades_opened,
        "Fills": result.n_fills,
        "PnL (SUSDT)": f"{result.pnl_susdt:+.2f}",
        "PnL %": f"{result.pnl_pct:+.2f}%",
        "Errors": len(result.errors),
        "Wall time": f"{result.wall_time_seconds:.2f}s",
    })
    print("=" * 60)

    # Try to generate report if report module exists
    if not args.no_report:
        try:
            from superbot.reporting.report import generate_backtest_report
            paths = generate_backtest_report(result, out_dir)
            print(f"Report: {paths['html']}")
        except ImportError:
            logger.info("Report module not available; skipping")
        except Exception as e:
            logger.warning("Report generation failed: %r", e)

    # Try to append to ledger if ledger module exists
    if not args.no_ledger:
        try:
            from superbot.reporting.ledger import TradeLedger
            ledger_dir = Path(cfg["paths"]["ledger_dir"])
            ledger = TradeLedger(ledger_dir)
            frag = ledger.append_from_backtest(result)
            if frag:
                print(f"Ledger fragment: {frag}")
        except ImportError:
            logger.info("Ledger module not available; skipping")
        except Exception as e:
            logger.warning("Ledger append failed: %r", e)

    print(f"Output dir: {out_dir}")
    return 0