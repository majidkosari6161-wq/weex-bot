"""superbot status — show current project state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from superbot.cli_utils import load_config


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("status", help="Show project status")
    p.add_argument("--config", default="config.yaml")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print("SuperBot Status")
    print("=" * 60)

    # Data
    data = Path(cfg["paths"]["data_csv"])
    print(f"Data CSV        : {data} ({'OK' if data.exists() else 'MISSING'})")

    # Runs
    runs_dir = Path(cfg["paths"]["runs_dir"])
    n_runs = len(list(runs_dir.iterdir())) if runs_dir.exists() else 0
    print(f"Runs dir        : {runs_dir} ({n_runs} runs)")

    # Models
    models_dir = Path(cfg["paths"]["models_dir"])
    n_models = len(list(models_dir.iterdir())) if models_dir.exists() else 0
    print(f"Models dir      : {models_dir} ({n_models} models)")

    # Ledger
    ledger_dir = Path(cfg["paths"]["ledger_dir"])
    if ledger_dir.exists():
        idx_path = ledger_dir / "index.json"
        if idx_path.exists():
            idx = json.loads(idx_path.read_text())
            n_trades = idx.get("total_trades", 0)
            n_frags = len(idx.get("fragments", []))
            print(f"Ledger          : {ledger_dir} ({n_trades} trades, {n_frags} fragments)")
        else:
            print(f"Ledger          : {ledger_dir} (empty)")
    else:
        print(f"Ledger          : {ledger_dir} (not initialized)")

    # ML status
    ml = cfg.get("ml", {})
    print(f"ML enabled      : {ml.get('enabled', False)}")
    print(f"ML policy       : {ml.get('policy', 'unknown')}")
    model_path = Path(ml.get("model_path", ""))
    print(f"ML model        : {model_path} ({'OK' if model_path.exists() else 'MISSING'})")

    # Bot
    bot_file = Path(cfg["paths"]["bot_file"])
    print(f"Bot file        : {bot_file} ({'OK' if bot_file.exists() else 'MISSING'})")

    return 0