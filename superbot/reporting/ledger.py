"""
Persistent trade ledger for SuperBot.

Records every trade with full attribution:
    - Entry/exit timing and prices
    - Risk distance, ATR, slippage
    - Signal scores, regime, confidence
    - ML decision (p_win, action, top features)
    - Guard pass/block reasons
    - Exit legs (TP1, TP2, SL, BE, TRAIL, ...)
    - Final R, PnL, fees

Storage:
    - Append-only Parquet fragments in superbot_ledger/fragments/
    - index.json lists fragments + schema version
    - Query via pandas; optional DuckDB for SQL

Design:
    - Atomic writes (tmp + rename).
    - Deduplication by trade_id.
    - Schema version for future migration.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from superbot.backtest.engine import BacktestResult, SignalRecord
from superbot.backtest.exchange_sim import FillEvent

logger = logging.getLogger(__name__)

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

SCHEMA_VERSION = 1
LEDGER_DIR = "superbot_ledger"
FRAGMENTS_SUBDIR = "fragments"
INDEX_FILE = "index.json"

LEDGER_COLUMNS = [
    "trade_id", "run_id", "symbol",
    "entry_ts", "exit_ts", "duration_bars", "duration_seconds",
    "side", "qty", "entry_price_signal", "entry_price_fill",
    "entry_slippage", "entry_slippage_pct", "entry_slippage_risk_ratio",
    "notional", "margin_used",
    "entry_atr", "risk_distance", "hard_sl_price", "hard_sl_source",
    "exchange_failsafe_sl", "exchange_failsafe_tp",
    "long_score", "short_score", "edge_gap", "confidence",
    "trigger", "direction", "regime", "is_reduced_size",
    "exit_reason", "exit_price_avg", "n_exit_legs", "exit_legs_json",
    "final_r", "pnl_susdt", "fees_paid", "mfe_r", "mae_r",
    "ml_p_win", "ml_action", "ml_size_multiplier", "ml_top_features_json",
    "guards_checked_json", "guards_blocked_by_json",
    "thesis_status", "thesis_invalidated_at",
    "tp_compressed", "be_armed", "trailing_activated",
]


# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------

@dataclass
class LedgerEntry:
    """One trade's full record. Maps 1:1 to a LEDGER_COLUMNS row."""
    trade_id: str
    run_id: str
    symbol: str = "BTCSUSDT"

    entry_ts: pd.Timestamp | None = None
    exit_ts: pd.Timestamp | None = None
    duration_bars: int = 0
    duration_seconds: int = 0

    side: str = ""
    qty: float = 0.0
    entry_price_signal: float = 0.0
    entry_price_fill: float = 0.0
    entry_slippage: float = 0.0
    entry_slippage_pct: float = 0.0
    entry_slippage_risk_ratio: float = 0.0
    notional: float = 0.0
    margin_used: float = 0.0

    entry_atr: float = 0.0
    risk_distance: float = 0.0
    hard_sl_price: float = 0.0
    hard_sl_source: str = ""
    exchange_failsafe_sl: float = 0.0
    exchange_failsafe_tp: float = 0.0

    long_score: int = 0
    short_score: int = 0
    edge_gap: int = 0
    confidence: float = 0.0
    trigger: str = ""
    direction: str = ""
    regime: str = ""
    is_reduced_size: bool = False

    exit_reason: str = ""
    exit_price_avg: float = 0.0
    n_exit_legs: int = 0
    exit_legs: list[dict[str, Any]] = field(default_factory=list)

    final_r: float = 0.0
    pnl_susdt: float = 0.0
    fees_paid: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0

    ml_p_win: float = float("nan")
    ml_action: str = ""
    ml_size_multiplier: float = 1.0
    ml_top_features: dict[str, float] = field(default_factory=dict)

    guards_checked: list[str] = field(default_factory=list)
    guards_blocked_by: list[str] = field(default_factory=list)

    thesis_status: str = ""
    thesis_invalidated_at: str | None = None
    tp_compressed: bool = False
    be_armed: bool = False
    trailing_activated: bool = False

    def to_row(self) -> dict[str, Any]:
        """Flatten to a single-row dict compatible with LEDGER_COLUMNS."""
        row: dict[str, Any] = {}
        for col in LEDGER_COLUMNS:
            if col.endswith("_json"):
                base = col[:-5]
                value = getattr(self, base, None)
                row[col] = json.dumps(value, default=str) if value else None
            else:
                row[col] = getattr(self, col, None)
        return row


@dataclass
class LedgerIndex:
    schema_version: int = SCHEMA_VERSION
    fragments: list[str] = field(default_factory=list)
    total_trades: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LedgerIndex":
        return cls(
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            fragments=list(d.get("fragments", [])),
            total_trades=int(d.get("total_trades", 0)),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
        )


# ------------------------------------------------------------
# Ledger
# ------------------------------------------------------------

class TradeLedger:
    """Persistent, append-only trade ledger."""

    def __init__(self, root: str | Path = LEDGER_DIR) -> None:
        self.root = Path(root)
        self.fragments_dir = self.root / FRAGMENTS_SUBDIR
        self.fragments_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / INDEX_FILE
        self._index: LedgerIndex | None = None

    @property
    def index(self) -> LedgerIndex:
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def _load_index(self) -> LedgerIndex:
        if not self.index_path.exists():
            return LedgerIndex(
                created_at=pd.Timestamp.now(tz="UTC").isoformat(),
                updated_at=pd.Timestamp.now(tz="UTC").isoformat(),
            )
        try:
            with self.index_path.open("r", encoding="utf-8") as f:
                d = json.load(f)
            return LedgerIndex.from_dict(d)
        except Exception as e:
            logger.error("Failed to load ledger index: %r", e)
            return LedgerIndex()

    def _save_index(self) -> None:
        if self._index is None:
            return
        self._index.updated_at = pd.Timestamp.now(tz="UTC").isoformat()
        tmp = self.index_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._index.to_dict(), f, indent=2, default=str)
        os.replace(tmp, self.index_path)

    def append_entries(self, entries: list[LedgerEntry]) -> Path | None:
        """Append entries as a new fragment. Deduplicates by trade_id."""
        if not entries:
            return None

        existing_ids = self._existing_trade_ids()
        unique_entries = [e for e in entries if e.trade_id not in existing_ids]
        n_dups = len(entries) - len(unique_entries)
        if n_dups:
            logger.info("Ledger: skipped %d duplicate trade(s)", n_dups)
        if not unique_entries:
            return None

        rows = [e.to_row() for e in unique_entries]
        df = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
        df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
        df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")

        frag_name = self._next_fragment_name()
        frag_path = self.fragments_dir / frag_name
        tmp = frag_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, engine="pyarrow", compression="snappy")
        os.replace(tmp, frag_path)

        self.index.fragments.append(frag_name)
        self.index.total_trades += len(unique_entries)
        self._save_index()

        logger.info("Ledger: appended %d trade(s) to %s", len(unique_entries), frag_name)
        return frag_path

    def append_from_backtest(self, result: BacktestResult) -> Path | None:
        """Extract trades from a BacktestResult and append to the ledger."""
        entries = extract_ledger_entries(result)
        return self.append_entries(entries)

    def _existing_trade_ids(self) -> set[str]:
        if not self.index.fragments:
            return set()
        try:
            paths = [self.fragments_dir / f for f in self.index.fragments if (self.fragments_dir / f).exists()]
            if not paths:
                return set()
            df = pd.read_parquet(paths, columns=["trade_id"])
            return set(df["trade_id"].astype(str))
        except Exception as e:
            logger.warning("Failed to read existing trade_ids: %r", e)
            return set()

    def _next_fragment_name(self) -> str:
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S_%f")
        return f"frag_{ts}.parquet"

    def load_all(self) -> pd.DataFrame:
        """Load all fragments into a single DataFrame."""
        if not self.index.fragments:
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        paths = [self.fragments_dir / f for f in self.index.fragments]
        paths = [p for p in paths if p.exists()]
        if not paths:
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        dfs = [pd.read_parquet(p) for p in paths]
        return pd.concat(dfs, ignore_index=True)

    def query(self, where: str | None = None) -> pd.DataFrame:
        """Query the ledger. Uses DuckDB if available, else pandas."""
        if HAS_DUCKDB and where:
            return self._query_duckdb(where)
        df = self.load_all()
        if where:
            try:
                df = df.query(where)
            except Exception as e:
                logger.warning("pandas query failed: %r; returning all rows", e)
        return df

    def _query_duckdb(self, where: str) -> pd.DataFrame:
        path_glob = str(self.fragments_dir / "*.parquet")
        if not list(self.fragments_dir.glob("*.parquet")):
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        con = duckdb.connect()
        try:
            sql = f"SELECT * FROM read_parquet('{path_glob}') WHERE {where}"
            return con.execute(sql).df()
        finally:
            con.close()

    def stats(self) -> dict[str, Any]:
        df = self.load_all()
        if df.empty:
            return {"n_trades": 0}
        return {
            "n_trades": int(len(df)),
            "n_runs": int(df["run_id"].nunique()),
            "first_ts": str(df["entry_ts"].min()),
            "last_ts": str(df["entry_ts"].max()),
            "win_rate": float((df["final_r"] > 0).mean()),
            "mean_r": float(df["final_r"].mean()),
            "total_r": float(df["final_r"].sum()),
            "by_side": df["side"].value_counts().to_dict(),
            "by_exit": df["exit_reason"].value_counts().to_dict(),
        }

    def compact(self) -> Path | None:
        """Merge all fragments into a single fragment."""
        df = self.load_all()
        if df.empty:
            return None
        new_name = self._next_fragment_name()
        new_path = self.fragments_dir / new_name
        tmp = new_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, engine="pyarrow", compression="snappy")
        os.replace(tmp, new_path)

        old = list(self.index.fragments)
        self.index.fragments = [new_name]
        self.index.total_trades = len(df)
        self._save_index()
        for f in old:
            try:
                (self.fragments_dir / f).unlink()
            except FileNotFoundError:
                pass
        logger.info("Ledger compacted: %d trades in %s", len(df), new_name)
        return new_path


# ------------------------------------------------------------
# Extraction from BacktestResult
# ------------------------------------------------------------

def extract_ledger_entries(result: BacktestResult) -> list[LedgerEntry]:
    """Convert a BacktestResult's fills into a list of LedgerEntry."""
    fills = result.fills
    if not fills:
        return []

    groups = _group_fills_into_trades(fills)
    entries: list[LedgerEntry] = []

    # Build ML decisions lookup
    ml_by_ts: dict[str, dict] = {}
    for d in result.ml_decisions:
        ml_by_ts[str(d.get("bar_ts", ""))] = d

    signals_by_ts: dict[pd.Timestamp, SignalRecord] = {
        s.bar_ts: s for s in result.signals
    }

    for i, group in enumerate(groups):
        try:
            entry = _build_entry(result.run_id, i, group, signals_by_ts, ml_by_ts)
            entries.append(entry)
        except Exception as e:
            logger.warning("Failed to build ledger entry %d: %r", i, e)
            continue
    return entries


def _group_fills_into_trades(fills: list[FillEvent]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for fill in fills:
        if fill.reason == "ENTRY":
            if current is not None and current["legs"]:
                groups.append(current)
            current = {"entry_fill": fill, "legs": []}
        elif current is not None:
            current["legs"].append(fill)
            if fill.reason in {"SL", "BE", "TRAIL", "TIMEEXIT", "THESIS_EXIT",
                               "EXTERNAL_CLOSE", "SLIPPAGE_EXIT", "TP"}:
                groups.append(current)
                current = None

    if current is not None and current["legs"]:
        groups.append(current)
    return groups


def _build_entry(
    run_id: str,
    index: int,
    group: dict[str, Any],
    signals_by_ts: dict[pd.Timestamp, SignalRecord],
    ml_by_ts: dict[str, dict],
) -> LedgerEntry:
    entry_fill: FillEvent = group["entry_fill"]
    legs: list[FillEvent] = group["legs"]

    trade_id = _trade_id(run_id, entry_fill)

    e = LedgerEntry(
        trade_id=trade_id,
        run_id=run_id,
        symbol="BTCSUSDT",
        entry_ts=entry_fill.timestamp_utc,
        side=entry_fill.position_side,
        qty=entry_fill.quantity,
        entry_price_fill=entry_fill.price,
    )

    if legs:
        last = legs[-1]
        e.exit_ts = last.timestamp_utc
        e.exit_reason = last.reason

    if e.exit_ts is not None and e.entry_ts is not None:
        delta = pd.Timestamp(e.exit_ts) - pd.Timestamp(e.entry_ts)
        e.duration_seconds = int(delta.total_seconds())
        e.duration_bars = e.duration_seconds // (15 * 60)

    total_qty = sum(f.quantity for f in legs)
    if total_qty > 0:
        e.exit_price_avg = sum(f.price * f.quantity for f in legs) / total_qty
        e.n_exit_legs = len(legs)

    e.exit_legs = [
        {
            "reason": f.reason,
            "price": float(f.price),
            "qty": float(f.quantity),
            "ts": str(f.timestamp_utc),
        }
        for f in legs
    ]

    entry_px = e.entry_price_fill
    if entry_px > 0 and total_qty > 0:
        if e.side == "LONG":
            e.final_r = (e.exit_price_avg - entry_px) / entry_px * 100.0
        else:
            e.final_r = (entry_px - e.exit_price_avg) / entry_px * 100.0

    e.notional = entry_px * e.qty
    e.margin_used = e.notional / 5.0

    e.fees_paid = float(sum(f.fee for f in [entry_fill] + legs))

    # Signal context (nearest previous signal)
    sig = _lookup_signal_for(entry_fill, signals_by_ts)
    if sig is not None:
        e.long_score = sig.long_score
        e.short_score = sig.short_score
        e.edge_gap = sig.edge_gap
        e.confidence = sig.confidence
        e.trigger = sig.trigger or ""
        e.direction = sig.direction
        e.regime = sig.regime
        e.is_reduced_size = sig.is_reduced_size
        # ML fields from SignalRecord
        if sig.ml_action:
            e.ml_action = sig.ml_action
            e.ml_p_win = sig.ml_p_win
            e.ml_size_multiplier = sig.ml_size_multiplier

    return e


def _trade_id(run_id: str, entry_fill: FillEvent) -> str:
    h = hashlib.sha256()
    h.update(run_id.encode())
    h.update(str(entry_fill.timestamp_utc).encode())
    h.update(entry_fill.side.encode())
    h.update(str(entry_fill.order_id).encode())
    return h.hexdigest()[:16]


def _lookup_signal_for(
    fill: FillEvent,
    signals_by_ts: dict[pd.Timestamp, SignalRecord],
) -> SignalRecord | None:
    fill_ts = pd.Timestamp(fill.timestamp_utc)
    best: SignalRecord | None = None
    best_delta: pd.Timedelta | None = None
    for ts, sig in signals_by_ts.items():
        if ts > fill_ts:
            continue
        delta = fill_ts - ts
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = sig
    return best


# ------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m superbot.reporting.ledger <config.yaml> [ledger_dir]")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from superbot.backtest.data_loader import load_market_data
    from superbot.backtest.engine import BacktestEngine

    md = load_market_data(cfg["paths"]["data_csv"], use_cache=False)
    engine = BacktestEngine(cfg, md)
    result = engine.run()

    ledger_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("superbot_ledger")
    ledger = TradeLedger(ledger_dir)
    frag = ledger.append_from_backtest(result)

    print(f"Ledger root: {ledger.root}")
    print(f"Fragment: {frag}")
    print(f"Stats: {json.dumps(ledger.stats(), indent=2, default=str)}")

    df = ledger.load_all()
    print(f"\nLoaded {len(df)} trades")
    if not df.empty:
        print(df[["trade_id", "side", "entry_ts", "exit_reason", "final_r", "ml_p_win", "ml_action"]].head())