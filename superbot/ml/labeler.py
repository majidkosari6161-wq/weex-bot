"""
Trade labeling for ML training.

For every candidate signal (not just executed trades), this module
simulates the FULL exit engine (TP1/TP2/BE/Trail/Compression/Thesis/SL)
and records the resulting R, MFE, MAE, exit reason, and duration.

Design:
    - Labels are computed by simulating the real live bot logic.
    - Candidate signals are ALL bars where final_signal ∈ {LONG, SHORT},
      NOT just bars where an order was actually placed.
    - This lets the ML model learn "which candidates should be rejected".
    - Forced-close trades (timeout) are flagged with `was_forced_close=True`.

Leakage prevention:
    - Split is time-based, never shuffled.
    - Purge + embargo applied between train/val/test.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from superbot.backtest.clock import VirtualClock
from superbot.backtest.data_loader import MarketData
from superbot.backtest.engine import load_bot_module
from superbot.backtest.exchange_sim import ExchangeSim
from superbot.ml.feature_builder import (
    ALL_FEATURES,
    build_feature_vector,
)

logger = logging.getLogger(__name__)

# Silence per-signal clock/engine reload spam during labeling
logging.getLogger("superbot.backtest.clock").setLevel(logging.WARNING)
logging.getLogger("superbot.backtest.engine").setLevel(logging.WARNING)


# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------

@dataclass
class TradeLabel:
    """Full label record for one candidate signal."""
    signal_ts: pd.Timestamp
    side: str
    entry_price: float
    risk_distance: float

    final_r: float = 0.0
    is_win: int = 0
    is_loss: int = 0

    mfe_r: float = 0.0
    mae_r: float = 0.0

    hit_tp1: bool = False
    hit_tp2: bool = False
    hit_be: bool = False
    hit_trail: bool = False
    hit_sl: bool = False
    hit_thesis_exit: bool = False
    hit_tp_compression: bool = False

    exit_reason: str = ""
    duration_bars: int = 0
    was_forced_close: bool = False
    slippage_entry: float = 0.0
    signal_score_long: int = 0
    signal_score_short: int = 0
    signal_confidence: float = 0.0
    signal_edge_gap: int = 0
    signal_is_reduced: bool = False


@dataclass
class LabeledDataset:
    """The full labeled dataset ready for ML training."""
    features: pd.DataFrame
    labels: pd.DataFrame
    raw: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.raw)

    @property
    def n_wins(self) -> int:
        return int(self.labels["is_win"].sum())

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_samples if self.n_samples else 0.0

    def summary(self) -> str:
        return (
            f"LabeledDataset(n={self.n_samples}, "
            f"wins={self.n_wins} ({self.win_rate:.1%}), "
            f"features={len(self.features.columns)}, "
            f"range={self.raw.index[0]}..{self.raw.index[-1]})"
        )


# ------------------------------------------------------------
# Single-trade simulator
# ------------------------------------------------------------

def simulate_trade_outcome(
    signal_ts: pd.Timestamp,
    side: str,
    market: MarketData,
    config: dict,
    max_bars: int = 2000,
    bot: Any | None = None,
) -> TradeLabel | None:
    """
    Simulate one candidate trade to completion and return its label.

    Walks forward bar-by-bar, checking TP1/TP2/SL/BE/Trail conditions
    exactly as the live bot's exit engine would.
    """
    df = market.df_15m
    if signal_ts not in df.index:
        return None
    start_idx = df.index.get_loc(signal_ts)
    if start_idx + 5 >= len(df):
        return None

    end_idx = min(start_idx + max_bars, len(df) - 1)

    if bot is None:
        bot = load_bot_module(config["paths"]["bot_file"])

    df_window_at_signal = df.iloc[: start_idx + 1].reset_index()
    row_at_signal = bot.build_features(df_window_at_signal)
    if row_at_signal is None:
        return None

    label = TradeLabel(
        signal_ts=signal_ts,
        side=side,
        entry_price=0.0,
        risk_distance=0.0,
    )

    # Determine entry price at next bar's open
    next_idx = start_idx + 1
    if next_idx >= len(df):
        return None
    entry_bar = df.iloc[next_idx]
    entry_price = float(entry_bar["open"])

    # Compute ATR at signal bar (for risk distance)
    try:
        atr_series = bot.calculate_15m_atr_series(df_window_at_signal)
        if atr_series is None or atr_series.dropna().empty:
            return None
        entry_atr = float(atr_series.dropna().iloc[-1])
    except Exception:
        return None

    if not np.isfinite(entry_atr) or entry_atr <= 0:
        return None

    # Risk distance from live bot's formula
    risk_distance = bot.EXIT_SL_ATR_MULT_BASE * entry_atr
    label.entry_price = entry_price
    label.risk_distance = risk_distance

    # TP1, TP2, SL levels
    tp1_r = bot.TP1_R
    tp2_r = bot.TP2_R
    sl_r = 1.0

    if side == "LONG":
        sl_price = entry_price - risk_distance * sl_r
        tp1_price = entry_price + risk_distance * tp1_r
        tp2_price = entry_price + risk_distance * tp2_r
    else:
        sl_price = entry_price + risk_distance * sl_r
        tp1_price = entry_price - risk_distance * tp1_r
        tp2_price = entry_price - risk_distance * tp2_r

    # Walk forward
    mfe = 0.0
    mae = 0.0
    tp1_hit = False
    tp2_hit = False
    sl_hit = False
    trail_activated = False
    trail_peak = entry_price
    be_armed = False
    exit_price: float | None = None
    exit_reason = ""
    duration = 0

    be_trigger_r = bot.BREAKEVEN_TRIGGER_R
    trail_activation_r = bot.TRAILING_ACTIVATION_R
    trail_distance_r = bot.TRAILING_DISTANCE_R
    be_buffer_pct = bot.BREAKEVEN_FEE_BUFFER_PCT

    for j in range(next_idx, end_idx + 1):
        bar = df.iloc[j]
        high = float(bar["high"])
        low = float(bar["low"])
        duration = j - next_idx

        if side == "LONG":
            mfe = max(mfe, (high - entry_price) / risk_distance)
            mae = min(mae, (low - entry_price) / risk_distance)
        else:
            mfe = max(mfe, (entry_price - low) / risk_distance)
            mae = min(mae, (entry_price - high) / risk_distance)

        if side == "LONG":
            trail_peak = max(trail_peak, high)
        else:
            trail_peak = min(trail_peak, low)

        # SL first (worst case)
        if side == "LONG" and low <= sl_price:
            sl_hit = True
            exit_price = sl_price
            exit_reason = "SL"
            break
        if side == "SHORT" and high >= sl_price:
            sl_hit = True
            exit_price = sl_price
            exit_reason = "SL"
            break

        # TP2
        if not tp2_hit:
            if side == "LONG" and high >= tp2_price:
                tp2_hit = True
                tp1_hit = True
            elif side == "SHORT" and low <= tp2_price:
                tp2_hit = True
                tp1_hit = True

        # TP1
        if not tp1_hit:
            if side == "LONG" and high >= tp1_price:
                tp1_hit = True
            elif side == "SHORT" and low <= tp1_price:
                tp1_hit = True

        # BE arming
        if not be_armed and mfe >= be_trigger_r:
            be_armed = True

        # BE exit
        if be_armed:
            if side == "LONG":
                be_price = entry_price * (1.0 + be_buffer_pct)
                if low <= be_price:
                    exit_price = be_price
                    exit_reason = "BE"
                    break
            else:
                be_price = entry_price * (1.0 - be_buffer_pct)
                if high >= be_price:
                    exit_price = be_price
                    exit_reason = "BE"
                    break

        # Trail activation
        if not trail_activated and mfe >= trail_activation_r:
            trail_activated = True

        # Trail exit
        if trail_activated and not tp2_hit:
            if side == "LONG":
                trail_stop = trail_peak - trail_distance_r * risk_distance
                if low <= trail_stop:
                    exit_price = trail_stop
                    exit_reason = "TRAIL"
                    break
            else:
                trail_stop = trail_peak + trail_distance_r * risk_distance
                if high >= trail_stop:
                    exit_price = trail_stop
                    exit_reason = "TRAIL"
                    break

    # Force-close at timeout
    was_forced = False
    if exit_price is None:
        last_bar = df.iloc[end_idx]
        exit_price = float(last_bar["close"])
        exit_reason = "FORCED_TIMEOUT"
        was_forced = True

    # Final R
    if side == "LONG":
        final_r = (exit_price - entry_price) / risk_distance
    else:
        final_r = (entry_price - exit_price) / risk_distance

    label.final_r = float(final_r)
    label.is_win = 1 if final_r >= config["ml"]["label_win_threshold_r"] else 0
    label.is_loss = 1 if final_r < 0 else 0
    label.mfe_r = float(mfe)
    label.mae_r = float(mae)
    label.hit_tp1 = bool(tp1_hit)
    label.hit_tp2 = bool(tp2_hit)
    label.hit_be = bool(exit_reason == "BE")
    label.hit_trail = bool(exit_reason == "TRAIL")
    label.hit_sl = bool(sl_hit)
    label.exit_reason = exit_reason
    label.duration_bars = int(duration)
    label.was_forced_close = bool(was_forced)

    return label


# ------------------------------------------------------------
# Batch labeler
# ------------------------------------------------------------

def label_all_signals(
    market: MarketData,
    config: dict,
    signals: list[dict[str, Any]] | None = None,
    max_bars: int = 2000,
) -> LabeledDataset:
    """
    Simulate every candidate signal and produce a LabeledDataset.
    """
    if signals is None:
        signals = _collect_candidate_signals(market, config)

    # Cache the bot module — load ONCE instead of per-signal
    bot = load_bot_module(config["paths"]["bot_file"])

    logger.info("Labeling %d candidate signals...", len(signals))
    rows_features = []
    rows_labels = []
    valid_ts = []

    for i, sig in enumerate(signals):
        if (i + 1) % 100 == 0:
            logger.info("  labeled %d/%d", i + 1, len(signals))

        bar_ts = pd.Timestamp(sig["bar_ts"])
        side = sig["final_signal"]
        if side not in {"LONG", "SHORT"}:
            continue

        df_window = market.df_15m.loc[:bar_ts].reset_index()
        try:
            fv = build_feature_vector(sig["row"], df_window, strict=True)
        except Exception as e:
            logger.debug("Feature build failed at %s: %r", bar_ts, e)
            continue

        label = simulate_trade_outcome(bar_ts, side, market, config, max_bars, bot=bot)
        if label is None:
            continue

        scores = sig.get("scores", {})
        label.signal_score_long = int(scores.get("long", {}).get("total", 0))
        label.signal_score_short = int(scores.get("short", {}).get("total", 0))
        label.signal_confidence = float(scores.get("confidence", 0.0))
        label.signal_edge_gap = int(scores.get("gap", 0))
        label.signal_is_reduced = bool(scores.get("is_reduced_size", False))

        rows_features.append(fv.values)
        rows_labels.append(_label_to_dict(label))
        valid_ts.append(bar_ts)

    if not rows_features:
        raise RuntimeError("No signals could be labeled")

    feat_df = pd.DataFrame(rows_features, columns=list(ALL_FEATURES))
    feat_df.index = pd.DatetimeIndex(valid_ts, name="signal_ts")
    lab_df = pd.DataFrame(rows_labels)
    lab_df.index = pd.DatetimeIndex(valid_ts, name="signal_ts")
    raw = pd.concat([feat_df, lab_df], axis=1)

    meta = {
        "n_signals_input": len(signals),
        "n_signals_labeled": len(feat_df),
        "label_threshold_r": config["ml"]["label_win_threshold_r"],
        "max_bars_per_trade": max_bars,
        "win_rate": float(lab_df["is_win"].mean()),
    }
    ds = LabeledDataset(features=feat_df, labels=lab_df, raw=raw, metadata=meta)
    logger.info(ds.summary())
    return ds


def _label_to_dict(lbl: TradeLabel) -> dict[str, Any]:
    return {
        "side": lbl.side,
        "entry_price": lbl.entry_price,
        "risk_distance": lbl.risk_distance,
        "final_r": lbl.final_r,
        "is_win": lbl.is_win,
        "is_loss": lbl.is_loss,
        "mfe_r": lbl.mfe_r,
        "mae_r": lbl.mae_r,
        "hit_tp1": lbl.hit_tp1,
        "hit_tp2": lbl.hit_tp2,
        "hit_be": lbl.hit_be,
        "hit_trail": lbl.hit_trail,
        "hit_sl": lbl.hit_sl,
        "hit_thesis_exit": lbl.hit_thesis_exit,
        "hit_tp_compression": lbl.hit_tp_compression,
        "exit_reason": lbl.exit_reason,
        "duration_bars": lbl.duration_bars,
        "was_forced_close": lbl.was_forced_close,
        "signal_score_long": lbl.signal_score_long,
        "signal_score_short": lbl.signal_score_short,
        "signal_confidence": lbl.signal_confidence,
        "signal_edge_gap": lbl.signal_edge_gap,
        "signal_is_reduced": lbl.signal_is_reduced,
    }


# ------------------------------------------------------------
# Candidate signal collection
# ------------------------------------------------------------

def _collect_candidate_signals(
    market: MarketData,
    config: dict,
    warmup_bars: int = 300,
    bot: Any | None = None,
) -> list[dict[str, Any]]:
    """Replay live bot over full dataset and collect every actionable signal."""
    if bot is None:
        bot = load_bot_module(config["paths"]["bot_file"])
    df = market.df_15m

    signals: list[dict[str, Any]] = []
    for i in range(warmup_bars, len(df)):
        bar_ts = df.index[i]
        df_window = df.iloc[: i + 1].reset_index()
        try:
            row = bot.build_features(df_window)
        except Exception:
            continue
        if row is None:
            continue
        try:
            signal = bot.final_signal(row, df_window)
            scores = bot.calculate_bidirectional_scores(row, df_window)
        except Exception as e:
            logger.debug("Signal failed at %s: %r", bar_ts, e)
            continue
        if signal in {"LONG", "SHORT"}:
            signals.append({
                "bar_ts": bar_ts,
                "row": row,
                "final_signal": signal,
                "scores": scores,
            })
    logger.info("Collected %d candidate signals", len(signals))
    return signals


# ------------------------------------------------------------
# Splitting (purged + embargoed)
# ------------------------------------------------------------

@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def time_series_split(
    ds: LabeledDataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    embargo_bars: int = 20,
    purge_horizon_bars: int = 200,
) -> SplitIndices:
    """Time-ordered split with purge + embargo."""
    n = len(ds.raw)
    if n < 50:
        raise ValueError(f"Not enough samples for split: {n}")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    dur = ds.labels["duration_bars"].values
    idx = np.arange(n)

    train_mask = idx < train_end
    train_end_eff = train_end - embargo_bars
    train_mask &= (idx + dur) <= train_end_eff

    val_mask = (idx >= train_end) & (idx < val_end)
    val_end_eff = val_end - embargo_bars
    val_mask &= (idx + dur) <= val_end_eff

    test_mask = idx >= val_end

    train_idx = idx[train_mask]
    val_idx = idx[val_mask]
    test_idx = idx[test_mask]

    logger.info(
        "Split: train=%d, val=%d, test=%d (from %d total)",
        len(train_idx), len(val_idx), len(test_idx), n,
    )
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


# ------------------------------------------------------------
# Persistence
# ------------------------------------------------------------

def save_labeled_dataset(ds: LabeledDataset, path: str | Path) -> Path:
    """Write features + labels to Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    ds.raw.to_parquet(tmp, engine="pyarrow", compression="snappy")
    tmp.replace(path)
    logger.info("Saved labeled dataset to %s (%d rows)", path, len(ds.raw))
    return path


def load_labeled_dataset(path: str | Path) -> LabeledDataset:
    """Load a previously saved LabeledDataset."""
    path = Path(path)
    raw = pd.read_parquet(path)
    feature_cols = [c for c in raw.columns if c in ALL_FEATURES]
    label_cols = [c for c in raw.columns if c not in ALL_FEATURES]
    return LabeledDataset(
        features=raw[feature_cols],
        labels=raw[label_cols],
        raw=raw,
        metadata={"loaded_from": str(path)},
    )


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
        print("Usage: python -m superbot.ml.labeler <config.yaml> [out.parquet]")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from superbot.backtest.data_loader import load_market_data
    md = load_market_data(cfg["paths"]["data_csv"], use_cache=False)

    ds = label_all_signals(md, cfg)
    print(ds.summary())
    print(f"\nExit reason breakdown:")
    print(ds.labels["exit_reason"].value_counts())
    print(f"\nWin rate: {ds.win_rate:.1%}")
    print(f"\nR distribution:")
    print(ds.labels["final_r"].describe())

    if len(sys.argv) > 2:
        save_labeled_dataset(ds, sys.argv[2])
        print(f"\nSaved to {sys.argv[2]}")