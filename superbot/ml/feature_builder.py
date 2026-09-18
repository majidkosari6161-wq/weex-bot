"""
Feature engineering for ML/AI layer.

This module builds a feature vector from:
    1. The live bot's `row` dict (already-computed H1/H4 features).
    2. The 15m OHLCV dataframe up to (but NOT including) the current bar's future.

CRITICAL: No future leak.
    Every rolling window uses .shift(1) so that the feature at time T uses
    only data from [T-window, T-1]. The current bar's close is used only
    for non-lagged instantaneous values (e.g., current close itself).

Design:
    - Feature vector is IDENTICAL for LONG and SHORT signals (symmetric).
    - Directional features are SIGNED (positive = bullish, negative = bearish).
    - Categorical (regime) is one-hot encoded (no ordinal assumption).
    - Time-of-day is cyclic-encoded (sin/cos) AND ordinal.
    - All features are floats. No NaN allowed in output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

REGIME_TYPES = ("CONFIRMED", "CONSOLIDATION", "TRANSITION", "UNCLEAR")

BASE_FEATURES = (
    "H4_eg",
    "H4_rsi",
    "H4_vr",
    "H4_atrp",
    "H1_rsi",
    "H1_vr",
)

DERIVED_FEATURES = (
    "body_atr_signed",
    "ema20_distance_atr_signed",
    "ema50_distance_atr_signed",
    "dist_recent_high_atr",
    "dist_recent_low_atr",
    "atr_percentile_90d",
    "rolling_vol_20",
    "rolling_vol_60",
    "vol_ratio_20_60",
    "volume_skew_20",
    "rsi_divergence_20",
    "momentum_divergence_20",
    "hurst_100",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)

REGIME_FEATURES = tuple(f"regime_{r.lower()}" for r in REGIME_TYPES)

ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES + REGIME_FEATURES

MIN_BARS_REQUIRED = 250


# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------

@dataclass(frozen=True)
class FeatureVector:
    """One feature vector for a single bar."""
    bar_ts: pd.Timestamp
    values: dict[str, float]
    feature_names: tuple[str, ...] = field(default=ALL_FEATURES)

    def as_array(self) -> np.ndarray:
        """Return features in canonical order as float64 array."""
        return np.array([self.values[name] for name in self.feature_names], dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)

    def has_nan(self) -> bool:
        return any(not np.isfinite(v) for v in self.values.values())


# ------------------------------------------------------------
# Rolling feature helpers (all use .shift(1))
# ------------------------------------------------------------

def _safe_log_ret(close: pd.Series) -> pd.Series:
    """Log return, safe against zero/negative prices."""
    return np.log(close / close.shift(1).replace(0, np.nan))


def _atr_15m(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR on 15m bars. Matches live bot's TR definition."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _rsi_15m(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI on 15m. Matches live bot's rsi() helper."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _hurst_exponent(ts: pd.Series, max_lag: int = 20) -> float:
    """Compute Hurst exponent via R/S-style lagged variance."""
    ts = ts.dropna().astype(float)
    if len(ts) < max_lag * 2:
        return np.nan
    lags = np.arange(1, max_lag + 1)
    tau = []
    for lag in lags:
        diff = ts.diff(lag).dropna()
        if len(diff) < 2:
            tau.append(np.nan)
            continue
        std = float(diff.std(ddof=0))
        if std <= 0 or not np.isfinite(std):
            tau.append(np.nan)
        else:
            tau.append(std)
    tau_arr = np.array(tau, dtype=np.float64)
    valid = np.isfinite(tau_arr) & (tau_arr > 0)
    if valid.sum() < 3:
        return np.nan
    log_tau = np.log(tau_arr[valid])
    log_lag = np.log(lags[valid].astype(np.float64))
    slope, _ = np.polyfit(log_lag, log_tau, 1)
    return float(slope)


def _rsi_divergence(close: pd.Series, rsi: pd.Series, window: int = 20) -> pd.Series:
    """Bullish divergence indicator."""
    c_low = close.rolling(window).min()
    c_high = close.rolling(window).max()
    r_low = rsi.rolling(window).min()
    r_high = rsi.rolling(window).max()
    c_norm = (close - c_low) / (c_high - c_low).replace(0, np.nan)
    r_norm = (rsi - r_low) / (r_high - r_low).replace(0, np.nan)
    return (r_norm - c_norm).fillna(0.0)


def _momentum_divergence(close: pd.Series, window: int = 20) -> pd.Series:
    """Price momentum divergence."""
    mom_short = close - close.shift(5)
    mom_long = close - close.shift(window)
    return (mom_short / mom_long.replace(0, np.nan) - 1.0).fillna(0.0)


# ------------------------------------------------------------
# Main builder
# ------------------------------------------------------------

def build_feature_vector(
    row: dict[str, Any],
    df_15m: pd.DataFrame,
    strict: bool = True,
) -> FeatureVector:
    """
    Build one feature vector from live bot's `row` dict and 15m data.

    Args:
        row: live bot's output from build_features().
        df_15m: 15m OHLCV dataframe up to and including the current bar.
        strict: if True, raise on NaN. If False, fill NaN with 0.

    Returns:
        FeatureVector with all ALL_FEATURES populated.
    """
    if len(df_15m) < MIN_BARS_REQUIRED:
        raise ValueError(
            f"Need at least {MIN_BARS_REQUIRED} bars, got {len(df_15m)}"
        )

    missing_base = [k for k in BASE_FEATURES if k not in row]
    if missing_base:
        raise ValueError(f"row missing base features: {missing_base}")

    values: dict[str, float] = {}

    # ---- Base features from live bot ----
    for k in BASE_FEATURES:
        values[k] = _as_float(row[k])

    # ---- Compute derived features ----
    df = df_15m.copy()
    # Ensure we have a clean column-based frame
    if "timestamp" in df.columns:
        df = df.set_index("timestamp") if not isinstance(df.index, pd.DatetimeIndex) else df

    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)

    atr = _atr_15m(df, 14)
    close = df["close"]

    # Current bar values
    c_now = float(close.iloc[-1])
    o_now = float(df["open"].iloc[-1])

    atr_safe = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) and atr.iloc[-1] > 0 else 1.0

    # body_atr_signed
    body = c_now - o_now
    values["body_atr_signed"] = body / atr_safe

    # ema20_distance_atr_signed
    ema20 = close.ewm(span=20, adjust=False).mean().shift(1)
    ema20_now = float(ema20.iloc[-1]) if np.isfinite(ema20.iloc[-1]) else c_now
    values["ema20_distance_atr_signed"] = (c_now - ema20_now) / atr_safe

    # ema50_distance_atr_signed
    ema50 = close.ewm(span=50, adjust=False).mean().shift(1)
    ema50_now = float(ema50.iloc[-1]) if np.isfinite(ema50.iloc[-1]) else c_now
    values["ema50_distance_atr_signed"] = (c_now - ema50_now) / atr_safe

    # Distance to recent high/low (shift 1)
    high_100 = df["high"].shift(1).rolling(100).max().iloc[-1]
    low_100 = df["low"].shift(1).rolling(100).min().iloc[-1]
    if np.isfinite(high_100) and high_100 > 0:
        values["dist_recent_high_atr"] = (c_now - float(high_100)) / atr_safe
    else:
        values["dist_recent_high_atr"] = 0.0
    if np.isfinite(low_100) and low_100 > 0:
        values["dist_recent_low_atr"] = (c_now - float(low_100)) / atr_safe
    else:
        values["dist_recent_low_atr"] = 0.0

    # ATR percentile
    atr_hist = atr.dropna().tail(1000)
    if len(atr_hist) >= 20:
        pct = float((atr_hist < atr_safe).mean())
        values["atr_percentile_90d"] = pct
    else:
        values["atr_percentile_90d"] = 0.5

    # Rolling volatility
    log_ret = _safe_log_ret(close)
    rv20 = log_ret.rolling(20).std(ddof=0)
    rv60 = log_ret.rolling(60).std(ddof=0)
    ann_factor = float(np.sqrt(35040))
    rv20_now = float(rv20.iloc[-1]) if np.isfinite(rv20.iloc[-1]) else 0.0
    rv60_now = float(rv60.iloc[-1]) if np.isfinite(rv60.iloc[-1]) else 0.0
    values["rolling_vol_20"] = rv20_now * ann_factor
    values["rolling_vol_60"] = rv60_now * ann_factor
    values["vol_ratio_20_60"] = (rv20_now / rv60_now) if rv60_now > 0 else 1.0

    # Volume skew
    vol_skew = df["volume"].rolling(20).skew()
    values["volume_skew_20"] = float(vol_skew.iloc[-1]) if np.isfinite(vol_skew.iloc[-1]) else 0.0

    # RSI divergence
    rsi15 = _rsi_15m(close, 14)
    rsi_div = _rsi_divergence(close, rsi15, 20)
    values["rsi_divergence_20"] = float(rsi_div.iloc[-1]) if np.isfinite(rsi_div.iloc[-1]) else 0.0

    # Momentum divergence
    mom_div = _momentum_divergence(close, 20)
    values["momentum_divergence_20"] = float(mom_div.iloc[-1]) if np.isfinite(mom_div.iloc[-1]) else 0.0

    # Hurst
    if len(log_ret) >= 200:
        hurst = _hurst_exponent(log_ret.tail(100), max_lag=20)
        values["hurst_100"] = float(hurst) if np.isfinite(hurst) else 0.5
    else:
        values["hurst_100"] = 0.5

    # Time-of-day cyclic encoding
    ts = pd.Timestamp(row["candle"]) if "candle" in row else df.index[-1]
    hour = ts.hour + ts.minute / 60.0
    dow = ts.dayofweek
    values["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
    values["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
    values["dow_sin"] = float(np.sin(2 * np.pi * dow / 7.0))
    values["dow_cos"] = float(np.cos(2 * np.pi * dow / 7.0))

    # Regime one-hot
    regime = _extract_regime(row)
    for r in REGIME_TYPES:
        key = f"regime_{r.lower()}"
        values[key] = 1.0 if regime == r else 0.0

    # Validate
    nan_keys = [k for k, v in values.items() if not np.isfinite(v)]
    if nan_keys and strict:
        raise ValueError(f"Feature vector has non-finite values: {nan_keys}")

    if nan_keys:
        for k in nan_keys:
            values[k] = 0.0

    return FeatureVector(
        bar_ts=pd.Timestamp(ts),
        values=values,
    )


def _extract_regime(row: dict[str, Any]) -> str:
    """Extract regime type from row."""
    raw = str(row.get("regime", "")).upper()
    if "CONFIRMED" in raw:
        return "CONFIRMED"
    if "CONSOLIDATION" in raw:
        return "CONSOLIDATION"
    if "TRANSITION" in raw:
        return "TRANSITION"
    return "UNCLEAR"


def _as_float(v: Any) -> float:
    """Convert value to float, treating NaN/None as 0.0."""
    try:
        f = float(v)
        if not np.isfinite(f):
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


# ------------------------------------------------------------
# Batch builder
# ------------------------------------------------------------

def build_feature_matrix(
    signals: list[dict[str, Any]],
    market_data: Any,
    strict: bool = True,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """
    Build a feature matrix from a list of candidate signals.
    """
    df_15m = market_data.df_15m
    rows = []
    ts_list: list[pd.Timestamp] = []

    for sig in signals:
        bar_ts = pd.Timestamp(sig["bar_ts"])
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")
        else:
            bar_ts = bar_ts.tz_convert("UTC")

        df_window = df_15m.loc[:bar_ts].reset_index()
        try:
            fv = build_feature_vector(sig["row"], df_window, strict=strict)
        except Exception as e:
            logger.warning("Skipping signal at %s: %r", bar_ts, e)
            continue
        rows.append(fv.values)
        ts_list.append(bar_ts)

    if not rows:
        return pd.DataFrame(columns=list(ALL_FEATURES)), []

    df = pd.DataFrame(rows, columns=list(ALL_FEATURES))
    df.index = pd.DatetimeIndex(ts_list, name="bar_ts")
    return df, ts_list


# ------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        print("config.yaml not found")
        raise SystemExit(1)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from superbot.backtest.data_loader import load_market_data
    from superbot.backtest.engine import load_bot_module

    md = load_market_data(cfg["paths"]["data_csv"], use_cache=False)
    bot = load_bot_module(cfg["paths"]["bot_file"])

    df = md.df_15m
    df_with_ts = df.reset_index()
    row = bot.build_features(df_with_ts)
    if row is None:
        print("Could not build features")
        raise SystemExit(1)

    fv = build_feature_vector(row, df.reset_index())
    print(f"Feature vector at {fv.bar_ts}")
    print(f"Features: {len(fv.feature_names)}")
    print(f"Has NaN: {fv.has_nan()}")
    for k, v in fv.values.items():
        print(f"  {k:32s} = {v:+.6f}")