"""
Market data loader for SuperBot backtesting.

Responsibilities:
    1. Load 15m OHLCV from CSV with strict validation.
    2. Provide UTC-aware, gap-free 15m data.
    3. Resample to 1H and 4H using EXACTLY the same logic as the live bot.
    4. Cache parsed data as Parquet for speed.

Non-responsibilities (do NOT add here):
    - Feature engineering (that's ml/feature_builder.py)
    - Trade simulation (that's exchange_sim.py)
    - Any I/O other than local CSV/Parquet

Compatibility guarantee:
    The resample logic in `resample_ohlcv` MUST match
    weex_v167_8_14_DEMO_EXEC.build_features() line-for-line.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

PRIMARY_TIMEFRAME = "15min"
H1_TIMEFRAME = "1h"
H4_TIMEFRAME = "4h"

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
REQUIRED_COLUMNS = ("timestamp",) + OHLCV_COLUMNS


# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------

@dataclass(frozen=True)
class ValidationReport:
    """Result of validating a 15m OHLCV dataframe."""
    total_rows: int
    first_ts: pd.Timestamp
    last_ts: pd.Timestamp
    expected_rows: int
    missing_count: int
    missing_timestamps: tuple[pd.Timestamp, ...] = field(default_factory=tuple)
    duplicated_count: int = 0
    non_monotonic_count: int = 0
    null_count: int = 0

    @property
    def is_valid(self) -> bool:
        return (
            self.missing_count == 0
            and self.duplicated_count == 0
            and self.non_monotonic_count == 0
            and self.null_count == 0
        )

    def raise_if_invalid(self) -> None:
        if self.is_valid:
            return
        problems = []
        if self.missing_count:
            problems.append(
                f"{self.missing_count} missing 15m candles "
                f"(first 5: {[str(t) for t in self.missing_timestamps[:5]]})"
            )
        if self.duplicated_count:
            problems.append(f"{self.duplicated_count} duplicated timestamps")
        if self.non_monotonic_count:
            problems.append(f"{self.non_monotonic_count} non-monotonic rows")
        if self.null_count:
            problems.append(f"{self.null_count} null values in OHLCV")
        raise ValueError("Market data validation failed: " + "; ".join(problems))


@dataclass(frozen=True)
class MarketData:
    """
    Immutable container for a loaded market dataset.

    Attributes:
        df_15m: 15m OHLCV, UTC-indexed, gap-free, monotonic.
        df_1h:  1H resampled, UTC-indexed.
        df_4h:  4H resampled, UTC-indexed.
        symbol: e.g. "BTCUSDT".
        source: path to original CSV (for provenance).
        report: validation report from load.
    """
    df_15m: pd.DataFrame
    df_1h: pd.DataFrame
    df_4h: pd.DataFrame
    symbol: str
    source: Path
    report: ValidationReport

    def slice(
        self,
        start_utc: str | pd.Timestamp | None = None,
        end_utc: str | pd.Timestamp | None = None,
    ) -> "MarketData":
        """
        Return a new MarketData restricted to [start, end] inclusive.

        The 1H and 4H data are re-derived from the sliced 15m data to
        preserve consistency with the live bot's build_features().
        """
        df = self.df_15m
        if start_utc is not None:
            start = _to_utc_ts(start_utc)
            df = df[df.index >= start]
        if end_utc is not None:
            end = _to_utc_ts(end_utc)
            df = df[df.index <= end]
        if df.empty:
            raise ValueError(
                f"Slice produced empty dataframe: "
                f"start={start_utc}, end={end_utc}, "
                f"available={self.df_15m.index[0]}..{self.df_15m.index[-1]}"
            )
        h1 = resample_ohlcv(df, H1_TIMEFRAME)
        h4 = resample_ohlcv(df, H4_TIMEFRAME)
        new_report = ValidationReport(
            total_rows=len(df),
            first_ts=df.index[0],
            last_ts=df.index[-1],
            expected_rows=len(df),
            missing_count=0,
        )
        return MarketData(
            df_15m=df,
            df_1h=h1,
            df_4h=h4,
            symbol=self.symbol,
            source=self.source,
            report=new_report,
        )

    def summary(self) -> str:
        return (
            f"MarketData(symbol={self.symbol}, rows_15m={len(self.df_15m)}, "
            f"rows_1h={len(self.df_1h)}, rows_4h={len(self.df_4h)}, "
            f"range={self.df_15m.index[0]}..{self.df_15m.index[-1]})"
        )


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _to_utc_ts(value: str | pd.Timestamp) -> pd.Timestamp:
    """Convert str or naive/aware Timestamp to tz-aware UTC Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _expected_15m_index(first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    """Generate the expected gap-free 15m index between first and last inclusive."""
    return pd.date_range(start=first, end=last, freq=PRIMARY_TIMEFRAME, tz="UTC")


# ------------------------------------------------------------
# Resampling (MUST match live bot's build_features)
# ------------------------------------------------------------

def resample_ohlcv(df_15m: pd.DataFrame, timeframe: Literal["1h", "4h"]) -> pd.DataFrame:
    """
    Resample a 15m OHLCV dataframe to 1H or 4H.

    This function MUST produce bit-identical output to the live bot's
    internal build_features() resampling. Any change here breaks parity.

    Args:
        df_15m: 15m OHLCV with DatetimeIndex (UTC).
        timeframe: "1h" or "4h".

    Returns:
        Resampled OHLCV with dropped NaN rows.
    """
    if timeframe not in {H1_TIMEFRAME, H4_TIMEFRAME}:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # Live bot uses: label="left", closed="left"
    resampled = df_15m.resample(timeframe, label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return resampled.dropna()


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame, symbol: str = "UNKNOWN") -> ValidationReport:
    """
    Validate a 15m OHLCV dataframe.

    Checks:
        - DatetimeIndex is UTC-aware
        - Index is monotonic increasing
        - No duplicate timestamps
        - No gaps in the 15m grid
        - No null values in OHLCV columns
        - High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
          (log-only, does not fail)

    Args:
        df: 15m OHLCV with DatetimeIndex.
        symbol: for error messages only.

    Returns:
        ValidationReport.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"[{symbol}] df must have a DatetimeIndex")

    if df.index.tz is None:
        raise ValueError(f"[{symbol}] DatetimeIndex must be tz-aware (UTC)")

    if str(df.index.tz) != "UTC":
        logger.warning("[%s] DatetimeIndex tz is %s, converting to UTC", symbol, df.index.tz)
        df.index = df.index.tz_convert("UTC")

    missing_cols = set(OHLCV_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"[{symbol}] Missing OHLCV columns: {missing_cols}")

    duplicated = int(df.index.duplicated().sum())
    non_monotonic = int((df.index.to_series().diff().dropna() <= pd.Timedelta(0)).sum())
    null_count = int(df[list(OHLCV_COLUMNS)].isna().sum().sum())

    first_ts = df.index[0]
    last_ts = df.index[-1]
    expected_idx = _expected_15m_index(first_ts, last_ts)
    missing_idx = expected_idx.difference(df.index)
    missing_count = len(missing_idx)

    # Sanity: OHLC consistency (log-only)
    bad_high = int((df["high"] < df[["open", "close", "low"]].max(axis=1)).sum())
    bad_low = int((df["low"] > df[["open", "close", "high"]].min(axis=1)).sum())
    if bad_high or bad_low:
        logger.warning(
            "[%s] OHLC consistency issues: high_lt_max=%d, low_gt_min=%d",
            symbol, bad_high, bad_low,
        )

    return ValidationReport(
        total_rows=len(df),
        first_ts=first_ts,
        last_ts=last_ts,
        expected_rows=len(expected_idx),
        missing_count=missing_count,
        missing_timestamps=tuple(missing_idx[:20]),
        duplicated_count=duplicated,
        non_monotonic_count=non_monotonic,
        null_count=null_count,
    )


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def load_market_data(
    csv_path: str | Path,
    symbol: str = "BTCUSDT",
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
    strict: bool = True,
) -> MarketData:
    """
    Load 15m OHLCV from CSV, validate, resample to 1H/4H, and return MarketData.

    Args:
        csv_path: path to CSV file with columns [timestamp, open, high, low, close, volume].
        symbol: trading symbol for logging.
        use_cache: if True, cache parsed dataframe as Parquet in cache_dir.
        cache_dir: directory for Parquet cache. Defaults to "<csv_dir>/.superbot_cache".
        strict: if True, raise on validation failure. If False, log and continue.

    Returns:
        MarketData with df_15m, df_1h, df_4h and validation report.
    """
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if cache_dir is None:
        cache_dir = csv_path.parent / ".superbot_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    parquet_cache = cache_dir / f"{csv_path.stem}.15m.parquet"
    csv_mtime = csv_path.stat().st_mtime
    cached_valid = (
        use_cache
        and parquet_cache.exists()
        and parquet_cache.stat().st_mtime >= csv_mtime
    )

    if cached_valid:
        logger.info("[%s] Loading from Parquet cache: %s", symbol, parquet_cache)
        try:
            df = pd.read_parquet(parquet_cache)
        except Exception as e:
            logger.warning("[%s] Parquet cache corrupt (%r); reloading CSV", symbol, e)
            df = _load_csv(csv_path, symbol)
            _write_parquet(df, parquet_cache)
    else:
        logger.info("[%s] Loading CSV: %s", symbol, csv_path)
        df = _load_csv(csv_path, symbol)
        if use_cache:
            _write_parquet(df, parquet_cache)

    report = validate_ohlcv(df, symbol)
    if strict:
        report.raise_if_invalid()
    elif not report.is_valid:
        logger.warning("[%s] Validation issues (non-strict): %s", symbol, report)

    h1 = resample_ohlcv(df, H1_TIMEFRAME)
    h4 = resample_ohlcv(df, H4_TIMEFRAME)

    logger.info(
        "[%s] Loaded: rows_15m=%d rows_1h=%d rows_4h=%d range=%s..%s",
        symbol, len(df), len(h1), len(h4), df.index[0], df.index[-1],
    )

    return MarketData(
        df_15m=df,
        df_1h=h1,
        df_4h=h4,
        symbol=symbol,
        source=csv_path,
        report=report,
    )


def _load_csv(csv_path: Path, symbol: str) -> pd.DataFrame:
    """Load CSV, enforce schema, set UTC DatetimeIndex, sort, dedupe."""
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"[{symbol}] CSV missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")

    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    df = df[list(OHLCV_COLUMNS)]
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write dataframe to Parquet, atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", compression="snappy")
    tmp.replace(path)


# ------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m superbot.backtest.data_loader <path-to-csv> [symbol]")
        sys.exit(1)
    csv = sys.argv[1]
    sym = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"
    md = load_market_data(csv, symbol=sym)
    print(md.summary())
    print("\nValidation report:")
    print(f"  rows_15m      : {md.report.total_rows}")
    print(f"  expected      : {md.report.expected_rows}")
    print(f"  missing       : {md.report.missing_count}")
    print(f"  duplicated    : {md.report.duplicated_count}")
    print(f"  null          : {md.report.null_count}")
    print(f"  first         : {md.report.first_ts}")
    print(f"  last          : {md.report.last_ts}")
    print(f"  is_valid      : {md.report.is_valid}")