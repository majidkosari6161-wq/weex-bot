"""
Walk-Forward Optimization for SuperBot.

For each rolling window [train | val | test], run Optuna to find parameters
that maximize in-sample Sortino (penalized by max drawdown), then evaluate
on the held-out test period.

Outputs:
    - per_window_results.parquet: every window's best params + train/val/test metrics
    - best_params.json: median across windows (final recommended params)
    - stability.json: parameter stability metrics (CV)
    - wfo_summary.json: aggregate report

Design:
    - ONLY exit parameters are optimized. Entry is frozen.
    - Test period is NEVER shown to the optimizer.
    - Same seed -> same results.
    - Windows do not overlap in test phase (test_days == step_days).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from superbot.backtest.data_loader import MarketData
from superbot.backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not installed; WFO will not run")


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

OPTIMIZABLE_PARAMS = {
    "TP1_R",
    "TP2_R",
    "TRAILING_DISTANCE_R",
    "BREAKEVEN_TRIGGER_R",
    "TP_COMPRESSION_ATR_RATIO",
    "TP_COMPRESSION_MIN_AGE_HOURS",
    "MIN_CONFIDENCE",
}

FROZEN_PARAMS = {
    "ENTRY_THRESHOLD",
    "MIN_EDGE_GAP",
    "ACCOUNT_RISK_PERCENT",
    "MAX_LEVERAGE",
    "MAX_MARGIN_UTILIZATION",
    "MAX_NOTIONAL_PCT",
    "QTY_STEP",
    "MIN_QTY",
    "EXIT_SL_ATR_MULT_BASE",
    "MIN_RISK_DISTANCE_PCT",
    "MIN_RISK_DISTANCE_ATR_MULT",
    "FAILSAFE_TP_R",
    "FAILSAFE_SL_R",
}


# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------

@dataclass
class WindowResult:
    """Result of one WFO window."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    best_params: dict[str, float] = field(default_factory=dict)
    train_metric: float = 0.0
    val_metric: float = 0.0
    test_metric: float = 0.0

    train_n_trades: int = 0
    test_n_trades: int = 0
    test_total_r: float = 0.0
    test_sharpe: float = 0.0
    test_sortino: float = 0.0
    test_max_dd: float = 0.0
    test_win_rate: float = 0.0
    test_pnl_pct: float = 0.0

    n_trials: int = 0
    wall_time_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("train_start", "train_end", "val_start", "val_end",
                  "test_start", "test_end"):
            d[k] = str(d[k])
        return d


@dataclass
class WFOResult:
    """Full WFO result across all windows."""
    run_id: str
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    windows: list[WindowResult] = field(default_factory=list)
    best_params: dict[str, float] = field(default_factory=dict)
    stability: dict[str, dict[str, float]] = field(default_factory=dict)
    mean_test_sortino: float = 0.0
    mean_test_sharpe: float = 0.0
    mean_test_win_rate: float = 0.0
    mean_test_pnl_pct: float = 0.0
    n_windows: int = 0
    wall_time_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"WFOResult(run_id={self.run_id}, "
            f"n_windows={self.n_windows}, "
            f"mean_test_sortino={self.mean_test_sortino:+.3f}, "
            f"mean_test_win_rate={self.mean_test_win_rate:.1%}, "
            f"mean_test_pnl={self.mean_test_pnl_pct:+.2f}%, "
            f"wall_time={self.wall_time_seconds:.1f}s)"
        )


# ------------------------------------------------------------
# WFO Engine
# ------------------------------------------------------------

class WalkForwardOptimizer:
    """Walk-Forward Optimization over rolling windows."""

    def __init__(
        self,
        config: dict,
        market_data: MarketData,
        run_id: str | None = None,
    ) -> None:
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required for WFO")
        self.config = config
        self.market = market_data
        self.run_id = run_id or f"wfo_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"

        wfo = config["wfo"]
        self.train_days = int(wfo["train_days"])
        self.val_days = int(wfo["val_days"])
        self.test_days = int(wfo["test_days"])
        self.step_days = int(wfo["step_days"])
        self.search_space = dict(wfo["search_space"])
        self.objective = str(wfo.get("objective", "sortino_penalized"))
        self.n_trials = int(config["ml"]["optuna_n_trials"])
        self.timeout = int(config["ml"]["optuna_timeout_seconds"])
        self.base_seed = int(config["backtest"]["seed"])

        self._validate_search_space()

    def _validate_search_space(self) -> None:
        illegal = set(self.search_space.keys()) & FROZEN_PARAMS
        if illegal:
            raise ValueError(
                f"Search space includes frozen params: {illegal}. "
                f"These must never be optimized."
            )
        unknown = set(self.search_space.keys()) - OPTIMIZABLE_PARAMS
        if unknown:
            logger.warning("Search space includes unknown params: %s", unknown)

    def _generate_windows(self) -> list[tuple[pd.Timestamp, ...]]:
        """Generate rolling (train_start, train_end, val_start, val_end, test_start, test_end)."""
        df = self.market.df_15m
        first = df.index[0]
        last = df.index[-1]

        train_td = pd.Timedelta(days=self.train_days)
        val_td = pd.Timedelta(days=self.val_days)
        test_td = pd.Timedelta(days=self.test_days)
        step_td = pd.Timedelta(days=self.step_days)

        windows = []
        train_start = first
        while True:
            train_end = train_start + train_td
            val_start = train_end
            val_end = val_start + val_td
            test_start = val_end
            test_end = test_start + test_td

            if test_end > last:
                break

            windows.append((train_start, train_end, val_start, val_end, test_start, test_end))
            train_start = train_start + step_td

        logger.info("Generated %d WFO windows", len(windows))
        return windows

    def run(self, max_windows: int | None = None) -> WFOResult:
        """Run WFO across all rolling windows."""
        windows = self._generate_windows()
        if max_windows is not None:
            windows = windows[:max_windows]

        if not windows:
            raise ValueError(
                f"No WFO windows fit in {len(self.market.df_15m)} bars. "
                f"Try reducing train_days/val_days/test_days."
            )

        t_wall = time.monotonic()
        results: list[WindowResult] = []

        for i, w in enumerate(windows):
            logger.info("===== WFO window %d/%d =====", i + 1, len(windows))
            wr = self._run_window(i, *w)
            results.append(wr)
            logger.info(
                "Window %d: test_sortino=%.3f test_win_rate=%.1f%% trades=%d pnl=%.2f%%",
                i, wr.test_sortino, wr.test_win_rate * 100, wr.test_n_trades, wr.test_pnl_pct,
            )

        best_params = self._aggregate_best_params(results)
        stability = self._compute_stability(results)

        result = WFOResult(
            run_id=self.run_id,
            config_snapshot={},
            windows=results,
            best_params=best_params,
            stability=stability,
            mean_test_sortino=float(np.mean([r.test_sortino for r in results])) if results else 0.0,
            mean_test_sharpe=float(np.mean([r.test_sharpe for r in results])) if results else 0.0,
            mean_test_win_rate=float(np.mean([r.test_win_rate for r in results])) if results else 0.0,
            mean_test_pnl_pct=float(np.mean([r.test_pnl_pct for r in results])) if results else 0.0,
            n_windows=len(results),
            wall_time_seconds=time.monotonic() - t_wall,
        )
        logger.info(result.summary())
        return result

    def _run_window(
        self,
        window_id: int,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        val_start: pd.Timestamp,
        val_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> WindowResult:
        """Run one window's Optuna study + test evaluation."""
        wr = WindowResult(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        )
        t0 = time.monotonic()
        try:
            market_train = self.market.slice(start_utc=train_start, end_utc=train_end)
            market_val = self.market.slice(start_utc=val_start, end_utc=val_end)
            market_test = self.market.slice(start_utc=test_start, end_utc=test_end)

            study = self._optimize(market_train, market_val, window_id)
            wr.best_params = self._params_from_trial(study.best_trial)
            wr.train_metric = float(study.best_trial.user_attrs.get("train_metric", 0.0))
            wr.val_metric = float(study.best_trial.value or 0.0)
            wr.n_trials = len(study.trials)

            test_result = self._run_backtest(market_test, wr.best_params)
            wr.test_metric = self._compute_metric_from_backtest(test_result)
            wr.test_n_trades = test_result.n_trades_opened
            wr.test_total_r = self._total_r(test_result)
            wr.test_sharpe = self._sharpe(test_result)
            wr.test_sortino = self._sortino(test_result)
            wr.test_max_dd = self._max_dd(test_result)
            wr.test_win_rate = self._win_rate(test_result)
            wr.test_pnl_pct = test_result.pnl_pct
            wr.train_n_trades = int(study.best_trial.user_attrs.get("train_n_trades", 0))

        except Exception as e:
            logger.error("Window %d failed: %r", window_id, e)
            wr.error = repr(e)

        wr.wall_time_seconds = time.monotonic() - t0
        return wr

    def _optimize(
        self,
        market_train: MarketData,
        market_val: MarketData,
        window_id: int,
    ) -> "optuna.Study":
        """Run Optuna for one window. Only exit params are sampled."""
        seed = self.base_seed + window_id

        def objective(trial: "optuna.Trial") -> float:
            params = self._sample_params(trial)
            train_result = self._run_backtest(market_train, params)
            train_metric = self._compute_metric_from_backtest(train_result)
            val_result = self._run_backtest(market_val, params)
            val_metric = self._compute_metric_from_backtest(val_result)

            overfit_gap = max(0.0, train_metric - val_metric)
            score = val_metric - 0.3 * overfit_gap

            trial.set_user_attr("train_metric", train_metric)
            trial.set_user_attr("val_metric", val_metric)
            trial.set_user_attr("train_n_trades", train_result.n_trades_opened)
            trial.set_user_attr("val_n_trades", val_result.n_trades_opened)

            return float(score)

        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )
        return study

    def _sample_params(self, trial: "optuna.Trial") -> dict[str, float]:
        params: dict[str, float] = {}
        for name, spec in self.search_space.items():
            low = float(spec["low"])
            high = float(spec["high"])
            if "step" in spec and spec["step"] is not None:
                params[name] = trial.suggest_float(name, low, high, step=float(spec["step"]))
            else:
                params[name] = trial.suggest_float(name, low, high)
        return params

    def _params_from_trial(self, trial: "optuna.Trial") -> dict[str, float]:
        return {k: float(v) for k, v in trial.params.items()}

    def _run_backtest(
        self,
        market: MarketData,
        params: dict[str, float],
    ) -> BacktestResult:
        """Run a backtest with the given exit-parameter overrides."""
        cfg = _deep_copy_config(self.config)
        engine = BacktestEngine(cfg, market)
        bot = engine.bot

        originals: dict[str, Any] = {}
        for name, value in params.items():
            if hasattr(bot, name):
                originals[name] = getattr(bot, name)
                setattr(bot, name, value)
            else:
                logger.warning("Bot has no attribute %s; skipping override", name)

        try:
            n_bars = len(market.df_15m)
            warmup = max(50, min(300, n_bars - 10))
            result = engine.run(
                start_utc=market.df_15m.index[0],
                end_utc=market.df_15m.index[-1],
                warmup_bars=warmup,
            )
        except Exception as e:
            logger.warning("Backtest failed: %r", e)
            result = None
        finally:
            for name, value in originals.items():
                setattr(bot, name, value)

        if result is None:
            result = BacktestResult(
                run_id="empty",
                start_utc=market.df_15m.index[0],
                end_utc=market.df_15m.index[-1],
                initial_balance=10000.0,
                final_balance=10000.0,
            )
        return result

    def _compute_metric_from_backtest(self, result: BacktestResult) -> float:
        if self.objective == "total_r":
            return self._total_r(result)
        if self.objective == "sharpe":
            return self._sharpe(result)
        if self.objective == "sortino":
            return self._sortino(result)
        if self.objective == "sortino_penalized":
            sortino = self._sortino(result)
            dd = abs(self._max_dd(result))
            return sortino - 0.5 * dd
        if self.objective == "profit_factor":
            return self._profit_factor(result)
        if self.objective == "pnl_pct":
            return result.pnl_pct
        raise ValueError(f"Unknown objective: {self.objective}")

    def _trade_returns(self, result: BacktestResult) -> np.ndarray:
        trades: list[float] = []
        current_entry: float | None = None
        current_side: str | None = None
        current_legs: list[tuple[float, float]] = []

        for fill in result.fills:
            reason = fill.reason
            if reason == "ENTRY":
                if current_entry is not None and current_legs:
                    trades.append(self._trade_r(current_entry, current_side, current_legs))
                current_entry = fill.price
                current_side = fill.position_side
                current_legs = []
            elif current_entry is not None:
                current_legs.append((fill.price, fill.quantity))
                if reason in {"SL", "BE", "TRAIL", "TIMEEXIT", "THESIS_EXIT",
                              "EXTERNAL_CLOSE", "SLIPPAGE_EXIT", "TP"}:
                    trades.append(self._trade_r(current_entry, current_side, current_legs))
                    current_entry = None
                    current_side = None
                    current_legs = []

        return np.array(trades, dtype=np.float64) if trades else np.array([], dtype=np.float64)

    @staticmethod
    def _trade_r(entry: float, side: str | None, legs: list[tuple[float, float]]) -> float:
        if not legs:
            return 0.0
        total_qty = sum(q for _, q in legs)
        if total_qty <= 0:
            return 0.0
        avg_exit = sum(p * q for p, q in legs) / total_qty
        if side == "LONG":
            return (avg_exit - entry) / entry * 100.0
        return (entry - avg_exit) / entry * 100.0

    def _total_r(self, result: BacktestResult) -> float:
        r = self._trade_returns(result)
        return float(r.sum()) if r.size else 0.0

    def _sharpe(self, result: BacktestResult) -> float:
        r = self._trade_returns(result)
        if r.size < 2 or r.std(ddof=1) == 0:
            return 0.0
        return float(r.mean() / r.std(ddof=1) * np.sqrt(len(r)))

    def _sortino(self, result: BacktestResult) -> float:
        r = self._trade_returns(result)
        if r.size < 2:
            return 0.0
        downside = r[r < 0]
        if downside.size == 0:
            return float(r.mean() * np.sqrt(len(r))) if r.mean() > 0 else 0.0
        dd_std = downside.std(ddof=1)
        if dd_std == 0:
            return 0.0
        return float(r.mean() / dd_std * np.sqrt(len(r)))

    def _max_dd(self, result: BacktestResult) -> float:
        r = self._trade_returns(result)
        if r.size == 0:
            return 0.0
        cum = np.cumsum(r)
        running_max = np.maximum.accumulate(cum)
        dd = running_max - cum
        return float(dd.max()) if dd.size else 0.0

    def _win_rate(self, result: BacktestResult) -> float:
        r = self._trade_returns(result)
        if r.size == 0:
            return 0.0
        return float((r > 0).mean())

    def _profit_factor(self, result: BacktestResult) -> float:
        r = self._trade_returns(result)
        if r.size == 0:
            return 0.0
        gains = r[r > 0].sum()
        losses = -r[r < 0].sum()
        if losses <= 0:
            return float(gains) if gains > 0 else 0.0
        return float(gains / losses)

    def _aggregate_best_params(self, results: list[WindowResult]) -> dict[str, float]:
        params = list(self.search_space.keys())
        out: dict[str, float] = {}
        for p in params:
            vals = [r.best_params.get(p) for r in results if r.best_params.get(p) is not None]
            if vals:
                out[p] = float(np.median(vals))
        return out

    def _compute_stability(self, results: list[WindowResult]) -> dict[str, dict[str, float]]:
        stability: dict[str, dict[str, float]] = {}
        params = list(self.search_space.keys())
        for p in params:
            vals = np.array([
                r.best_params.get(p, np.nan) for r in results
            ], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                stability[p] = {
                    "median": float("nan"), "std": 0.0, "cv": 0.0,
                    "min": float("nan"), "max": float("nan"),
                    "unstable": 0.0,
                }
                continue
            med = float(np.median(vals))
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            cv = std / abs(med) if med != 0 else 0.0
            stability[p] = {
                "median": med,
                "std": std,
                "cv": cv,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "unstable": 1.0 if cv > 0.3 else 0.0,
            }
        return stability

    def save(self, result: WFOResult, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for w in result.windows:
            d = w.to_dict()
            d["best_params_json"] = json.dumps(d.pop("best_params"), default=str)
            rows.append(d)
        df = pd.DataFrame(rows)
        df.to_parquet(output_dir / "per_window_results.parquet", engine="pyarrow")

        with (output_dir / "best_params.json").open("w", encoding="utf-8") as f:
            json.dump(result.best_params, f, indent=2)

        with (output_dir / "stability.json").open("w", encoding="utf-8") as f:
            json.dump(result.stability, f, indent=2)

        summary = {
            "run_id": result.run_id,
            "n_windows": result.n_windows,
            "mean_test_sortino": result.mean_test_sortino,
            "mean_test_sharpe": result.mean_test_sharpe,
            "mean_test_win_rate": result.mean_test_win_rate,
            "mean_test_pnl_pct": result.mean_test_pnl_pct,
            "wall_time_seconds": result.wall_time_seconds,
            "best_params": result.best_params,
        }
        with (output_dir / "wfo_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Saved WFO results to %s", output_dir)
        return output_dir


def _deep_copy_config(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg, default=str))


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
        print("Usage: python -m superbot.wfo.optimizer <config.yaml> [out_dir] [max_windows]")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from superbot.backtest.data_loader import load_market_data
    md = load_market_data(cfg["paths"]["data_csv"], use_cache=False)

    out_dir = sys.argv[2] if len(sys.argv) > 2 else "superbot_runs/wfo"
    max_windows = int(sys.argv[3]) if len(sys.argv) > 3 else None

    wfo = WalkForwardOptimizer(cfg, md)
    result = wfo.run(max_windows=max_windows)
    print(result.summary())
    print()
    print("Best params (median across windows):")
    for k, v in result.best_params.items():
        print(f"  {k:32s} = {v:.4f}")
    print()
    print("Stability:")
    for k, st in result.stability.items():
        flag = " !!! UNSTABLE" if st["unstable"] else ""
        print(f"  {k:32s} median={st['median']:.4f} cv={st['cv']:.3f}{flag}")
    wfo.save(result, out_dir)
    print()
    print(f"Saved to {out_dir}")