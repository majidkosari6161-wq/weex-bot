"""
Event-driven backtest engine.
"""
from __future__ import annotations

import importlib.util
import logging
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from superbot.backtest.clock import VirtualClock
from superbot.backtest.data_loader import MarketData, load_market_data
from superbot.backtest.exchange_sim import ExchangeSim, FillEvent

try:
    from superbot.ml.filter import MLFilter, ACTION_REJECT, ACTION_REDUCE
    HAS_ML_FILTER = True
except ImportError:
    HAS_ML_FILTER = False
    MLFilter = None  # type: ignore
    ACTION_REJECT = "REJECT"
    ACTION_REDUCE = "REDUCE"

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    bar_ts: pd.Timestamp
    final_signal: str
    long_score: int
    short_score: int
    edge_gap: int
    confidence: float
    price_move_pct: float
    trigger: str | None
    direction: str
    regime: str
    is_reduced_size: bool
    position_opened: bool = False
    blocked_reason: str | None = None
    ml_action: str = ""
    ml_p_win: float = float("nan")
    ml_size_multiplier: float = 1.0
    ml_blocked: bool = False


@dataclass
class BacktestResult:
    run_id: str
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    initial_balance: float
    final_balance: float
    signals: list[SignalRecord] = field(default_factory=list)
    fills: list[FillEvent] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    ml_decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_signals(self) -> int:
        return len(self.signals)

    @property
    def n_trades_opened(self) -> int:
        return sum(1 for s in self.signals if s.position_opened)

    @property
    def n_ml_blocked(self) -> int:
        return sum(1 for s in self.signals if s.ml_blocked)

    @property
    def n_fills(self) -> int:
        return len(self.fills)

    @property
    def pnl_susdt(self) -> float:
        return self.final_balance - self.initial_balance

    @property
    def pnl_pct(self) -> float:
        return 100.0 * (self.final_balance / self.initial_balance - 1.0)

    def summary(self) -> str:
        return (
            f"BacktestResult(run_id={self.run_id}, "
            f"bars={self.n_signals}, trades={self.n_trades_opened}, "
            f"ml_blocked={self.n_ml_blocked}, "
            f"fills={self.n_fills}, "
            f"pnl={self.pnl_susdt:+.2f} SUSDT ({self.pnl_pct:+.2f}%), "
            f"wall_time={self.wall_time_seconds:.2f}s)"
        )


def load_bot_module(bot_path: str | Path) -> Any:
    bot_path = Path(bot_path).resolve()
    if not bot_path.exists():
        raise FileNotFoundError(f"Bot file not found: {bot_path}")
    spec = importlib.util.spec_from_file_location("weex_bot", bot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {bot_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info("Loaded bot module from %s", bot_path)
    return module


@contextmanager
def _no_sleep() -> Iterator[None]:
    orig = time.sleep
    time.sleep = lambda _x: None  # type: ignore[assignment]
    try:
        yield
    finally:
        time.sleep = orig  # type: ignore[assignment]


@contextmanager
def _patch_time_time(clock: VirtualClock) -> Iterator[None]:
    orig = time.time
    time.time = lambda: clock.now().timestamp()  # type: ignore[assignment]
    try:
        yield
    finally:
        time.time = orig  # type: ignore[assignment]


@contextmanager
def _isolated_state_dir(bot_module: Any, run_dir: Path) -> Iterator[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    state_attrs = [
        "STATE_FILE", "SHARED_STATE_FILE", "EXIT_STATE_FILE",
        "EXIT_LOG_FILE", "ENTRY_ATR_FILE", "CIRCUIT_BREAKER_FILE",
        "LAST_PRICE_TELEMETRY_FILE", "DECAY_STATE_FILE",
        "BREAKEVEN_STATE_FILE", "DATA_FILE",
    ]
    originals: dict[str, Any] = {}
    for attr in state_attrs:
        if hasattr(bot_module, attr):
            originals[attr] = getattr(bot_module, attr)
            new_name = str(run_dir / Path(str(originals[attr])).name)
            setattr(bot_module, attr, new_name)
    if hasattr(bot_module, "_SHARED_STATE_LOCK_FILE"):
        originals["_SHARED_STATE_LOCK_FILE"] = bot_module._SHARED_STATE_LOCK_FILE
        bot_module._SHARED_STATE_LOCK_FILE = str(
            run_dir / Path(str(bot_module._SHARED_STATE_LOCK_FILE)).name
        )
    try:
        yield run_dir
    finally:
        for attr, value in originals.items():
            setattr(bot_module, attr, value)


@contextmanager
def _disable_shared_state(bot_module: Any) -> Iterator[None]:
    orig_load = bot_module.load_shared_state
    orig_save = bot_module.save_shared_state
    orig_is_primary = bot_module.is_primary_alive
    orig_can_send = bot_module.can_send_order

    _mem_state: dict[str, Any] = {
        "system_id": "BACKTEST",
        "status": "RUNNING",
        "last_candle": None,
        "last_heartbeat": None,
        "mode": "PRIMARY",
    }

    def _load():
        return dict(_mem_state)

    def _save(last_candle=None, status=None, mode=None):
        _mem_state["last_heartbeat"] = pd.Timestamp.now(tz="UTC").isoformat()
        if last_candle is not None:
            _mem_state["last_candle"] = str(last_candle)
        if status is not None:
            _mem_state["status"] = status
        if mode is not None:
            _mem_state["mode"] = mode

    def _is_primary():
        return True

    def _can_send(current_candle):
        return True

    bot_module.load_shared_state = _load
    bot_module.save_shared_state = _save
    bot_module.is_primary_alive = _is_primary
    bot_module.can_send_order = _can_send
    try:
        yield
    finally:
        bot_module.load_shared_state = orig_load
        bot_module.save_shared_state = orig_save
        bot_module.is_primary_alive = orig_is_primary
        bot_module.can_send_order = orig_can_send


def _force_utf8_io() -> None:
    import sys
    import io
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
            else:
                buf = getattr(stream, "buffer", None)
                if buf is not None:
                    new_stream = io.TextIOWrapper(
                        buf, encoding="utf-8", errors="replace", line_buffering=True
                    )
                    setattr(sys, stream_name, new_stream)
        except Exception:
            pass


class BacktestEngine:
    def __init__(
        self,
        config: dict,
        market_data: MarketData,
        bot_module: Any | None = None,
        run_id: str | None = None,
        ml_filter: Any | None = None,
    ) -> None:
        self.config = config
        self.market = market_data
        if bot_module is None:
            bot_path = Path(config["paths"]["bot_file"])
            bot_module = load_bot_module(bot_path)
        self.bot = bot_module
        self.run_id = run_id or f"run_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"

        # Apply bot_params overrides from config
        bot_params = config.get("bot_params", {})
        if bot_params:
            for name, value in bot_params.items():
                if hasattr(self.bot, name):
                    old = getattr(self.bot, name)
                    if old != value:
                        setattr(self.bot, name, value)
                        logger.info("bot_params override: %s = %s (was %s)", name, value, old)
                else:
                    logger.warning("bot_params: bot has no attribute %s", name)

        self.ml_filter = ml_filter
        if self.ml_filter is None and HAS_ML_FILTER:
            ml_cfg = config.get("ml", {})
            if ml_cfg.get("enabled", False):
                try:
                    self.ml_filter = MLFilter.from_config(config, strict=False)
                    logger.info("ML filter loaded: %s", self.ml_filter.status_line())
                except Exception as e:
                    logger.warning("ML filter load failed: %r — proceeding without ML", e)
                    self.ml_filter = None

    def run(
        self,
        start_utc: str | pd.Timestamp | None = None,
        end_utc: str | pd.Timestamp | None = None,
        warmup_bars: int = 300,
    ) -> BacktestResult:
        _force_utf8_io()

        if start_utc is not None or end_utc is not None:
            market = self.market.slice(start_utc=start_utc, end_utc=end_utc)
        else:
            market = self.market

        df = market.df_15m
        if len(df) <= warmup_bars:
            raise ValueError(f"Not enough bars: got {len(df)}, need > {warmup_bars}")

        run_dir = Path(tempfile.mkdtemp(prefix=f"superbot_{self.run_id}_"))
        logger.info("Backtest run dir: %s", run_dir)

        clock = VirtualClock.from_iso(str(df.index[0]))
        sim = ExchangeSim(
            self.config, clock, market,
            initial_balance=self.config["backtest"]["initial_balance_susdt"],
        )

        result = BacktestResult(
            run_id=self.run_id,
            start_utc=df.index[0],
            end_utc=df.index[-1],
            initial_balance=sim.initial_balance,
            final_balance=sim.initial_balance,
        )

        t_wall_start = time.monotonic()

        try:
            with _isolated_state_dir(self.bot, run_dir), \
                 _disable_shared_state(self.bot), \
                 _no_sleep(), \
                 _patch_time_time(clock), \
                 clock.installed():
                sim.install(self.bot)

                for i, bar_ts in enumerate(df.index):
                    if i < warmup_bars:
                        continue
                    self._process_bar(bar_ts, market, sim, result, warmup_bars)

        finally:
            try:
                sim.uninstall(self.bot)
            except Exception as e:
                logger.warning("ExchangeSim uninstall failed: %r", e)
            result.wall_time_seconds = time.monotonic() - t_wall_start
            result.final_balance = sim.balance
            result.orders = [o.to_dict() for o in sim.orders]

        logger.info(result.summary())
        return result

    def _process_bar(
        self,
        bar_ts: pd.Timestamp,
        market: MarketData,
        sim: ExchangeSim,
        result: BacktestResult,
        warmup_bars: int,
    ) -> None:
        # 1) Set sim to this bar
        clock = sim.clock
        clock.set(bar_ts)
        bar = market.df_15m.loc[bar_ts]
        mark = float(bar["close"])
        last = mark
        sim.set_bar(bar_ts, mark, last)
        sim.refresh_unrealized_pnl()

        # 2) TP/SL conditional triggers
        n_fills_before_tpsl = len(sim.fills)
        try:
            triggered = sim.check_tp_sl_triggers()
            for ev in triggered:
                logger.debug("TP/SL triggered: %s %s @ %.1f",
                             ev.reason, ev.position_side, ev.price)
        except Exception as e:
            msg = f"TP/SL check failed at {bar_ts}: {e!r}"
            logger.warning(msg)
            result.errors.append(msg)
        for ev in sim.fills[n_fills_before_tpsl:]:
            result.fills.append(ev)

        # 3) Build features
        df_window = market.df_15m.loc[:bar_ts].reset_index()
        try:
            row = self.bot.build_features(df_window)
        except Exception as e:
            msg = f"build_features failed at {bar_ts}: {e!r}"
            logger.warning(msg)
            result.errors.append(msg)
            return
        if row is None:
            return

        # 4) Exit engine
        n_fills_before_exit = len(sim.fills)
        try:
            self.bot.manage_position_exits(df_window)
        except Exception as e:
            msg = f"manage_position_exits failed at {bar_ts}: {e!r}"
            logger.warning(msg)
            result.errors.append(msg)
        for ev in sim.fills[n_fills_before_exit:]:
            result.fills.append(ev)

        # 5) Entry decision
        try:
            signal = self.bot.final_signal(row, df_window)
        except Exception as e:
            msg = f"final_signal failed at {bar_ts}: {e!r}"
            logger.warning(msg)
            result.errors.append(msg)
            signal = "HOLD"

        try:
            scores = self.bot.calculate_bidirectional_scores(row, df_window)
        except Exception:
            scores = {}

        sig_rec = SignalRecord(
            bar_ts=bar_ts,
            final_signal=signal,
            long_score=int(scores.get("long", {}).get("total", 0)),
            short_score=int(scores.get("short", {}).get("total", 0)),
            edge_gap=int(scores.get("gap", 0)),
            confidence=float(scores.get("confidence", 0.0)),
            price_move_pct=float(scores.get("price_move_pct", 0.0)),
            trigger=scores.get("trigger"),
            direction=str(scores.get("direction", "NEUTRAL")),
            regime=str(scores.get("regime", "UNKNOWN")),
            is_reduced_size=bool(scores.get("is_reduced_size", False)),
        )

        # 6) ML filter
        if signal in {"LONG", "SHORT"} and self.ml_filter is not None:
            try:
                decision = self.ml_filter.evaluate(signal, row, df_window)
                sig_rec.ml_action = decision.action
                sig_rec.ml_p_win = float(decision.p_win) if np.isfinite(decision.p_win) else float("nan")
                sig_rec.ml_size_multiplier = float(decision.size_multiplier)
                sig_rec.ml_blocked = decision.blocked

                result.ml_decisions.append({
                    "bar_ts": str(bar_ts),
                    "signal": signal,
                    **decision.to_dict(),
                })

                if decision.blocked:
                    sig_rec.blocked_reason = f"ML_{decision.action}: {decision.reason}"
                    result.signals.append(sig_rec)
                    return
            except Exception as e:
                logger.warning("ML filter error at %s: %r — proceeding without ML", bar_ts, e)

        # 7) Place order — skip if position already exists
        if signal in {"LONG", "SHORT"}:
            if sim.positions:
                sig_rec.blocked_reason = "ACTIVE_POSITION_EXISTS"
                result.signals.append(sig_rec)
                return

            positions_before = len(sim.positions)
            orders_before = len(sim.orders)
            n_fills_before_entry = len(sim.fills)

            try:
                self.bot.place_demo_market_order(
                    signal, row["candle"], df_window, row
                )
                new_position = len(sim.positions) > positions_before
                new_order = len(sim.orders) > orders_before
                if new_position or new_order:
                    sig_rec.position_opened = True
                else:
                    sig_rec.blocked_reason = "NO_FILL_CREATED"
            except Exception as e:
                msg = f"place_demo_market_order failed at {bar_ts}: {e!r}"
                logger.warning(msg)
                result.errors.append(msg)
                sig_rec.blocked_reason = repr(e)

            for ev in sim.fills[n_fills_before_entry:]:
                result.fills.append(ev)

        result.signals.append(sig_rec)


@dataclass
class ParityReport:
    n_bars_compared: int
    n_signals_matched: int
    n_signals_mismatched: int
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        if self.n_bars_compared == 0:
            return 0.0
        return self.n_signals_matched / self.n_bars_compared

    @property
    def is_pass(self) -> bool:
        return self.n_signals_mismatched == 0 and not self.parse_errors

    def summary(self) -> str:
        return (
            f"ParityReport(bars={self.n_bars_compared}, "
            f"matched={self.n_signals_matched}, "
            f"mismatched={self.n_signals_mismatched}, "
            f"match_rate={self.match_rate:.2%}, "
            f"pass={self.is_pass})"
        )


def parse_live_log_signals(log_path: str | Path) -> dict[pd.Timestamp, str]:
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")
    signals: dict[pd.Timestamp, str] = {}
    current_candle: pd.Timestamp | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("15M CANDLE :"):
                ts_str = line.split(":", 1)[1].strip()
                try:
                    current_candle = pd.Timestamp(ts_str)
                    if current_candle.tzinfo is None:
                        current_candle = current_candle.tz_localize("UTC")
                    else:
                        current_candle = current_candle.tz_convert("UTC")
                except Exception:
                    current_candle = None
            elif line.startswith("FINAL SIGNAL") and current_candle is not None:
                try:
                    sig = line.split(":", 1)[1].strip().upper()
                    if sig in {"LONG", "SHORT", "HOLD"}:
                        signals[current_candle] = sig
                except Exception:
                    pass
            elif line.startswith("FINAL") and current_candle is not None and ":" in line:
                try:
                    sig = line.split(":", 1)[1].strip().upper()
                    if sig in {"LONG", "SHORT", "HOLD"} and current_candle not in signals:
                        signals[current_candle] = sig
                except Exception:
                    pass
    return signals


def parity_test(
    config: dict,
    market_data: MarketData,
    live_log_path: str | Path,
    start_utc: str | pd.Timestamp | None = None,
    end_utc: str | pd.Timestamp | None = None,
) -> ParityReport:
    live_signals = parse_live_log_signals(live_log_path)
    if not live_signals:
        return ParityReport(
            n_bars_compared=0, n_signals_matched=0, n_signals_mismatched=0,
            parse_errors=["no signals parsed from live log"],
        )
    log_start = min(live_signals.keys())
    log_end = max(live_signals.keys())
    if start_utc is None:
        start_utc = log_start
    if end_utc is None:
        end_utc = log_end
    engine = BacktestEngine(config, market_data)
    result = engine.run(start_utc=start_utc, end_utc=end_utc, warmup_bars=300)
    bt_signals = {s.bar_ts: s.final_signal for s in result.signals}
    report = ParityReport(n_bars_compared=0, n_signals_matched=0, n_signals_mismatched=0)
    for ts, live_sig in live_signals.items():
        if ts < pd.Timestamp(start_utc) or ts > pd.Timestamp(end_utc):
            continue
        report.n_bars_compared += 1
        bt_sig = bt_signals.get(ts)
        if bt_sig is None:
            report.n_signals_mismatched += 1
            report.mismatches.append({"bar_ts": str(ts), "live": live_sig, "backtest": "MISSING"})
        elif bt_sig == live_sig:
            report.n_signals_matched += 1
        else:
            report.n_signals_mismatched += 1
            report.mismatches.append({"bar_ts": str(ts), "live": live_sig, "backtest": bt_sig})
    return report


if __name__ == "__main__":
    import sys
    import yaml
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    if len(sys.argv) < 2:
        print("Usage: python -m superbot.backtest.engine <config.yaml> [start] [end]")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    md = load_market_data(cfg["paths"]["data_csv"], use_cache=False)
    start = sys.argv[2] if len(sys.argv) > 2 else None
    end = sys.argv[3] if len(sys.argv) > 3 else None
    engine = BacktestEngine(cfg, md)
    result = engine.run(start_utc=start, end_utc=end)
    print(result.summary())
    print(f"Signals: {result.n_signals}")
    print(f"Trades opened: {result.n_trades_opened}")
    print(f"ML blocked: {result.n_ml_blocked}")
    print(f"Fills: {result.n_fills}")
    print(f"PnL: {result.pnl_susdt:+.2f} SUSDT ({result.pnl_pct:+.2f}%)")
    if result.errors:
        print(f"Errors: {len(result.errors)}")
        for e in result.errors[:5]:
            print(f"  - {e}")