"""
Simulated WEEX DEMO exchange for backtesting.

This module provides a drop-in replacement for the live bot's
`authenticated_get` / `authenticated_post` calls. It is installed
by monkey-patching those functions in the live bot module.

Interface parity:
    - Same URLs (/capi/v3/sim/balance, /position/allPosition, etc.)
    - Same response schemas (list of dicts, same field names)
    - Same side effects (position open/close, balance updates)

Determinism:
    Every fill, fee, funding payment is deterministic given the same
    inputs. No randomness anywhere.

Usage:
    sim = ExchangeSim(config, clock, market_data)
    with sim.installed(live_bot_module):
        # inside this block, all exchange calls are simulated
        ...
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from superbot.backtest.clock import VirtualClock
from superbot.backtest.data_loader import MarketData

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

# WEEX DEMO paths
PATH_BALANCE = "/capi/v3/sim/balance"
PATH_POSITION = "/capi/v3/sim/position/allPosition"
PATH_ORDER = "/capi/v3/sim/order"
PATH_ORDER_HISTORY = "/capi/v3/sim/order/history"
PATH_MARK_PRICE = "/capi/v3/market/symbolPrice"
PATH_TICKER = "/capi/v3/market/ticker"

DEMO_SYMBOL = "BTCSUSDT"
MARKET_SYMBOL = "BTCUSDT"
ASSET = "SUSDT"

LEVERAGE = 5
QTY_STEP = 0.0001
MIN_QTY = 0.0001

# Order statuses (mirror WEEX)
STATUS_NEW = "NEW"
STATUS_FILLED = "FILLED"
STATUS_CANCELED = "CANCELED"
STATUS_UNTRIGGERED = "UNTRIGGERED"


# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------

@dataclass
class SimPosition:
    """A single simulated position (one symbol, one side)."""
    id: int
    symbol: str
    side: str            # "LONG" | "SHORT"
    size: float
    entry_price: float   # weighted average
    open_value: float
    margin_size: float
    leverage: int
    created_time_ms: int
    updated_time_ms: int
    unrealize_pnl: float = 0.0
    liquidate_price: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the exact schema the live bot expects."""
        return {
            "id": self.id,
            "asset": ASSET,
            "symbol": self.symbol,
            "side": self.side,
            "marginType": "ISOLATED",
            "separatedMode": "COMBINED",
            "separatedOpenOrderId": 0,
            "leverage": str(self.leverage),
            "size": f"{self.size:.4f}",
            "openValue": f"{self.open_value:.5f}",
            "openFee": "0",
            "fundingFee": "0",
            "marginSize": f"{self.margin_size:.8f}",
            "isolatedMargin": f"{self.margin_size:.8f}",
            "isAutoAppendIsolatedMargin": False,
            "cumOpenSize": f"{self.size:.4f}",
            "cumOpenValue": f"{self.open_value:.5f}",
            "cumOpenFee": "0",
            "cumCloseSize": "0",
            "cumCloseValue": "0",
            "cumCloseFee": "0",
            "cumFundingFee": "0",
            "cumLiquidateFee": "0",
            "createdMatchSequenceId": 0,
            "updatedMatchSequenceId": 0,
            "createdTime": self.created_time_ms,
            "updatedTime": self.updated_time_ms,
            "unrealizePnl": f"{self.unrealize_pnl:.5f}",
            "liquidatePrice": f"{self.liquidate_price:.1f}",
        }


@dataclass
class SimOrder:
    """A single order record (for order/history endpoint)."""
    order_id: int
    client_order_id: str
    symbol: str
    side: str              # "BUY" | "SELL"
    position_side: str     # "LONG" | "SHORT"
    order_type: str        # "MARKET" | "LIMIT"
    quantity: float
    price: float
    avg_price: float
    executed_qty: float
    status: str
    created_ms: int
    updated_ms: int
    tp_trigger_price: float | None = None
    sl_trigger_price: float | None = None
    reduce_only: bool = False
    fee: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "orderId": self.order_id,
            "clientOrderId": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "positionSide": self.position_side,
            "type": self.order_type,
            "quantity": f"{self.quantity:.4f}",
            "price": f"{self.price:.1f}",
            "avgPrice": f"{self.avg_price:.1f}",
            "executedQty": f"{self.executed_qty:.4f}",
            "status": self.status,
            "time": self.created_ms,
            "updateTime": self.updated_ms,
            "reduceOnly": self.reduce_only,
            "fee": f"{self.fee:.8f}",
        }
        if self.tp_trigger_price is not None:
            d["tpTriggerPrice"] = f"{self.tp_trigger_price:.1f}"
        if self.sl_trigger_price is not None:
            d["slTriggerPrice"] = f"{self.sl_trigger_price:.1f}"
        return d


@dataclass
class FillEvent:
    """A single fill for audit / ledger."""
    timestamp_utc: pd.Timestamp
    order_id: int
    side: str
    position_side: str
    price: float
    quantity: float
    fee: float
    reason: str  # "ENTRY" | "TP1" | "TP2" | "BE" | "TRAIL" | "SL" | "TIMEEXIT" | "THESIS_EXIT"


# ------------------------------------------------------------
# ExchangeSim
# ------------------------------------------------------------

class ExchangeSim:
    """
    Simulated WEEX DEMO exchange.

    Attributes:
        config: parsed config.yaml dict
        clock: VirtualClock
        market: MarketData
        balance: current available balance in SUSDT
        positions: dict[symbol, SimPosition]  (only one symbol supported)
        orders: list[SimOrder]  (append-only)
        order_counter: monotonic order id generator
        fills: list[FillEvent]  (for ledger)
        last_funding_ts: last time funding was applied
    """

    def __init__(
        self,
        config: dict,
        clock: VirtualClock,
        market: MarketData,
        initial_balance: float | None = None,
    ) -> None:
        self.config = config
        self.clock = clock
        self.market = market

        bt = config["backtest"]
        self.initial_balance = (
            float(initial_balance)
            if initial_balance is not None
            else float(bt["initial_balance_susdt"])
        )
        self.balance = self.initial_balance
        self.taker_fee_pct = float(bt["taker_fee_pct"])
        self.maker_fee_pct = float(bt["maker_fee_pct"])
        self.slippage_model = str(bt["slippage_model"])
        self.slippage_bps = float(bt["slippage_bps"])
        self.slippage_atr_fraction = float(bt["slippage_atr_fraction"])
        self.market_fill_price = str(bt["market_fill_price"])
        self.funding_enabled = bool(bt["funding_enabled"])
        self.funding_rate_pct = float(bt["funding_rate_pct_per_8h"])
        self.funding_boundaries = list(bt["funding_boundaries_utc"])

        self.positions: dict[str, SimPosition] = {}
        self.orders: list[SimOrder] = []
        self.order_counter = 1_000_000
        self.fills: list[FillEvent] = []
        self.last_funding_ts: pd.Timestamp | None = None

        # Caches (invalidated on each bar)
        self._mark_price: float | None = None
        self._last_price: float | None = None
        self._mark_ts: pd.Timestamp | None = None

        # Original functions (saved on install)
        self._orig_get = None
        self._orig_post = None
        self._orig_mark = None
        self._orig_last = None
        self._installed = False

    # ==========================================================
    # Price feed
    # ==========================================================

    def set_bar(self, bar_ts: pd.Timestamp, mark: float, last: float) -> None:
        """Called by engine at the start of each bar."""
        self._mark_ts = pd.Timestamp(bar_ts)
        self._mark_price = float(mark)
        self._last_price = float(last)
        if self.funding_enabled:
            self._maybe_apply_funding()

    def mark_price(self) -> float:
        if self._mark_price is None:
            raise RuntimeError("mark_price called before set_bar")
        return self._mark_price

    def last_price(self) -> float:
        if self._last_price is None:
            raise RuntimeError("last_price called before set_bar")
        return self._last_price

    # ==========================================================
    # Funding
    # ==========================================================

    def _maybe_apply_funding(self) -> None:
        """Apply funding if current time crossed an 8h boundary."""
        if self._mark_ts is None:
            return
        now = self._mark_ts
        hour = now.hour
        boundary_hour = max(h for h in self.funding_boundaries if h <= hour)
        boundary = now.replace(hour=boundary_hour, minute=0, second=0, microsecond=0)
        if self.last_funding_ts is None:
            self.last_funding_ts = boundary
            return
        if boundary <= self.last_funding_ts:
            return
        for pos in self.positions.values():
            notional = pos.entry_price * pos.size
            if pos.side == "SHORT":
                payment = notional * self.funding_rate_pct
                self.balance += payment
            else:
                payment = notional * self.funding_rate_pct
                self.balance -= payment
        self.last_funding_ts = boundary
        logger.debug("Funding applied at %s", boundary)

    # ==========================================================
    # Slippage
    # ==========================================================

    def _apply_slippage(self, price: float, side: str, atr: float | None = None) -> float:
        """
        Return fill price after slippage.

        Slippage is always adverse:
            - BUY (open LONG / close SHORT)  -> price goes up
            - SELL (open SHORT / close LONG) -> price goes down
        """
        if self.slippage_model == "zero":
            return price
        if self.slippage_model == "fixed_bps":
            slip_abs = price * (self.slippage_bps / 10_000.0)
        elif self.slippage_model == "atr_fraction":
            if atr is None or atr <= 0:
                slip_abs = 0.0
            else:
                slip_abs = atr * self.slippage_atr_fraction
        else:
            raise ValueError(f"Unknown slippage_model: {self.slippage_model}")

        if side == "BUY":
            return price + slip_abs
        else:
            return price - slip_abs

    def _next_open_price(self) -> float | None:
        """Return next bar's open (current bar's open)."""
        ts = self._mark_ts
        if ts is None:
            return None
        df = self.market.df_15m
        if ts not in df.index:
            return None
        return float(df.loc[ts, "open"])

    # ==========================================================
    # Order placement
    # ==========================================================

    def _gen_order_id(self) -> int:
        oid = self.order_counter
        self.order_counter += 1
        return oid

    def place_market_order(
        self,
        side: str,
        position_side: str,
        quantity: float,
        client_order_id: str,
        tp_trigger: float | None = None,
        sl_trigger: float | None = None,
        reason: str = "ENTRY",
        atr: float | None = None,
    ) -> dict[str, Any]:
        """Simulate a market order. Fills immediately at current bar's open."""
        if quantity < MIN_QTY:
            return {
                "success": False,
                "errorCode": -1013,
                "errorMessage": f"quantity {quantity} below MIN_QTY {MIN_QTY}",
            }

        if self.market_fill_price == "next_open":
            base = self._next_open_price()
            if base is None:
                base = self.mark_price()
        else:
            base = self.mark_price()

        fill = self._apply_slippage(base, side, atr)
        notional = fill * quantity
        fee = notional * self.taker_fee_pct

        oid = self._gen_order_id()
        now_ms = int(self.clock.epoch_ms())
        order = SimOrder(
            order_id=oid,
            client_order_id=client_order_id,
            symbol=DEMO_SYMBOL,
            side=side,
            position_side=position_side,
            order_type="MARKET",
            quantity=quantity,
            price=fill,
            avg_price=fill,
            executed_qty=quantity,
            status=STATUS_FILLED,
            created_ms=now_ms,
            updated_ms=now_ms,
            tp_trigger_price=tp_trigger,
            sl_trigger_price=sl_trigger,
            fee=fee,
        )
        self.orders.append(order)

        self._apply_fill_to_position(
            side=side,
            position_side=position_side,
            quantity=quantity,
            fill_price=fill,
            fee=fee,
            now_ms=now_ms,
        )

        self.fills.append(FillEvent(
            timestamp_utc=self.clock.now(),
            order_id=oid,
            side=side,
            position_side=position_side,
            price=fill,
            quantity=quantity,
            fee=fee,
            reason=reason,
        ))

        self.balance -= fee

        return {
            "success": True,
            "orderId": str(oid),
            "clientOrderId": client_order_id,
            "errorCode": None,
            "errorMessage": None,
        }

    def _apply_fill_to_position(
        self,
        side: str,
        position_side: str,
        quantity: float,
        fill_price: float,
        fee: float,
        now_ms: int,
    ) -> None:
        """Update internal position state after a fill."""
        symbol = DEMO_SYMBOL
        existing = self.positions.get(symbol)

        # Case 1: opening a new position
        if existing is None:
            margin = (fill_price * quantity) / LEVERAGE
            self.positions[symbol] = SimPosition(
                id=now_ms,
                symbol=symbol,
                side=position_side,
                size=quantity,
                entry_price=fill_price,
                open_value=fill_price * quantity,
                margin_size=margin,
                leverage=LEVERAGE,
                created_time_ms=now_ms,
                updated_time_ms=now_ms,
            )
            return

        # Case 2: reducing existing position
        if (existing.side == "LONG" and side == "SELL") or \
           (existing.side == "SHORT" and side == "BUY"):
            close_qty = min(quantity, existing.size)
            remaining_qty = existing.size - close_qty
            if existing.side == "LONG":
                pnl = (fill_price - existing.entry_price) * close_qty
            else:
                pnl = (existing.entry_price - fill_price) * close_qty
            self.balance += pnl

            if remaining_qty < MIN_QTY:
                del self.positions[symbol]
            else:
                existing.size = remaining_qty
                existing.open_value = existing.entry_price * remaining_qty
                existing.margin_size = (existing.entry_price * remaining_qty) / LEVERAGE
                existing.updated_time_ms = now_ms
            return

        # Case 3: adding to existing position (not allowed)
        logger.warning(
            "Pyramiding not supported: existing %s, fill %s",
            existing.side, side,
        )

    # ==========================================================
    # TP/SL checking
    # ==========================================================

    def check_tp_sl_triggers(self, atr: float | None = None) -> list[FillEvent]:
        """Check every open position's TP/SL orders against current mark price."""
        triggered: list[FillEvent] = []
        symbol = DEMO_SYMBOL
        pos = self.positions.get(symbol)
        if pos is None:
            return triggered

        mark = self.mark_price()

        entry_order = None
        for o in reversed(self.orders):
            if o.symbol == symbol and o.status == STATUS_FILLED and \
               (o.tp_trigger_price is not None or o.sl_trigger_price is not None) and \
               o.reduce_only is False:
                entry_order = o
                break
        if entry_order is None:
            return triggered

        sl = entry_order.sl_trigger_price
        tp = entry_order.tp_trigger_price

        if sl is not None:
            hit = (pos.side == "LONG" and mark <= sl) or \
                  (pos.side == "SHORT" and mark >= sl)
            if hit:
                ev = self._force_close(pos, price=sl, reason="SL", atr=atr)
                if ev:
                    triggered.append(ev)
                return triggered

        if tp is not None:
            hit = (pos.side == "LONG" and mark >= tp) or \
                  (pos.side == "SHORT" and mark <= tp)
            if hit:
                ev = self._force_close(pos, price=tp, reason="TP", atr=atr)
                if ev:
                    triggered.append(ev)
                return triggered

        return triggered

    def _force_close(
        self,
        pos: SimPosition,
        price: float,
        reason: str,
        atr: float | None = None,
    ) -> FillEvent | None:
        """Close an entire position at the given trigger price."""
        side = "SELL" if pos.side == "LONG" else "BUY"
        notional = price * pos.size
        fee = notional * self.maker_fee_pct
        oid = self._gen_order_id()
        now_ms = int(self.clock.epoch_ms())

        order = SimOrder(
            order_id=oid,
            client_order_id=f"SIM-{reason}-{oid}",
            symbol=pos.symbol,
            side=side,
            position_side=pos.side,
            order_type="MARKET",
            quantity=pos.size,
            price=price,
            avg_price=price,
            executed_qty=pos.size,
            status=STATUS_FILLED,
            created_ms=now_ms,
            updated_ms=now_ms,
            reduce_only=True,
            fee=fee,
        )
        self.orders.append(order)

        if pos.side == "LONG":
            pnl = (price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - price) * pos.size
        self.balance += pnl - fee

        fill_event = FillEvent(
            timestamp_utc=self.clock.now(),
            order_id=oid,
            side=side,
            position_side=pos.side,
            price=price,
            quantity=pos.size,
            fee=fee,
            reason=reason,
        )
        self.fills.append(fill_event)
        del self.positions[pos.symbol]
        return fill_event

    # ==========================================================
    # Unrealized PnL recompute (called per bar)
    # ==========================================================

    def refresh_unrealized_pnl(self) -> None:
        mark = self.mark_price()
        for pos in self.positions.values():
            if pos.side == "LONG":
                pos.unrealize_pnl = (mark - pos.entry_price) * pos.size
            else:
                pos.unrealize_pnl = (pos.entry_price - mark) * pos.size

    # ==========================================================
    # HTTP-style handlers (drop-in for authenticated_get/post)
    # ==========================================================

    def handle_get(self, path: str, params: dict | None = None) -> Any:
        """Return the simulated response for a GET request."""
        params = params or {}
        if path == PATH_BALANCE:
            return [{
                "asset": ASSET,
                "balance": f"{self.balance:.8f}",
                "availableBalance": f"{self.balance:.8f}",
                "frozen": "0",
                "unrealizePnl": f"{sum(p.unrealize_pnl for p in self.positions.values()):.8f}",
            }]
        if path == PATH_POSITION:
            return [p.to_dict() for p in self.positions.values()]
        if path == PATH_ORDER_HISTORY:
            limit = int(params.get("limit", 100))
            rows = [o.to_dict() for o in self.orders if o.symbol == params.get("symbol", DEMO_SYMBOL)]
            return rows[-limit:]
        if path == PATH_MARK_PRICE:
            return {"symbol": MARKET_SYMBOL, "price": f"{self.mark_price():.1f}"}
        if path == PATH_TICKER:
            return {"symbol": MARKET_SYMBOL, "lastPrice": f"{self.last_price():.1f}"}
        raise NotImplementedError(f"ExchangeSim: unhandled GET path {path}")

    def handle_post(self, path: str, body: dict) -> Any:
        """Return the simulated response for a POST request."""
        if path != PATH_ORDER:
            raise NotImplementedError(f"ExchangeSim: unhandled POST path {path}")

        side = str(body.get("side", "")).upper()
        position_side = str(body.get("positionSide", "")).upper()
        qty = float(body.get("quantity", 0))
        cid = str(body.get("newClientOrderId", ""))

        tp = None
        sl = None
        if "tpTriggerPrice" in body:
            try:
                tp = float(body["tpTriggerPrice"])
            except (TypeError, ValueError):
                tp = None
        if "slTriggerPrice" in body:
            try:
                sl = float(body["slTriggerPrice"])
            except (TypeError, ValueError):
                sl = None

        reason = "ENTRY"
        cid_upper = cid.upper()
        for token, r in (
            ("-TP1-", "TP1"), ("-TP2-", "TP2"),
            ("-TRAIL-", "TRAIL"), ("-BE-", "BE"),
            ("-SL-", "SL"), ("-TIMEEXIT", "TIMEEXIT"),
            ("-THESIS", "THESIS_EXIT"), ("-SLIPPAGE", "SLIPPAGE_EXIT"),
        ):
            if token in cid_upper:
                reason = r
                break

        existing = self.positions.get(DEMO_SYMBOL)
        if existing is None:
            return self.place_market_order(
                side=side,
                position_side=position_side,
                quantity=qty,
                client_order_id=cid,
                tp_trigger=tp,
                sl_trigger=sl,
                reason="ENTRY",
            )
        else:
            return self.place_market_order(
                side=side,
                position_side=position_side,
                quantity=qty,
                client_order_id=cid,
                tp_trigger=None,
                sl_trigger=None,
                reason=reason,
            )

    # ==========================================================
    # Install / uninstall
    # ==========================================================

    def install(self, bot_module: Any) -> None:
        """
        Monkey-patch authenticated_get/post and direct price functions
        in the bot module.
        """
        if self._installed:
            return
        self._orig_get = bot_module.authenticated_get
        self._orig_post = bot_module.authenticated_post

        sim = self

        def _get(path, params=None):
            return sim.handle_get(path, params)

        def _post(path, body):
            return sim.handle_post(path, body)

        bot_module.authenticated_get = _get   # type: ignore[assignment]
        bot_module.authenticated_post = _post  # type: ignore[assignment]

        # Patch direct price-feed functions that use SESSION directly
        # (the live bot's get_demo_mark_price / get_demo_last_price use
        # SESSION.get() and would crash with SESSION=None)
        def _mark(force_refresh=False):
            return sim.mark_price()

        def _last(force_refresh=False):
            return sim.last_price()

        if hasattr(bot_module, "get_demo_mark_price"):
            self._orig_mark = bot_module.get_demo_mark_price
            bot_module.get_demo_mark_price = _mark
        if hasattr(bot_module, "get_demo_last_price"):
            self._orig_last = bot_module.get_demo_last_price
            bot_module.get_demo_last_price = _last

        # Also neutralize SESSION to prevent accidental network calls
        if hasattr(bot_module, "SESSION"):
            bot_module.SESSION = None

        self._installed = True
        logger.info("ExchangeSim installed into bot module")

    def uninstall(self, bot_module: Any) -> None:
        if not self._installed:
            return
        if self._orig_get is not None:
            bot_module.authenticated_get = self._orig_get  # type: ignore[assignment]
        if self._orig_post is not None:
            bot_module.authenticated_post = self._orig_post  # type: ignore[assignment]
        if self._orig_mark is not None:
            bot_module.get_demo_mark_price = self._orig_mark
        if self._orig_last is not None:
            bot_module.get_demo_last_price = self._orig_last
        self._installed = False
        logger.info("ExchangeSim uninstalled")


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
        print("config.yaml not found in cwd")
        raise SystemExit(1)

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from superbot.backtest.data_loader import load_market_data
    md = load_market_data(cfg["paths"]["data_csv"], symbol="BTCUSDT", use_cache=False)

    clock = VirtualClock.from_iso(str(md.df_15m.index[0]))
    sim = ExchangeSim(cfg, clock, md)

    with clock.installed():
        bar_ts = md.df_15m.index[0]
        mark = float(md.df_15m.loc[bar_ts, "close"])
        sim.set_bar(bar_ts, mark, mark)

        print(f"Mark price: {sim.mark_price()}")
        print(f"Balance: {sim.balance}")

        resp = sim.handle_post("/capi/v3/sim/order", {
            "symbol": DEMO_SYMBOL,
            "side": "SELL",
            "positionSide": "SHORT",
            "type": "MARKET",
            "quantity": "0.1",
            "newClientOrderId": "TEST-SHORT-1",
            "tpTriggerPrice": f"{mark - 500:.1f}",
            "slTriggerPrice": f"{mark + 500:.1f}",
        })
        print(f"Order response: {resp}")

        next_ts = md.df_15m.index[1]
        clock.set(next_ts)
        next_mark = float(md.df_15m.loc[next_ts, "close"])
        sim.set_bar(next_ts, next_mark, next_mark)
        sim.refresh_unrealized_pnl()

        positions = sim.handle_get("/capi/v3/sim/position/allPosition", {})
        print(f"Positions: {positions}")
        print(f"Balance after entry: {sim.balance}")