# V167.8.9 ONLINE + DEMO BUILD
# Default mode = WEEX DEMO live runner (PRIMARY).
# Set V167_OFFLINE_TEST=1 only for offline self-tests.
# ------------------------------------------------------------------
# CHANGES FROM V167.8.8:
#   [V167.8.9-1] REMOVED: duplicate _normalize_utc_timestamp (defined twice)
#   [V167.8.9-2] FIXED:   _reconstruct_entry_atr_if_needed now uses the
#                         symmetric helper _risk_distance_is_sane()
#   [V167.8.9-3] ADDED:   _modify_exchange_tpsl() + post-fill reconcile
#                         now also updates exchange-side SL/TP so that
#                         actual fill price is the single source of truth.
#   [V167.8.9-4] FIXED:   Trailing activation now tied to effective TP2
#                         (tp2_r_effective) instead of hard-coded 1.75R.
#   [V167.8.9-5] ADDED:   proper dual __main__ handling. Online demo is
#                         default; offline self-tests require env var.
# ------------------------------------------------------------------

import os
import ast
import time
import json
import socket
from pathlib import Path
import hmac
import hashlib
import base64
from urllib.parse import urlencode
import requests
import pandas as pd
import numpy as np
import random


# ============================================================
# 1) CREDENTIALS
# ============================================================

API_KEY = "weex_450e52d17b4e157dca1513deebfdd86e"
SECRET_KEY = "b8778ac3b0c331979e6e896be2fb369c3c2d495b3ce0f6c678d5d77f52d784f1"
PASSPHRASE = "Majid13611361"


# ============================================================
# 2) SETTINGS
# ============================================================

TIMEFRAME = "15m"
MARKET_SYMBOL = "BTCUSDT"
DEMO_SYMBOL = "BTCSUSDT"

LOOKBACK_DAYS = 90

HISTORY_LIMIT = 100
CANDLE_MINUTES = 15

CANDLE_POLL_SECONDS = 30
EXIT_ACTIVE_POSITION_POLL_SECONDS = 1.0
EXIT_IDLE_POLL_SECONDS = 10.0
REQUEST_TIMEOUT = 20
RETRY_SECONDS = 2.0
NETWORK_RETRY_SECONDS = 15

EXIT_VERIFY_MAX_WAIT = 2.0
EXIT_VERIFY_POLL_SECONDS = 0.25
PENDING_EXIT_TIMEOUT = 3.0
POSITION_CACHE_TTL = 0.40
_MARK_PRICE_CACHE_TTL = 0.75
_MARK_PRICE_CACHE = {"price": None, "ts": 0.0}
_LAST_PRICE_CACHE = {"price": None, "ts": 0.0}

HARD_SL_WATCHDOG_ENABLED = True
HARD_SL_TRIGGER_TOLERANCE = 0.10

DATA_FILE = "weex_15m_v1677.csv"
STATE_FILE = "v1677_runtime_state.json"
SHARED_STATE_FILE = "v1677_shared_state.json"
EXIT_STATE_FILE = "v1677_exit_state.json"
EXIT_LOG_FILE = "v1677_exit_log.jsonl"
ENTRY_ATR_FILE = "v1677_entry_atr.json"

DEMO_ONLY = True
DEMO_ORDERS_ENABLED = True
QTY_STEP = 0.0001
MIN_QTY = 0.0001
TP_SL_ENABLED = True

ENTRY_THRESHOLD = 4
MIN_EDGE_GAP = 2
MIN_CONFIDENCE = 55

PRICE_MOVE_THRESHOLD = 0.20
REDUCED_SIZE_THRESHOLD = 0.25
REDUCED_SIZE_RATIO = 0.50

HEARTBEAT_INTERVAL = 30
FAILOVER_TIMEOUT = 300

BREAKEVEN_TRIGGER_R = 1.00
BREAKEVEN_FEE_BUFFER_PCT = 0.0020

TP1_R = 1.25
TP2_R = 1.75
TP1_CLOSE_PCT = 0.50
TP2_CLOSE_PCT = 0.25

TRAILING_ENABLED = True
TRAILING_ACTIVATION_R = 1.75
TRAILING_DISTANCE_R = 0.50
TRAILING_MODE = "R_BASED"
TRAILING_PERCENT = 0.005
TRAILING_ATR_MULT = 1.5

TIME_EXIT_ENABLED = False
TIME_EXIT_HOURS = 12
TIME_EXIT_MIN_PROFIT_R = 0.0

EXIT_VERIFICATION_ENABLED = True

CIRCUIT_BREAKER_FILE = "v1677_circuit_breaker.json"
CIRCUIT_MAX_LOSSES = 3
CIRCUIT_COOLDOWN_CANDLES = 10

CONFLUENCE_SCORING_ENABLED = True
CONFLUENCE_MAX_BONUS = 1

MAX_MARGIN_UTILIZATION = 0.90

FAILSAFE_TRIGGER_PRICE_TYPE = "CONTRACT_PRICE"

POLL_INTERVAL_TIGHT = 0.10
POLL_INTERVAL_NEAR  = 0.25
POLL_INTERVAL_PROFIT = 0.50
POLL_INTERVAL_NORMAL = 1.00

NY_SESSION_FILTER_ENABLED = True
NY_SESSION_FILTER_START_HOUR_UTC = 11
NY_SESSION_FILTER_END_HOUR_UTC = 14

DANGER_WINDOW_ATR_SHOCK = 1.75
DANGER_WINDOW_BODY_ATR = 1.50
DANGER_WINDOW_3CANDLE_ATR = 4.00

WEEX_LAST_PRICE_PATHS = [
    ("/capi/v3/market/ticker",       {"symbol": "BTCUSDT"}, "lastPrice"),
    ("/capi/v3/market/ticker/price", {"symbol": "BTCUSDT"}, "price"),
    ("/capi/v3/market/symbolPrice",  {"symbol": "BTCUSDT", "priceType": "CONTRACT"}, "price"),
    ("/capi/v3/market/symbolPrice",  {"symbol": "BTCUSDT", "priceType": "LAST"}, "price"),
]

TP_COMPRESSION_ENABLED = True
TP_COMPRESSION_ATR_RATIO = 0.50
TP_COMPRESSION_MIN_AGE_HOURS = 8
TP_COMPRESSION_TP1_R = 1.00
TP_COMPRESSION_TP2_R = 1.50

RISK_DISTANCE_SANITY_TOLERANCE = 0.50
ATR_SANITY_MIN_PCT = 0.0003
ATR_SANITY_MAX_PCT = 0.03

THESIS_OBSERVATION_ENABLED = True

LAST_PRICE_TELEMETRY_ENABLED = True
LAST_PRICE_TELEMETRY_FILE = "v1677_last_price_telemetry.json"
LAST_PRICE_TELEMETRY_FLUSH_SECONDS = 60.0

# V167.8.8 HARDENING
CLOSE_RECONCILIATION_ENABLED = True
ORDER_HISTORY_LOOKBACK_LIMIT = 100
EXTERNAL_CLOSE_LOOKBACK_MS = 24 * 60 * 60 * 1000
EXTERNAL_CLOSE_PRICE_TOLERANCE_R = 0.20
RISK_DISTANCE_SANITY_LOW = 1.0 - RISK_DISTANCE_SANITY_TOLERANCE
RISK_DISTANCE_SANITY_HIGH = 1.0 + RISK_DISTANCE_SANITY_TOLERANCE
AGE_UNKNOWN_SAFE_HOLD = True
EXCHANGE_PROTECTION_RECONCILIATION_ENABLED = True
EXCHANGE_OPEN_ORDERS_PATH = "/capi/v3/openOrders"
EXCHANGE_TPSL_MODIFY_PATH = "/capi/v3/modifyTpSlOrder"


# ============================================================
# OPERATION MODE
# ============================================================
def get_operation_mode():
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            mode = config.get("operation_mode", "PRIMARY").upper()
            if mode in {"PRIMARY", "BACKUP"}:
                return mode
            print(f"⚠️ config.json operation_mode={mode!r} invalid, defaulting to PRIMARY")
            return "PRIMARY"
    except FileNotFoundError:
        return "PRIMARY"
    except Exception as e:
        print("⚠️ CONFIG READ ERROR (defaulting to PRIMARY):", repr(e))
        return "PRIMARY"

OPERATION_MODE = get_operation_mode()
SYSTEM_ID = socket.gethostname()

ACCOUNT_RISK_PERCENT = 0.02
MAX_LEVERAGE = 5

EXIT_TP_ATR_MULT_MIN = 0.50
EXIT_TP_ATR_MULT_MAX = 2.50
EXIT_SL_ATR_MULT_BASE = 1.50

FAILSAFE_TP_R = 2.50
FAILSAFE_SL_R = 1.25

VOLUME_SPIKE_ENABLED = True
VOLUME_SPIKE_THRESHOLD = 2.5

DECAY_ENABLED = True
DECAY_STATE_FILE = "v1677_decay_state.json"

BREAKEVEN_ENABLED = True
BREAKEVEN_STATE_FILE = "v1677_breakeven_state.json"
TRIGGER_PRICE_TYPE = "MARK_PRICE"

PA_LOOKBACK = 8

RISK_GUARD_ENABLED = True
RISK_SINGLE_POSITION = True
RISK_MAX_BODY_ATR = 2.00
RISK_MAX_EMA_DISTANCE_ATR = 2.00
RISK_MAX_ATR_SHOCK_RATIO = 2.25
RISK_MIN_RR = 0.45
RISK_ATR_MEDIAN_WINDOW = 20

D_MIN_EMA_ATR = 0.25
D_MIN_BODY_ATR = 0.25

SIG_H4_VR_MIN = 0.10
SIG_H4_EG_LONG_MIN = 0.015
SIG_H4_EG_SHORT_MAX = -0.015
SIG_H4_ATRP_MIN = 0.007
SIG_H1_VR_MIN = 0.30
SIG_H4_RSI_LONG_RANGE = (55, 70)
SIG_H4_RSI_SHORT_RANGE = (30, 38)
SIG_H1_RSI_LONG_MIN = 50
SIG_H1_RSI_SHORT_MAX = 48


# ============================================================
# DETERMINISTIC ORDER REJECTION
# ============================================================
class DeterministicOrderRejection(RuntimeError):
    pass


DETERMINISTIC_REJECT_CODES = {
    -1054, -1100, -1130, -2010, -2011, -2019, -4164, -4165,
}


# ============================================================
# WEEX WORKING TYPE HELPER
# ============================================================
def _weex_working_type():
    v = str(FAILSAFE_TRIGGER_PRICE_TYPE).upper().strip()
    if v in {"LAST_PRICE", "LAST", "CONTRACT_PRICE"}:
        return "CONTRACT_PRICE"
    if v in {"MARK_PRICE", "MARK"}:
        return "MARK_PRICE"
    print(f"⚠️ Unknown FAILSAFE_TRIGGER_PRICE_TYPE={FAILSAFE_TRIGGER_PRICE_TYPE!r} "
          f"→ defaulting to MARK_PRICE for WEEX API")
    return "MARK_PRICE"

# ============================================================
# SESSION FILTER HELPERS
# ============================================================
def is_in_high_risk_session(now_utc=None):
    if not NY_SESSION_FILTER_ENABLED:
        return False
    if now_utc is None:
        now_utc = pd.Timestamp.now(tz="UTC")
    h = int(now_utc.hour)
    return NY_SESSION_FILTER_START_HOUR_UTC <= h < NY_SESSION_FILTER_END_HOUR_UTC


def high_risk_session_label(now_utc=None):
    if now_utc is None:
        now_utc = pd.Timestamp.now(tz="UTC")
    tehran = now_utc + pd.Timedelta(hours=3, minutes=30)
    return (
        f"DANGER WINDOW [{NY_SESSION_FILTER_START_HOUR_UTC:02d}:00, "
        f"{NY_SESSION_FILTER_END_HOUR_UTC:02d}:00) UTC "
        f"| now UTC={now_utc.strftime('%Y-%m-%d %H:%M')} "
        f"| Tehran={tehran.strftime('%H:%M')}"
    )


def is_high_risk_conditions(df, atr):
    if df is None or len(df) < 20 or atr is None or atr <= 0:
        return True, "INSUFFICIENT DATA FOR CONDITION CHECK"

    x = df.copy()
    for c in ("open", "high", "low", "close"):
        x[c] = x[c].astype(float)

    close = x["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(14).mean().dropna()
    if len(atr_series) >= RISK_ATR_MEDIAN_WINDOW:
        baseline = float(atr_series.iloc[-RISK_ATR_MEDIAN_WINDOW:].median())
        if baseline > 0:
            shock_ratio = atr / baseline
            if shock_ratio > DANGER_WINDOW_ATR_SHOCK:
                return True, f"ATR SHOCK {shock_ratio:.2f}x > {DANGER_WINDOW_ATR_SHOCK:.2f}"

    last = x.iloc[-1]
    body = abs(float(last["close"]) - float(last["open"]))
    body_atr = body / atr
    if body_atr > DANGER_WINDOW_BODY_ATR:
        return True, f"BODY/ATR {body_atr:.2f} > {DANGER_WINDOW_BODY_ATR:.2f}"

    if len(x) >= 3:
        last3 = x.iloc[-3:]
        rng3 = float(last3["high"].max() - last3["low"].min())
        rng3_atr = rng3 / atr
        if rng3_atr > DANGER_WINDOW_3CANDLE_ATR:
            return True, f"3-CANDLE RANGE {rng3_atr:.2f} ATR > {DANGER_WINDOW_3CANDLE_ATR:.2f}"

    return False, "CONDITIONS NORMAL"


# ============================================================
# THESIS VALIDITY CHECK
# ============================================================
def is_entry_thesis_valid(item, row):
    if row is None:
        return True

    side = str(item.get("side", "")).upper()
    try:
        h4_eg = float(row.get("H4_eg", 0.0))
    except Exception:
        return True

    if side == "SHORT":
        h4_bearish = h4_eg <= -0.005
        h1_close = float(row.get("H1_close", 0.0))
        h1_e200 = float(row.get("H1_e200", 0.0))
        h1_e20 = float(row.get("H1_e20", 0.0))
        h1_e50 = float(row.get("H1_e50", 0.0))
        h1_rsi = float(row.get("H1_rsi", 50.0))
        h1_short_checks = sum([
            h1_close < h1_e200,
            h1_e20 < h1_e50,
            h1_rsi <= 50.0,
        ])
        return bool(h4_bearish and h1_short_checks >= 2)

    elif side == "LONG":
        h4_bullish = h4_eg >= 0.005
        h1_close = float(row.get("H1_close", 0.0))
        h1_e200 = float(row.get("H1_e200", 0.0))
        h1_e20 = float(row.get("H1_e20", 0.0))
        h1_e50 = float(row.get("H1_e50", 0.0))
        h1_rsi = float(row.get("H1_rsi", 50.0))
        h1_long_checks = sum([
            h1_close > h1_e200,
            h1_e20 > h1_e50,
            h1_rsi >= 50.0,
        ])
        return bool(h4_bullish and h1_long_checks >= 2)

    return True


# ============================================================
# PERSISTENT ENTRY_ATR SNAPSHOT
# ============================================================
def _entry_atr_key(symbol, side, created_ts_ms):
    return f"{str(symbol).upper()}_{str(side).upper()}_{int(created_ts_ms)}"


def _load_entry_atr_snapshots():
    try:
        path = Path(ENTRY_ATR_FILE)
        if not path.exists():
            return {}
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        data = json.loads(content)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"⚠️ ENTRY_ATR LOAD ERROR (treating as empty): {repr(e)}")
        return {}


def _save_entry_atr_snapshot(symbol, side, created_ts_ms, entry_atr, entry_price, risk_distance):
    try:
        snapshots = _load_entry_atr_snapshots()
        key = _entry_atr_key(symbol, side, created_ts_ms)
        snapshots[key] = {
            "symbol": str(symbol).upper(),
            "side": str(side).upper(),
            "created_ts_ms": int(created_ts_ms),
            "entry_atr": float(entry_atr),
            "entry_price": float(entry_price),
            "risk_distance": float(risk_distance),
            "saved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        tmp = ENTRY_ATR_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshots, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, ENTRY_ATR_FILE)
        print(f"💾 V167.8.9 ENTRY_ATR SNAPSHOT SAVED: key={key} "
              f"entry_atr={entry_atr:.2f} risk_distance={risk_distance:.2f}")
    except Exception as e:
        print(f"⚠️ ENTRY_ATR SAVE ERROR: {repr(e)}")


def _lookup_entry_atr_snapshot(symbol, side, created_ts_ms):
    if created_ts_ms is None:
        return None
    snapshots = _load_entry_atr_snapshots()
    key = _entry_atr_key(symbol, side, created_ts_ms)
    return snapshots.get(key)


def _cleanup_entry_atr_snapshots(active_keys):
    try:
        snapshots = _load_entry_atr_snapshots()
        to_remove = [k for k in snapshots.keys() if k not in active_keys]
        if not to_remove:
            return
        for k in to_remove:
            snapshots.pop(k, None)
        tmp = ENTRY_ATR_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshots, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ENTRY_ATR_FILE)
        if to_remove:
            print(f"🧹 V167.8.9 ENTRY_ATR CLEANUP: removed {len(to_remove)} stale snapshot(s)")
    except Exception as e:
        print(f"⚠️ ENTRY_ATR CLEANUP ERROR: {repr(e)}")


# ============================================================
# REBUILD ENTRY_ATR FROM HISTORICAL DATA (VERIFIED)
# ============================================================
def _rebuild_entry_atr_from_history(df, created_ts):
    if df is None or len(df) < 20 or created_ts is None:
        return None
    try:
        x = df.copy()
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
        x = x.set_index("timestamp")
        if x.empty:
            return None

        candle_floor = created_ts.floor("15min")
        if candle_floor in x.index:
            entry_candle_ts = candle_floor
        else:
            try:
                candidates = x.index[x.index <= created_ts]
                if len(candidates) == 0:
                    return None
                entry_candle_ts = candidates[-1]
            except Exception:
                return None

        high = x["high"].astype(float)
        low = x["low"].astype(float)
        close = x["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean()

        atr_at_open = atr_series.asof(entry_candle_ts)
        if pd.isna(atr_at_open):
            print(f"🚫 V167.8.9 HISTORY ATR: NaN at candle {entry_candle_ts}")
            return None
        atr_val = float(atr_at_open)
        if not np.isfinite(atr_val) or atr_val <= 0:
            print(f"🚫 V167.8.9 HISTORY ATR: invalid value {atr_val}")
            return None

        entry_close = float(x.loc[entry_candle_ts, "close"])
        if entry_close <= 0:
            print(f"🚫 V167.8.9 HISTORY ATR: entry_close={entry_close} invalid")
            return None
        atr_pct = atr_val / entry_close
        if atr_pct < ATR_SANITY_MIN_PCT or atr_pct > ATR_SANITY_MAX_PCT:
            print(f"🚫 V167.8.9 HISTORY ATR SANITY FAIL: "
                  f"candle={entry_candle_ts} atr={atr_val:.2f} "
                  f"entry={entry_close:.2f} atr_pct={atr_pct*100:.3f}% "
                  f"(outside [{ATR_SANITY_MIN_PCT*100:.2f}%, "
                  f"{ATR_SANITY_MAX_PCT*100:.2f}%])")
            return None

        print(f"✅ V167.8.9 HISTORY ATR VERIFIED: candle={entry_candle_ts} "
              f"atr={atr_val:.2f} atr_pct={atr_pct*100:.3f}%")
        return atr_val
    except Exception as e:
        print(f"⚠️ REBUILD ENTRY_ATR ERROR: {repr(e)}")
        return None


# ============================================================
# SINGLE SOURCE OF TRUTH: _normalize_utc_timestamp
# (V167.8.9-1: the duplicate definition was removed)
# ============================================================
def _normalize_utc_timestamp(value):
    """Return a tz-aware UTC timestamp, or None for invalid/legacy values."""
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if ts.year < 2020 or ts.year > 2100:
            return None
        return ts
    except Exception:
        return None


def _risk_distance_is_sane(current_risk_distance, trusted_risk_distance,
                            min_ratio=0.50, max_ratio=1.50):
    """V167.8.9: symmetric recovery sanity check (single helper)."""
    try:
        cur = float(current_risk_distance)
        ref = float(trusted_risk_distance)
        if cur <= 0 or ref <= 0:
            return False
        ratio = cur / ref
        return min_ratio <= ratio <= max_ratio
    except Exception:
        return False


# ============================================================
# SHARED STATE FUNCTIONS
# ============================================================
_SHARED_STATE_LOCK_FILE = SHARED_STATE_FILE + ".lock"
_SHARED_STATE_LOCK_TIMEOUT = 5.0
_SHARED_STATE_LOCK_STALE_AGE = 15.0


def _acquire_shared_state_lock(timeout=_SHARED_STATE_LOCK_TIMEOUT):
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(_SHARED_STATE_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(_SHARED_STATE_LOCK_FILE)
                if age > _SHARED_STATE_LOCK_STALE_AGE:
                    print(f"⚠️ SHARED STATE LOCK STALE ({age:.1f}s old) — reclaiming")
                    os.remove(_SHARED_STATE_LOCK_FILE)
                    continue
            except FileNotFoundError:
                continue
            except Exception:
                pass
            if time.time() >= deadline:
                print("⚠️ SHARED STATE LOCK TIMEOUT — proceeding WITHOUT lock (best effort)")
                return False
            time.sleep(0.05)


def _release_shared_state_lock():
    try:
        os.remove(_SHARED_STATE_LOCK_FILE)
    except FileNotFoundError:
        pass
    except Exception as e:
        print("⚠️ SHARED STATE LOCK RELEASE ERROR:", repr(e))


def _load_shared_state_unlocked():
    try:
        with open(SHARED_STATE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        pass
    except Exception as e:
        print("⚠️ SHARED STATE READ ERROR (using defaults):", repr(e))
    return {
        "system_id": None,
        "status": "UNKNOWN",
        "last_candle": None,
        "last_heartbeat": None,
        "mode": "PRIMARY",
    }


def load_shared_state():
    return _load_shared_state_unlocked()


def save_shared_state(last_candle=None, status=None, mode=None):
    got_lock = _acquire_shared_state_lock()
    try:
        state = _load_shared_state_unlocked()
        state["system_id"] = SYSTEM_ID
        state["last_heartbeat"] = pd.Timestamp.now(tz="UTC").isoformat()
        if last_candle is not None:
            state["last_candle"] = str(last_candle)
        if status is not None:
            state["status"] = status
        if mode is not None:
            state["mode"] = mode
        tmp = SHARED_STATE_FILE + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, SHARED_STATE_FILE)
        except Exception as e:
            print("🚨 SHARED STATE SAVE ERROR:", repr(e))
    finally:
        if got_lock:
            _release_shared_state_lock()


def is_primary_alive():
    state = load_shared_state()
    if state.get("mode") != "PRIMARY":
        return False
    last_heartbeat = state.get("last_heartbeat")
    if last_heartbeat is None:
        return False
    last_heartbeat = pd.Timestamp(last_heartbeat)
    time_since = (pd.Timestamp.now(tz="UTC") - last_heartbeat).total_seconds()
    return time_since < FAILOVER_TIMEOUT


def can_send_order(current_candle):
    if OPERATION_MODE == "BACKUP":
        if is_primary_alive():
            print(f"🔒 BACKUP: PRIMARY is alive ({load_shared_state().get('system_id')}), skipping order")
            return False
        else:
            print(f"🔄 BACKUP: PRIMARY is down, taking over")
            return True
    state = load_shared_state()
    if state.get("last_candle") == str(current_candle):
        other_system = state.get("system_id")
        if other_system and other_system != SYSTEM_ID:
            print(f"⏳ System {other_system} already processed candle {current_candle}")
            return False
    return True

# ============================================================
# API
# ============================================================
BASE_URL = "https://api-contract.weex.com"

SERVER_TIME_PATH = "/capi/v3/market/time"
HISTORY_PATH = "/capi/v3/market/historyKlines"
BALANCE_PATH = "/capi/v3/sim/balance"
POSITION_PATH = "/capi/v3/sim/position/allPosition"
DEMO_ORDER_PATH = "/capi/v3/sim/order"
ORDER_HISTORY_PATH = "/capi/v3/sim/order/history"
MARK_PRICE_PATH = "/capi/v3/market/symbolPrice"

SESSION = requests.Session()


def clean_key(value):
    return str(value).strip()


API_KEY = clean_key(API_KEY)
SECRET_KEY = clean_key(SECRET_KEY)
PASSPHRASE = clean_key(PASSPHRASE)


def check_credentials():
    if not API_KEY:
        raise RuntimeError("WEEX_API_KEY is empty.")
    if not SECRET_KEY:
        raise RuntimeError("WEEX_SECRET_KEY is empty.")
    if not PASSPHRASE:
        raise RuntimeError("WEEX_PASSPHRASE is empty.")


def get_server_time_ms():
    r = SESSION.get(BASE_URL + SERVER_TIME_PATH, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected server-time response: {data}")
    ts = data.get("serverTime")
    if ts is None:
        raise RuntimeError(f"serverTime missing: {data}")
    ts = int(float(ts))
    if ts < 10_000_000_000:
        ts *= 1000
    return ts


def get_server_utc():
    return pd.to_datetime(get_server_time_ms(), unit="ms", utc=True)


def make_signature(timestamp, method, request_path, query_string="", body=""):
    method = method.upper()
    message = str(timestamp) + method + request_path
    if query_string:
        message += "?" + query_string
    message += body
    digest = hmac.new(
        SECRET_KEY.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).digest()
    return base64.b64encode(digest).decode()


def build_query(params):
    return urlencode(params, doseq=True)


def auth_headers(timestamp, signature):
    return {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "ACCESS-TIMESTAMP": str(timestamp),
        "Content-Type": "application/json",
        "User-Agent": "V167.8.9-Pro-Bot/1.0",
    }


def authenticated_get(path, params=None):
    check_credentials()
    if params is None:
        params = {}
    timestamp = get_server_time_ms()
    query_string = build_query(params)
    signature = make_signature(
        timestamp=timestamp,
        method="GET",
        request_path=path,
        query_string=query_string,
        body=""
    )
    headers = auth_headers(timestamp, signature)
    r = SESSION.get(BASE_URL + path, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    if not r.ok:
        print()
        print("=" * 72)
        print("AUTHENTICATED REQUEST ERROR")
        print("=" * 72)
        print("STATUS :", r.status_code)
        print("PATH   :", path)
        print("QUERY  :", query_string)
        print("BODY   :", r.text[:2000])
        print("=" * 72)
    r.raise_for_status()
    return r.json()


def authenticated_post(path, body):
    check_credentials()
    body_text = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
    timestamp = get_server_time_ms()
    signature = make_signature(
        timestamp=timestamp,
        method="POST",
        request_path=path,
        query_string="",
        body=body_text
    )
    headers = auth_headers(timestamp, signature)
    r = SESSION.post(BASE_URL + path, data=body_text.encode("utf-8"), headers=headers, timeout=REQUEST_TIMEOUT)

    payload = None
    try:
        payload = r.json()
    except Exception:
        payload = None

    if not r.ok:
        print()
        print("=" * 72)
        print("DEMO ORDER REQUEST ERROR")
        print("=" * 72)
        print("STATUS :", r.status_code)
        print("PATH   :", path)
        print("BODY   :", body_text)
        print("RESPONSE:", r.text[:2000])
        print("=" * 72)

        code = payload.get("code") if isinstance(payload, dict) else None
        if code in DETERMINISTIC_REJECT_CODES:
            msg = payload.get("msg") if isinstance(payload, dict) else ""
            raise DeterministicOrderRejection(
                f"Exchange rejected order (code={code}): {msg}"
            )

    r.raise_for_status()
    return r.json()


def parse_klines(payload):
    if isinstance(payload, dict):
        raw = payload.get("data", [])
        if not raw:
            raw = payload.get("result", [])
    else:
        raw = payload
    if not isinstance(raw, list):
        return []
    rows = []
    for item in raw:
        try:
            if isinstance(item, (list, tuple)):
                if len(item) < 6:
                    continue
                ts = int(float(item[0]))
                if ts < 10_000_000_000:
                    ts *= 1000
                rows.append([
                    ts, float(item[1]), float(item[2]),
                    float(item[3]), float(item[4]), float(item[5]),
                ])
            elif isinstance(item, dict):
                ts = item.get("timestamp") or item.get("ts") or item.get("time")
                if ts is None:
                    continue
                ts = int(float(ts))
                if ts < 10_000_000_000:
                    ts *= 1000
                rows.append([
                    ts, float(item["open"]), float(item["high"]),
                    float(item["low"]), float(item["close"]),
                    float(item.get("volume", item.get("vol", 0))),
                ])
        except Exception:
            continue
    return rows


def request_history(start_ms, end_ms):
    params = {
        "symbol": MARKET_SYMBOL,
        "interval": TIMEFRAME,
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": HISTORY_LIMIT,
        "priceType": "LAST",
    }
    try:
        r = SESSION.get(BASE_URL + HISTORY_PATH, params=params, timeout=REQUEST_TIMEOUT)
        if not r.ok:
            print()
            print(f"REQUEST ERROR: HTTP {r.status_code}")
            print("URL:", r.url)
            print("SERVER RESPONSE:", r.text[:2000])
            return None
        return r.json()
    except Exception as e:
        print("REQUEST ERROR:", repr(e))
        return None


def rows_to_dataframe(rows):
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def save_runtime_state(last_candle=None, status="RUNNING"):
    state = {
        "status": status,
        "last_candle": str(last_candle) if last_candle is not None else None,
        "saved_at_utc": str(pd.Timestamp.now(tz="UTC")),
        "data_file": DATA_FILE,
        "demo_only": DEMO_ONLY,
        "operation_mode": OPERATION_MODE,
        "system_id": SYSTEM_ID,
        "version": "V167.8.9",
    }
    tmp = STATE_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        print("STATE SAVE ERROR:", repr(e))


def load_local_dataset():
    path = Path(DATA_FILE)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        return None if df.empty else df
    except Exception as e:
        print("LOCAL DATA LOAD ERROR:", repr(e))
        return None


def network_wait(reason="NETWORK"):
    print()
    print("=" * 72)
    print(f"{reason}: waiting for WEEX connection")
    print("AUTO-RECONNECT: ENABLED")
    print("=" * 72)
    attempt = 0
    while True:
        attempt += 1
        try:
            server = get_server_utc()
            print(f"[RECONNECTED] attempt={attempt} server={server}")
            return server
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[RECONNECT] attempt={attempt} error={repr(e)}")
            time.sleep(NETWORK_RETRY_SECONDS)


def safe_get_server_utc():
    while True:
        try:
            return get_server_utc()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[NETWORK DOWN] {repr(e)}")
            network_wait("NETWORK RECOVERY")


def sync_from_local_dataset(df):
    last_local = pd.Timestamp(df["timestamp"].iloc[-1])
    if last_local.tzinfo is None:
        last_local = last_local.tz_localize("UTC")
    else:
        last_local = last_local.tz_convert("UTC")
    print()
    print("LOCAL DATA FOUND")
    print("LOCAL LAST CLOSED:", last_local)
    print("CATCH-UP: CHECKING NEW CLOSED 15M CANDLES")
    empty_attempts = 0
    while True:
        try:
            new_df, newest_closed = fetch_new_closed_candles(last_local)
            if new_df is None or new_df.empty:
                if last_local >= newest_closed:
                    print("CATCH-UP: COMPLETE")
                    return df
                empty_attempts += 1
                next_needed = last_local + pd.Timedelta(minutes=CANDLE_MINUTES)
                print(f"[CATCH-UP RETRY] expected={newest_closed} next_needed={next_needed} attempt={empty_attempts}/5")
                if empty_attempts >= 5:
                    network_wait("CATCH-UP API RECOVERY")
                    empty_attempts = 0
                else:
                    time.sleep(RETRY_SECONDS)
                continue
            empty_attempts = 0
            old_last = last_local
            df = append_new_candles(df, new_df)
            last_local = pd.Timestamp(df["timestamp"].iloc[-1])
            print(f"[CATCH-UP] +{len(new_df)} candle(s) {old_last} -> {last_local}")
            if last_local >= newest_closed:
                print("CATCH-UP: COMPLETE")
                return df
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print("CATCH-UP ERROR:", repr(e))
            network_wait("CATCH-UP RECOVERY")


def fetch_initial_history():
    print()
    print("=" * 72)
    print("INITIAL WEEX 15M BACKFILL")
    print("=" * 72)
    server_now = get_server_utc()
    current_15m = server_now.floor("15min")
    last_closed = current_15m - pd.Timedelta(minutes=CANDLE_MINUTES)
    first_needed = last_closed - pd.Timedelta(days=LOOKBACK_DAYS)
    print("WEEX SERVER UTC:", server_now)
    print("REQUEST FROM   :", first_needed)
    print("REQUEST TO     :", last_closed)
    batch_span = pd.Timedelta(minutes=CANDLE_MINUTES * (HISTORY_LIMIT - 1))
    cursor = first_needed
    all_rows = []
    request_no = 0
    failures = 0
    while cursor <= last_closed:
        window_end = min(cursor + batch_span, last_closed)
        start_ms = int(cursor.timestamp() * 1000)
        end_ms = int((window_end + pd.Timedelta(minutes=CANDLE_MINUTES) - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
        server_now_ms = get_server_time_ms()
        end_ms = min(end_ms, server_now_ms - 1000)
        request_no += 1
        payload = request_history(start_ms, end_ms)
        if payload is None:
            failures += 1
            print(f"REQUEST={request_no:03d} FAILED RETRY={failures}/5")
            if failures >= 5:
                raise RuntimeError("Initial WEEX history failed repeatedly.")
            time.sleep(RETRY_SECONDS)
            request_no -= 1
            continue
        failures = 0
        rows = parse_klines(payload)
        filtered = []
        for row in rows:
            if row[0] >= start_ms and row[0] <= end_ms:
                filtered.append(row)
        rows = filtered
        all_rows.extend(rows)
        if rows:
            oldest = pd.to_datetime(min(r[0] for r in rows), unit="ms", utc=True)
            newest = pd.to_datetime(max(r[0] for r in rows), unit="ms", utc=True)
            print(f"REQUEST={request_no:03d} BATCH={len(rows):3d} OLDEST={oldest} NEWEST={newest}")
        else:
            print(f"REQUEST={request_no:03d} BATCH=0")
        cursor = window_end + pd.Timedelta(minutes=CANDLE_MINUTES)
        time.sleep(0.15)
    df = rows_to_dataframe(all_rows)
    df = df[(df["timestamp"] >= first_needed) & (df["timestamp"] <= last_closed)].copy()
    expected = pd.date_range(first_needed, last_closed, freq="15min", tz="UTC")
    missing = expected.difference(pd.DatetimeIndex(df["timestamp"]))
    print()
    print("=" * 72)
    print("15M DATASET")
    print("=" * 72)
    print("ROWS          :", len(df))
    print("FIRST         :", df["timestamp"].iloc[0])
    print("LAST CLOSED   :", df["timestamp"].iloc[-1])
    print("EXPECTED      :", len(expected))
    print("MISSING 15M   :", len(missing))
    print("=" * 72)
    if len(missing):
        print("FIRST MISSING:", list(missing[:10]))
        raise RuntimeError("Initial dataset contains missing 15M candles.")
    df.to_csv(DATA_FILE, index=False)
    print("SAVED         :", DATA_FILE)
    return df


def fetch_new_closed_candles(last_timestamp):
    server_now = get_server_utc()
    current_15m = server_now.floor("15min")
    newest_closed = current_15m - pd.Timedelta(minutes=CANDLE_MINUTES)
    last_timestamp = pd.Timestamp(last_timestamp)
    if last_timestamp.tzinfo is None:
        last_timestamp = last_timestamp.tz_localize("UTC")
    else:
        last_timestamp = last_timestamp.tz_convert("UTC")
    next_needed = last_timestamp + pd.Timedelta(minutes=CANDLE_MINUTES)
    if next_needed > newest_closed:
        return None, newest_closed
    start_ms = int(next_needed.timestamp() * 1000)
    end_ms = int((newest_closed + pd.Timedelta(minutes=CANDLE_MINUTES) - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
    end_ms = min(end_ms, get_server_time_ms() - 1000)
    payload = request_history(start_ms, end_ms)
    if payload is None:
        return None, newest_closed
    rows = parse_klines(payload)
    filtered = []
    for row in rows:
        ts = pd.to_datetime(row[0], unit="ms", utc=True)
        if ts >= next_needed and ts <= newest_closed:
            filtered.append(row)
    new_df = rows_to_dataframe(filtered)
    return new_df, newest_closed


def append_new_candles(df, new_df):
    if new_df is None or new_df.empty:
        return df
    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    cutoff = combined["timestamp"].iloc[-1] - pd.Timedelta(days=LOOKBACK_DAYS)
    combined = combined[combined["timestamp"] >= cutoff].copy()
    combined.to_csv(DATA_FILE, index=False)
    return combined


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def feature_engine(x):
    x = x.copy()
    x["e20"] = ema(x["close"], 20)
    x["e50"] = ema(x["close"], 50)
    x["e200"] = ema(x["close"], 200)
    x["rsi"] = rsi(x["close"], 14)
    tr1 = x["high"] - x["low"]
    tr2 = (x["high"] - x["close"].shift(1)).abs()
    tr3 = (x["low"] - x["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    x["atrp"] = x["atr"] / x["close"]
    x["vr"] = x["volume"] / x["volume"].rolling(20).mean()
    x["eg"] = x["close"] / x["close"].shift(20) - 1
    return x


def build_features(df):
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x = x.set_index("timestamp")
    h1 = x.resample("1h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    h4 = x.resample("4h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    h1 = feature_engine(h1)
    h4 = feature_engine(h4)
    current_15m = x.index[-1]
    h1 = h1[h1.index + pd.Timedelta(hours=1) <= current_15m]
    h4 = h4[h4.index + pd.Timedelta(hours=4) <= current_15m]
    if h1.empty or h4.empty:
        return None
    h1_row = h1.iloc[-1]
    h4_row = h4.iloc[-1]
    p = x.iloc[-1]
    values = [
        h1_row["e200"], h1_row["rsi"], h1_row["vr"],
        h4_row["e200"], h4_row["rsi"], h4_row["vr"],
        h4_row["atrp"], h4_row["eg"],
    ]
    if any(pd.isna(v) for v in values):
        return None
    return {
        "candle": x.index[-1],
        "close": float(p["close"]),
        "H1_time": h1.index[-1],
        "H1_close": float(h1_row["close"]),
        "H1_e20": float(h1_row["e20"]),
        "H1_e50": float(h1_row["e50"]),
        "H1_e200": float(h1_row["e200"]),
        "H1_rsi": float(h1_row["rsi"]),
        "H1_vr": float(h1_row["vr"]),
        "H4_time": h4.index[-1],
        "H4_close": float(h4_row["close"]),
        "H4_e20": float(h4_row["e20"]),
        "H4_e50": float(h4_row["e50"]),
        "H4_e200": float(h4_row["e200"]),
        "H4_rsi": float(h4_row["rsi"]),
        "H4_vr": float(h4_row["vr"]),
        "H4_atrp": float(h4_row["atrp"]),
        "H4_eg": float(h4_row["eg"]),
    }


def get_decayed_params():
    if not DECAY_ENABLED:
        return {
            "pa_lookback": PA_LOOKBACK,
            "d_min_ema_atr": D_MIN_EMA_ATR,
            "d_min_body_atr": D_MIN_BODY_ATR,
        }
    state_file = Path(DECAY_STATE_FILE)
    today = pd.Timestamp.now(tz="UTC").date().isoformat()
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            if state.get("date") == today:
                return state["params"]
        except Exception:
            pass
    day_seed = int(pd.Timestamp.now(tz="UTC").timestamp()) // 86400
    rng = random.Random(day_seed)
    params = {
        "pa_lookback": max(5, int(PA_LOOKBACK * rng.uniform(0.93, 1.07))),
        "d_min_ema_atr": max(0.10, D_MIN_EMA_ATR * rng.uniform(0.90, 1.10)),
        "d_min_body_atr": max(0.10, D_MIN_BODY_ATR * rng.uniform(0.90, 1.10)),
    }
    with open(state_file, "w") as f:
        json.dump({"date": today, "params": params}, f, indent=2)
    print(f"🔀 V167.8.9 DECAY: new params = {params}")
    return params


# ============================================================
# SIGNAL ENGINE
# ============================================================
def _d_momentum_metrics(df):
    if df is None or len(df) < 15:
        return None
    x = df.copy()
    for col in ("open", "high", "low", "close"):
        x[col] = x[col].astype(float)
    close = x["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    current_close = float(close.iloc[-1])
    previous_close = float(close.iloc[-2])
    current_open = float(x["open"].iloc[-1])
    body = abs(current_close - current_open)
    ema_distance = (current_close - ema20) / atr
    return {
        "current_close": current_close,
        "previous_close": previous_close,
        "ema20": ema20,
        "atr": atr,
        "body": body,
        "ema_distance": ema_distance,
        "ema_distance_abs": abs(ema_distance),
        "body_atr": body / atr,
    }


def test_d_signal(row, df=None):
    decayed = get_decayed_params()
    d_min_ema_atr = decayed.get("d_min_ema_atr", D_MIN_EMA_ATR)
    d_min_body_atr = decayed.get("d_min_body_atr", D_MIN_BODY_ATR)

    h4_long = (
        row["H4_close"] > row["H4_e200"]
        and row["H4_e20"] > row["H4_e50"]
        and SIG_H4_RSI_LONG_RANGE[0] <= row["H4_rsi"] <= SIG_H4_RSI_LONG_RANGE[1]
        and row["H4_vr"] >= SIG_H4_VR_MIN
        and row["H4_atrp"] >= SIG_H4_ATRP_MIN
        and row["H4_eg"] >= SIG_H4_EG_LONG_MIN
    )
    h4_short = (
        row["H4_close"] < row["H4_e200"]
        and row["H4_e20"] < row["H4_e50"]
        and SIG_H4_RSI_SHORT_RANGE[0] <= row["H4_rsi"] <= SIG_H4_RSI_SHORT_RANGE[1]
        and row["H4_vr"] >= SIG_H4_VR_MIN
        and row["H4_atrp"] >= SIG_H4_ATRP_MIN
        and row["H4_eg"] <= SIG_H4_EG_SHORT_MAX
    )
    metrics = _d_momentum_metrics(df)
    if metrics is None:
        return "HOLD"
    current_close = metrics["current_close"]
    previous_close = metrics["previous_close"]
    ema_distance = metrics["ema_distance"]
    ema_distance_abs = metrics["ema_distance_abs"]
    long_trigger = (
        current_close > previous_close
        and current_close > metrics["ema20"]
        and ema_distance > 0
        and ema_distance_abs >= d_min_ema_atr
        and metrics["body_atr"] >= d_min_body_atr
    )
    short_trigger = (
        current_close < previous_close
        and current_close < metrics["ema20"]
        and ema_distance < 0
        and ema_distance_abs >= d_min_ema_atr
        and metrics["body_atr"] >= d_min_body_atr
    )
    if long_trigger and not short_trigger:
        return "LONG"
    if short_trigger and not long_trigger:
        return "SHORT"
    return "HOLD"


def is_doji(df, threshold=0.10):
    if df is None or len(df) < 1:
        return False
    last = df.iloc[-1]
    body = abs(float(last["close"]) - float(last["open"]))
    high_low = float(last["high"]) - float(last["low"])
    if high_low <= 0:
        return False
    return body / high_low < threshold


def is_hammer(df):
    if df is None or len(df) < 1:
        return False
    last = df.iloc[-1]
    open_ = float(last["open"])
    close = float(last["close"])
    high = float(last["high"])
    low = float(last["low"])
    body = abs(close - open_)
    high_low = high - low
    if high_low <= 0 or body <= 0:
        return False
    lower_shadow = min(open_, close) - low
    upper_shadow = high - max(open_, close)
    return (lower_shadow >= 2 * body and upper_shadow < 0.3 * body and body / high_low < 0.4)


def is_shooting_star(df):
    if df is None or len(df) < 1:
        return False
    last = df.iloc[-1]
    open_ = float(last["open"])
    close = float(last["close"])
    high = float(last["high"])
    low = float(last["low"])
    body = abs(close - open_)
    high_low = high - low
    if high_low <= 0 or body <= 0:
        return False
    lower_shadow = min(open_, close) - low
    upper_shadow = high - max(open_, close)
    return (upper_shadow >= 2 * body and lower_shadow < 0.3 * body and body / high_low < 0.4)


def is_bullish_pin_bar(df):
    if df is None or len(df) < 1:
        return False
    last = df.iloc[-1]
    open_ = float(last["open"])
    close = float(last["close"])
    high = float(last["high"])
    low = float(last["low"])
    body = abs(close - open_)
    high_low = high - low
    if high_low <= 0 or body <= 0:
        return False
    lower_shadow = min(open_, close) - low
    upper_shadow = high - max(open_, close)
    return lower_shadow >= 2 * body and upper_shadow < 0.5 * body


def is_bearish_pin_bar(df):
    if df is None or len(df) < 1:
        return False
    last = df.iloc[-1]
    open_ = float(last["open"])
    close = float(last["close"])
    high = float(last["high"])
    low = float(last["low"])
    body = abs(close - open_)
    high_low = high - low
    if high_low <= 0 or body <= 0:
        return False
    upper_shadow = high - max(open_, close)
    lower_shadow = min(open_, close) - low
    return upper_shadow >= 2 * body and lower_shadow < 0.5 * body


def is_engulfing(df):
    if df is None or len(df) < 2:
        return None
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    curr_open = float(curr["open"])
    curr_close = float(curr["close"])
    prev_open = float(prev["open"])
    prev_close = float(prev["close"])
    bullish = (curr_close > curr_open and prev_close < prev_open
               and curr_open < prev_close and curr_close > prev_open)
    bearish = (curr_close < curr_open and prev_close > prev_open
               and curr_open > prev_close and curr_close < prev_open)
    if bullish:
        return "BULLISH"
    elif bearish:
        return "BEARISH"
    return None


def is_morning_star(df):
    if df is None or len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    bearish = float(c1["close"]) < float(c1["open"])
    body1 = abs(float(c1["close"]) - float(c1["open"]))
    range1 = float(c1["high"]) - float(c1["low"])
    body2 = abs(float(c2["close"]) - float(c2["open"]))
    range2 = float(c2["high"]) - float(c2["low"])
    bullish = float(c3["close"]) > float(c3["open"])
    body3 = abs(float(c3["close"]) - float(c3["open"]))
    range3 = float(c3["high"]) - float(c3["low"])
    if range1 <= 0 or range2 <= 0 or range3 <= 0:
        return False
    return (bearish and body1 / range1 > 0.5 and body2 / range2 < 0.3
            and bullish and body3 / range3 > 0.5
            and float(c3["close"]) > (float(c1["open"]) + float(c1["close"])) / 2)


def is_evening_star(df):
    if df is None or len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    bullish = float(c1["close"]) > float(c1["open"])
    body1 = abs(float(c1["close"]) - float(c1["open"]))
    range1 = float(c1["high"]) - float(c1["low"])
    body2 = abs(float(c2["close"]) - float(c2["open"]))
    range2 = float(c2["high"]) - float(c2["low"])
    bearish = float(c3["close"]) < float(c3["open"])
    body3 = abs(float(c3["close"]) - float(c3["open"]))
    range3 = float(c3["high"]) - float(c3["low"])
    if range1 <= 0 or range2 <= 0 or range3 <= 0:
        return False
    return (bullish and body1 / range1 > 0.5 and body2 / range2 < 0.3
            and bearish and body3 / range3 > 0.5
            and float(c3["close"]) < (float(c1["open"]) + float(c1["close"])) / 2)


def get_candle_patterns(df):
    if df is None or len(df) < 3:
        return {"doji": False, "hammer": False, "shooting_star": False,
                "bullish_pin": False, "bearish_pin": False, "engulfing": None,
                "morning_star": False, "evening_star": False,
                "signal": "HOLD", "detected": []}
    result = {
        "doji": is_doji(df),
        "hammer": is_hammer(df),
        "shooting_star": is_shooting_star(df),
        "bullish_pin": is_bullish_pin_bar(df),
        "bearish_pin": is_bearish_pin_bar(df),
        "engulfing": is_engulfing(df),
        "morning_star": is_morning_star(df),
        "evening_star": is_evening_star(df),
        "signal": "HOLD",
        "detected": []
    }
    if result["doji"]: result["detected"].append("DOJI")
    if result["hammer"]: result["detected"].append("HAMMER")
    if result["shooting_star"]: result["detected"].append("SHOOTING_STAR")
    if result["bullish_pin"]: result["detected"].append("BULLISH_PIN")
    if result["bearish_pin"]: result["detected"].append("BEARISH_PIN")
    if result["engulfing"]: result["detected"].append(f"ENGULFING_{result['engulfing']}")
    if result["morning_star"]: result["detected"].append("MORNING_STAR")
    if result["evening_star"]: result["detected"].append("EVENING_STAR")
    if (result["hammer"] or result["bullish_pin"] or result["morning_star"]
            or result["engulfing"] == "BULLISH"):
        result["signal"] = "LONG"
    elif (result["shooting_star"] or result["bearish_pin"] or result["evening_star"]
          or result["engulfing"] == "BEARISH"):
        result["signal"] = "SHORT"
    return result


def _candle_geometry(df):
    if df is None or len(df) < 20:
        return None
    x = df.copy()
    for c in ("open", "high", "low", "close"):
        x[c] = x[c].astype(float)
    prev_close = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    last = x.iloc[-1]
    rng = float(last["high"] - last["low"])
    if rng <= 0:
        return None
    body = abs(float(last["close"] - last["open"]))
    close_pos = (float(last["close"]) - float(last["low"])) / rng
    return x, float(atr.iloc[-1]), body, close_pos


def price_action_signal(df):
    g = _candle_geometry(df)
    if g is None:
        return "HOLD"
    x, atr, body, close_pos = g
    if not np.isfinite(atr) or atr <= 0:
        return "HOLD"
    last = x.iloc[-1]
    prev = x.iloc[-2]
    body_atr = body / atr
    bullish = (
        float(last["close"]) > float(last["open"])
        and body_atr >= 0.45
        and close_pos >= 0.65
        and float(last["close"]) > float(prev["high"])
    )
    bearish = (
        float(last["close"]) < float(last["open"])
        and body_atr >= 0.45
        and close_pos <= 0.35
        and float(last["close"]) < float(prev["low"])
    )
    if bullish and not bearish:
        return "LONG"
    if bearish and not bullish:
        return "SHORT"
    return "HOLD"


def entry_confluence(df):
    pa = price_action_signal(df)
    patterns = get_candle_patterns(df)
    pattern_signal = patterns["signal"]
    detected = patterns["detected"]
    if detected:
        print(f"🕯️ PATTERNS DETECTED: {', '.join(detected)}")
    if pa == "LONG" or pattern_signal == "LONG":
        return {"pa": pa, "patterns": detected, "signal": "LONG"}
    elif pa == "SHORT" or pattern_signal == "SHORT":
        return {"pa": pa, "patterns": detected, "signal": "SHORT"}
    else:
        return {"pa": pa, "patterns": detected, "signal": "HOLD"}


def is_volume_spike(df):
    if not VOLUME_SPIKE_ENABLED or df is None or len(df) < 20:
        return False
    current_volume = float(df["volume"].iloc[-1])
    avg_volume = float(df["volume"].iloc[-20:].mean())
    if avg_volume <= 0:
        return False
    return current_volume > avg_volume * VOLUME_SPIKE_THRESHOLD


def get_h4_status(row):
    if row["H4_close"] > row["H4_e200"] and row["H4_e20"] > row["H4_e50"]:
        structure = "LONG"
    elif row["H4_close"] < row["H4_e200"] and row["H4_e20"] < row["H4_e50"]:
        structure = "SHORT"
    else:
        structure = "NEUTRAL"
    if 55 <= row["H4_rsi"] <= 70 and row["H4_eg"] >= 0.02:
        momentum = "STRONG"
    elif row["H4_rsi"] < 50 and row["H4_eg"] < 0.01:
        momentum = "WEAK"
    else:
        momentum = "NEUTRAL"
    if row["H4_eg"] >= 0.02:
        edge_direction, edge_strength = "LONG", "HIGH"
    elif row["H4_eg"] <= -0.02:
        edge_direction, edge_strength = "SHORT", "HIGH"
    elif row["H4_eg"] > 0.01:
        edge_direction, edge_strength = "LONG", "MEDIUM"
    elif row["H4_eg"] < -0.01:
        edge_direction, edge_strength = "SHORT", "MEDIUM"
    else:
        edge_direction, edge_strength = "NEUTRAL", "LOW"
    return structure, momentum, edge_direction, edge_strength


def get_market_regime(row):
    structure, momentum, edge_direction, edge_strength = get_h4_status(row)
    h1_short_bias = sum([
        row["H1_close"] < row["H1_e200"],
        row["H1_e20"] < row["H1_e50"],
        row["H1_rsi"] <= SIG_H1_RSI_SHORT_MAX,
    ]) >= 2
    h1_long_bias = sum([
        row["H1_close"] > row["H1_e200"],
        row["H1_e20"] > row["H1_e50"],
        row["H1_rsi"] >= SIG_H1_RSI_LONG_MIN,
    ]) >= 2
    if structure == "LONG" and edge_direction == "SHORT" and h1_short_bias:
        return "TRANSITION (LONG→SHORT)", "TRANSITION", "SHORT"
    elif structure == "SHORT" and edge_direction == "LONG" and h1_long_bias:
        return "TRANSITION (SHORT→LONG)", "TRANSITION", "LONG"
    elif structure == "NEUTRAL":
        return "CONSOLIDATION", "CONSOLIDATION", "NEUTRAL"
    elif structure == "LONG" and edge_direction == "LONG":
        return "BULLISH CONFIRMED", "CONFIRMED", "LONG"
    elif structure == "SHORT" and edge_direction == "SHORT":
        return "BEARISH CONFIRMED", "CONFIRMED", "SHORT"
    else:
        return "UNCLEAR", "UNCLEAR", "NEUTRAL"


def get_price_move_pct(df):
    if df is None or len(df) < 2:
        return 0
    d_metrics = _d_momentum_metrics(df)
    if d_metrics is None:
        return 0
    prev_close = d_metrics["previous_close"]
    current_close = d_metrics["current_close"]
    if prev_close == 0:
        return 0
    return abs(current_close - prev_close) / prev_close * 100


def get_15m_signal(row, df=None):
    pa = price_action_signal(df)
    test_d = test_d_signal(row, df)

    if pa == "LONG" or test_d == "LONG":
        return "LONG", pa, test_d
    elif pa == "SHORT" or test_d == "SHORT":
        return "SHORT", pa, test_d

    price_move_pct = get_price_move_pct(df)
    if price_move_pct >= PRICE_MOVE_THRESHOLD:
        d_metrics = _d_momentum_metrics(df)
        if d_metrics is not None:
            if d_metrics["current_close"] > d_metrics["previous_close"]:
                return "LONG", pa, test_d
            else:
                return "SHORT", pa, test_d

    return "HOLD", pa, test_d


def calculate_bidirectional_scores(row, df=None):
    threshold = ENTRY_THRESHOLD
    min_gap = MIN_EDGE_GAP
    min_conf = MIN_CONFIDENCE

    h4_bullish = row["H4_close"] > row["H4_e200"] and row["H4_e20"] > row["H4_e50"]
    h4_bearish = row["H4_close"] < row["H4_e200"] and row["H4_e20"] < row["H4_e50"]

    if h4_bullish:
        long_struct, short_struct = 2, 0
    elif h4_bearish:
        long_struct, short_struct = 0, 2
    else:
        long_struct, short_struct = 0, 0

    if row["H4_eg"] >= 0.02:
        long_edge, short_edge = 2, 0
    elif row["H4_eg"] <= -0.02:
        long_edge, short_edge = 0, 2
    elif row["H4_eg"] > 0.01:
        long_edge, short_edge = 1, 0
    elif row["H4_eg"] < -0.01:
        long_edge, short_edge = 0, 1
    else:
        long_edge, short_edge = 0, 0

    if 55 <= row["H4_rsi"] <= 70 and row["H4_eg"] >= 0.02:
        long_momentum, short_momentum = 1, 0
    elif row["H4_rsi"] < 50 and row["H4_eg"] < 0.01:
        long_momentum, short_momentum = 0, 1
    else:
        long_momentum, short_momentum = 0, 0

    regime, regime_type, regime_dir = get_market_regime(row)
    if regime_type == "CONFIRMED":
        if regime_dir == "LONG":
            long_regime, short_regime = 1, 0
        elif regime_dir == "SHORT":
            long_regime, short_regime = 0, 1
        else:
            long_regime, short_regime = 0, 0
    elif regime_type == "TRANSITION":
        long_regime, short_regime = -1, -1
    else:
        long_regime, short_regime = 0, 0

    long_h1_checks = [
        row["H1_close"] > row["H1_e200"],
        row["H1_e20"] > row["H1_e50"],
        row["H1_rsi"] >= SIG_H1_RSI_LONG_MIN,
    ]
    short_h1_checks = [
        row["H1_close"] < row["H1_e200"],
        row["H1_e20"] < row["H1_e50"],
        row["H1_rsi"] <= SIG_H1_RSI_SHORT_MAX,
    ]
    long_h1_raw = sum(long_h1_checks)
    short_h1_raw = sum(short_h1_checks)
    long_h1 = 2 if long_h1_raw >= 3 else 1 if long_h1_raw >= 2 else -1 if long_h1_raw >= 1 else -2
    short_h1 = 2 if short_h1_raw >= 3 else 1 if short_h1_raw >= 2 else -1 if short_h1_raw >= 1 else -2

    signal_15m, pa, test_d = get_15m_signal(row, df)
    if signal_15m == "LONG":
        long_15m, short_15m = 2, 0
    elif signal_15m == "SHORT":
        long_15m, short_15m = 0, 2
    else:
        long_15m, short_15m = 0, 0

    vol_score = 1 if row["H4_vr"] >= 1.3 else 0
    long_volume = vol_score
    short_volume = vol_score

    long_confluence_bonus = 0
    short_confluence_bonus = 0
    confluence_info = {"signal": "HOLD", "pa": pa, "patterns": []}
    if CONFLUENCE_SCORING_ENABLED and df is not None:
        try:
            c = entry_confluence(df)
            confluence_info = c
            if c["signal"] == "LONG":
                long_confluence_bonus = min(CONFLUENCE_MAX_BONUS, 1)
            elif c["signal"] == "SHORT":
                short_confluence_bonus = min(CONFLUENCE_MAX_BONUS, 1)
        except Exception as e:
            print("CONFLUENCE ERROR (ignored):", repr(e))

    long_score = (
        long_struct + long_edge + long_momentum + long_regime
        + long_h1 + long_15m + long_volume + long_confluence_bonus
    )
    short_score = (
        short_struct + short_edge + short_momentum + short_regime
        + short_h1 + short_15m + short_volume + short_confluence_bonus
    )

    if long_score > short_score:
        direction = "LONG"
        evidence = [
            h4_bullish,
            row["H4_eg"] >= 0.02,
            55 <= row["H4_rsi"] <= 70 and row["H4_eg"] >= 0.02,
            long_h1_raw >= 2,
            pa == "LONG",
            test_d == "LONG",
            regime_type == "CONFIRMED" and regime_dir == "LONG",
        ]
    elif short_score > long_score:
        direction = "SHORT"
        evidence = [
            h4_bearish,
            row["H4_eg"] <= -0.02,
            row["H4_rsi"] < 50 and row["H4_eg"] < 0.01,
            short_h1_raw >= 2,
            pa == "SHORT",
            test_d == "SHORT",
            regime_type == "CONFIRMED" and regime_dir == "SHORT",
        ]
    else:
        direction = "NEUTRAL"
        evidence = [False] * 7

    total_evidence = len(evidence)
    aligned = sum(evidence)
    confidence = (aligned / total_evidence) * 100 if total_evidence > 0 else 0
    edge_gap = abs(long_score - short_score)
    trigger = signal_15m if signal_15m != "HOLD" else None
    has_trigger = trigger is not None

    price_move_pct = get_price_move_pct(df)

    return {
        "long": {
            "structure": long_struct, "edge": long_edge, "momentum": long_momentum,
            "regime": long_regime, "h1": long_h1, "h1_raw": f"{long_h1_raw}/3",
            "signal_15m": long_15m, "volume": long_volume,
            "confluence_bonus": long_confluence_bonus, "total": long_score,
        },
        "short": {
            "structure": short_struct, "edge": short_edge, "momentum": short_momentum,
            "regime": short_regime, "h1": short_h1, "h1_raw": f"{short_h1_raw}/3",
            "signal_15m": short_15m, "volume": short_volume,
            "confluence_bonus": short_confluence_bonus, "total": short_score,
        },
        "gap": edge_gap, "confidence": confidence, "trigger": trigger,
        "has_trigger": has_trigger, "threshold": threshold,
        "min_gap": min_gap, "min_conf": min_conf,
        "regime": regime, "regime_type": regime_type, "regime_dir": regime_dir,
        "pa": pa, "test_d": test_d, "signal_15m": signal_15m,
        "evidence": {"aligned": aligned, "total": total_evidence, "details": evidence},
        "direction": direction, "price_move_pct": price_move_pct,
        "is_reduced_size": False, "confluence": confluence_info,
    }


def final_signal(row, df=None):
    scores = calculate_bidirectional_scores(row, df)

    print()
    print("=" * 72)
    print("V167.8.9 ADAPTIVE ENTRY (EXIT REDESIGN)")
    print("=" * 72)
    print(f"  MODE            : {OPERATION_MODE}")
    print(f"  SYSTEM ID       : {SYSTEM_ID}")
    print("                    LONG    SHORT")
    print(f"  STRUCTURE        {scores['long']['structure']:+3d}      {scores['short']['structure']:+3d}")
    print(f"  EDGE             {scores['long']['edge']:+3d}      {scores['short']['edge']:+3d}")
    print(f"  MOMENTUM         {scores['long']['momentum']:+3d}      {scores['short']['momentum']:+3d}")
    print(f"  REGIME           {scores['long']['regime']:+3d}      {scores['short']['regime']:+3d}")
    print(f"  H1 ({scores['long']['h1_raw']})      {scores['long']['h1']:+3d}      {scores['short']['h1']:+3d}")
    print(f"  15M SIGNAL       {scores['long']['signal_15m']:+3d}      {scores['short']['signal_15m']:+3d}")
    print(f"  VOLUME           {scores['long']['volume']:+3d}      {scores['short']['volume']:+3d}")
    print(f"  CONFLUENCE       {scores['long']['confluence_bonus']:+3d}      {scores['short']['confluence_bonus']:+3d}")
    print("-" * 72)
    print(f"  TOTAL            {scores['long']['total']:+3d}      {scores['short']['total']:+3d}")
    print("-" * 72)
    print(f"  EDGE GAP         : {scores['gap']:+.0f} (min: {scores['min_gap']} FROZEN)")
    print(f"  CONFIDENCE       : {scores['confidence']:.0f}% (min: {scores['min_conf']}% FROZEN)")
    print(f"  PRICE MOVE       : {scores['price_move_pct']:.2f}%")
    print(f"  TRIGGER          : {scores['trigger'] or 'NONE'}")
    print(f"  REGIME           : {scores['regime']}")
    print(f"  DIRECTION        : {scores['direction']}")
    ev = scores['evidence']
    print(f"  EVIDENCE         : {ev['aligned']}/{ev['total']} aligned")
    c = scores['confluence']
    print(f"  CONFLUENCE SIGNAL: {c['signal']} (integrated as bounded bonus)")
    print("=" * 72)

    if not scores['has_trigger']:
        print()
        print("-" * 72)
        print("DECISION TRACE:")
        print(f"  DIRECTION       : {scores['direction']}")
        print(f"  SCORE           : {'PASS' if max(scores['long']['total'], scores['short']['total']) >= scores['threshold'] else 'FAIL'}")
        print(f"  EDGE GAP        : {'PASS' if scores['gap'] >= scores['min_gap'] else 'FAIL'}")
        print(f"  CONFIDENCE      : {'PASS' if scores['confidence'] >= scores['min_conf'] else 'FAIL'}")
        print(f"  PRICE MOVE      : {scores['price_move_pct']:.2f}%")
        print(f"  TRIGGER ALIGN   : FAIL (NONE)")
        print(f"  FINAL           : HOLD")
        print("-" * 72)
        return "HOLD"

    long_score = scores['long']['total']
    short_score = scores['short']['total']

    if long_score > short_score:
        dominant_direction = "LONG"
        dominant_score = long_score
    elif short_score > long_score:
        dominant_direction = "SHORT"
        dominant_score = short_score
    else:
        dominant_direction = "NEUTRAL"
        dominant_score = 0

    score_pass = dominant_score >= scores['threshold']
    gap_pass = scores['gap'] >= scores['min_gap']
    conf_pass = scores['confidence'] >= scores['min_conf']
    trigger_aligned = (scores['trigger'] == dominant_direction)
    price_move_pct = scores['price_move_pct']

    if not score_pass or not gap_pass or not trigger_aligned:
        decision = "HOLD"
        print()
        print("-" * 72)
        print("DECISION TRACE:")
        print(f"  DIRECTION       : {dominant_direction}")
        print(f"  SCORE           : {'PASS' if score_pass else 'FAIL'} ({dominant_score:.0f} >= {scores['threshold']} FROZEN)")
        print(f"  EDGE GAP        : {'PASS' if gap_pass else 'FAIL'} ({scores['gap']:.0f} >= {scores['min_gap']} FROZEN)")
        print(f"  CONFIDENCE      : {'PASS' if conf_pass else 'FAIL'} ({scores['confidence']:.0f}% >= {scores['min_conf']}% FROZEN)")
        print(f"  PRICE MOVE      : {price_move_pct:.2f}%")
        print(f"  TRIGGER ALIGN   : {'PASS' if trigger_aligned else 'FAIL'} ({scores['trigger']} vs {dominant_direction})")
        print(f"  FINAL           : {decision}")
        print("-" * 72)
        print(f"💤 DECISION: {decision} (Hard requirements failed)")
        return "HOLD"

    if conf_pass:
        decision = dominant_direction
        scores['is_reduced_size'] = False
        print()
        print("-" * 72)
        print("DECISION TRACE:")
        print(f"  DIRECTION       : {dominant_direction}")
        print(f"  SCORE           : PASS ({dominant_score:.0f} >= {scores['threshold']} FROZEN)")
        print(f"  EDGE GAP        : PASS ({scores['gap']:.0f} >= {scores['min_gap']} FROZEN)")
        print(f"  CONFIDENCE      : PASS ({scores['confidence']:.0f}% >= {scores['min_conf']}% FROZEN)")
        print(f"  PRICE MOVE      : {price_move_pct:.2f}%")
        print(f"  TRIGGER ALIGN   : PASS ({scores['trigger']} vs {dominant_direction})")
        print(f"  SIZE            : FULL (100%)")
        print(f"  FINAL           : {decision}")
        print("-" * 72)
        print(f"✅ DECISION: {decision} (FULL SIZE)")
        return decision

    elif price_move_pct >= REDUCED_SIZE_THRESHOLD:
        decision = dominant_direction
        scores['is_reduced_size'] = True
        print()
        print("-" * 72)
        print("DECISION TRACE:")
        print(f"  DIRECTION       : {dominant_direction}")
        print(f"  SCORE           : PASS ({dominant_score:.0f} >= {scores['threshold']} FROZEN)")
        print(f"  EDGE GAP        : PASS ({scores['gap']:.0f} >= {scores['min_gap']} FROZEN)")
        print(f"  CONFIDENCE      : FAIL ({scores['confidence']:.0f}% < {scores['min_conf']}% FROZEN)")
        print(f"  PRICE MOVE      : {price_move_pct:.2f}% (>= {REDUCED_SIZE_THRESHOLD}%)")
        print(f"  TRIGGER ALIGN   : PASS ({scores['trigger']} vs {dominant_direction})")
        print(f"  SIZE            : REDUCED (50%)")
        print(f"  FINAL           : {decision}")
        print("-" * 72)
        print(f"🟡 DECISION: {decision} (REDUCED SIZE 50%)")
        return decision

    else:
        decision = "HOLD"
        print()
        print("-" * 72)
        print("DECISION TRACE:")
        print(f"  DIRECTION       : {dominant_direction}")
        print(f"  SCORE           : PASS ({dominant_score:.0f} >= {scores['threshold']} FROZEN)")
        print(f"  EDGE GAP        : PASS ({scores['gap']:.0f} >= {scores['min_gap']} FROZEN)")
        print(f"  CONFIDENCE      : FAIL ({scores['confidence']:.0f}% < {scores['min_conf']}% FROZEN)")
        print(f"  PRICE MOVE      : {price_move_pct:.2f}% (< {REDUCED_SIZE_THRESHOLD}%)")
        print(f"  TRIGGER ALIGN   : PASS ({scores['trigger']} vs {dominant_direction})")
        print(f"  SIZE            : NONE")
        print(f"  FINAL           : {decision}")
        print("-" * 72)
        print(f"💤 DECISION: HOLD (Price move insufficient for reduced size)")
        return "HOLD"


def get_h1_bias(row):
    h1_short_bias = sum([
        row["H1_close"] < row["H1_e200"],
        row["H1_e20"] < row["H1_e50"],
        row["H1_rsi"] <= SIG_H1_RSI_SHORT_MAX,
    ]) >= 2
    h1_long_bias = sum([
        row["H1_close"] > row["H1_e200"],
        row["H1_e20"] > row["H1_e50"],
        row["H1_rsi"] >= SIG_H1_RSI_LONG_MIN,
    ]) >= 2
    if h1_long_bias:
        return "LONG"
    elif h1_short_bias:
        return "SHORT"
    else:
        return "NEUTRAL"


def print_regime_summary(row, signal, df=None):
    h4_structure, h4_momentum, h4_edge_dir, h4_edge_str = get_h4_status(row)
    regime, regime_type, regime_dir = get_market_regime(row)
    h1_bias = get_h1_bias(row)
    scores = calculate_bidirectional_scores(row, df)
    summary = (
        f"📊 REGIME: {regime} | "
        f"H4: {h4_structure}/{h4_edge_dir} | "
        f"H1: {h1_bias} | "
        f"15M: {scores['signal_15m']} | "
        f"LONG: {scores['long']['total']:+.0f} | "
        f"SHORT: {scores['short']['total']:+.0f} | "
        f"GAP: {scores['gap']:+.0f} | "
        f"CONF: {scores['confidence']:.0f}% | "
        f"MOVE: {scores['price_move_pct']:.2f}% | "
        f"TRIGGER: {scores['trigger'] or 'NONE'} | "
        f"FINAL: {signal}"
    )
    print(summary)
    return summary


def print_state(row, signal, df=None):
    scores = calculate_bidirectional_scores(row, df)
    print()
    print("=" * 72)
    print("V167.8.9 PROFESSIONAL BOT - EXIT REDESIGN")
    print("=" * 72)
    print(f"  MODE            : {OPERATION_MODE}")
    print(f"  SYSTEM ID       : {SYSTEM_ID}")
    print("15M CANDLE :", row["candle"])
    print(f"BTC        : {row['close']:.2f}")
    print()
    print("H1 TIME    :", row["H1_time"])
    print(f"H1 CLOSE   : {row['H1_close']:.2f}")
    print(f"H1 RSI     : {row['H1_rsi']:.4f}")
    print(f"H1 VR      : {row['H1_vr']:.6f}")
    print()
    print("H4 TIME    :", row["H4_time"])
    print(f"H4 CLOSE   : {row['H4_close']:.2f}")
    print(f"H4 RSI     : {row['H4_rsi']:.4f}")
    print(f"H4 VR      : {row['H4_vr']:.6f}")
    print(f"H4 ATRP    : {row['H4_atrp']:.6f}")
    print(f"H4 EG      : {row['H4_eg']:.6f}")
    print()

    h4_structure, h4_momentum, h4_edge_dir, h4_edge_str = get_h4_status(row)
    regime, regime_type, regime_dir = get_market_regime(row)

    print("HIERARCHICAL SYSTEM STATUS (DETAILED)")
    print(f"  H4 STRUCTURE      : {h4_structure} {'✅' if h4_structure != 'NEUTRAL' else '❌'}")
    print(f"  H4 MOMENTUM       : {h4_momentum}")
    print(f"  H4 EDGE DIRECTION : {h4_edge_dir}")
    print(f"  H4 EDGE STRENGTH  : {h4_edge_str}")
    print(f"  H1 CONFIRMER      : LONG: {scores['long']['h1_raw']} | SHORT: {scores['short']['h1_raw']}")
    print(f"  15M SIGNAL        : {scores['signal_15m']}")
    print(f"  PRICE MOVE        : {scores['price_move_pct']:.2f}%")
    print(f"  MARKET REGIME     : {regime}")
    print(f"  FINAL SIGNAL      : {signal}")
    print("-" * 72)

    print("BIDIRECTIONAL SCORES")
    print("                    LONG    SHORT")
    print(f"  STRUCTURE        {scores['long']['structure']:+3d}      {scores['short']['structure']:+3d}")
    print(f"  EDGE             {scores['long']['edge']:+3d}      {scores['short']['edge']:+3d}")
    print(f"  MOMENTUM         {scores['long']['momentum']:+3d}      {scores['short']['momentum']:+3d}")
    print(f"  REGIME           {scores['long']['regime']:+3d}      {scores['short']['regime']:+3d}")
    print(f"  H1 ({scores['long']['h1_raw']})      {scores['long']['h1']:+3d}      {scores['short']['h1']:+3d}")
    print(f"  15M SIGNAL       {scores['long']['signal_15m']:+3d}      {scores['short']['signal_15m']:+3d}")
    print(f"  VOLUME           {scores['long']['volume']:+3d}      {scores['short']['volume']:+3d}")
    print(f"  CONFLUENCE       {scores['long']['confluence_bonus']:+3d}      {scores['short']['confluence_bonus']:+3d}")
    print("-" * 72)
    print(f"  TOTAL            {scores['long']['total']:+3d}      {scores['short']['total']:+3d}")
    print("-" * 72)
    print(f"  EDGE GAP         : {scores['gap']:+.0f} (min: {scores['min_gap']} FROZEN)")
    print(f"  CONFIDENCE       : {scores['confidence']:.0f}% (min: {scores['min_conf']}% FROZEN)")
    print(f"  PRICE MOVE       : {scores['price_move_pct']:.2f}%")
    print(f"  TRIGGER          : {scores['trigger'] or 'NONE'}")
    print(f"  EVIDENCE         : {scores['evidence']['aligned']}/{scores['evidence']['total']} aligned")
    if scores['is_reduced_size']:
        print(f"  SIZE             : REDUCED (50%)")
    print("=" * 72)

    print_regime_summary(row, signal, df)
    print("-" * 72)

    print("V167.8.9 LONG DIAGNOSTIC")
    for name, ok in _long_checks(row):
        print(f"  [{'PASS ' if ok else 'BLOCK'}] {name}")
    print()
    print("V167.8.9 SHORT DIAGNOSTIC")
    for name, ok in _short_checks(row):
        print(f"  [{'PASS ' if ok else 'BLOCK'}] {name}")

    if df is not None and len(df) >= 2:
        d_metrics = _d_momentum_metrics(df)
        if d_metrics is not None:
            print()
            print("TEST-D 15M MOMENTUM QUALITY")
            print(f"  Previous 15M close : {d_metrics['previous_close']:.2f}")
            print(f"  Current 15M close  : {d_metrics['current_close']:.2f}")
            print(f"  15M EMA20          : {d_metrics['ema20']:.2f}")
            print(f"  ATR(14)            : {d_metrics['atr']:.2f}")
            print(f"  Body / ATR         : {d_metrics['body_atr']:.3f} (need >= {D_MIN_BODY_ATR:.2f})")
            print(f"  EMA distance       : {d_metrics['ema_distance']:.3f} ATR "
                  f"(abs: {d_metrics['ema_distance_abs']:.3f}, need >= {D_MIN_EMA_ATR:.2f})")
            print(f"  Position vs EMA20  : "
                  f"{'ABOVE' if d_metrics['ema_distance'] > 0 else 'BELOW' if d_metrics['ema_distance'] < 0 else 'AT'}")
    print("=" * 72)


def print_confluence_diagnostic(df, test_d):
    c = entry_confluence(df)
    print()
    print("V167.8.9 ENTRY CONFLUENCE (integrated as bounded bonus)")
    print(f"  PRICE ACTION   : {c['pa']}")
    print(f"  PATTERNS       : {', '.join(c['patterns']) if c['patterns'] else 'NONE'}")
    print(f"  CONFLUENCE     : {c['signal']}")
    print(f"  TEST-D         : {test_d}")
    print(f"  BONUS          : max +{CONFLUENCE_MAX_BONUS} to matching side")
    if test_d in {"LONG", "SHORT"}:
        vol_spike = is_volume_spike(df)
        print(f"  VOLUME SPIKE   : {'✅' if vol_spike else '❌'}")
    if c['patterns']:
        print(f"  🔍 ACTIVE PATTERNS: {', '.join(c['patterns'])}")
    return c


# ============================================================
# DIAGNOSTIC CHECKS
# ============================================================
def _long_checks(row):
    return [
        ("H4 trend: close > EMA200", row["H4_close"] > row["H4_e200"]),
        ("H4 trend: EMA20 > EMA50", row["H4_e20"] > row["H4_e50"]),
        (f"H4 RSI in [{SIG_H4_RSI_LONG_RANGE[0]}, {SIG_H4_RSI_LONG_RANGE[1]}]",
         SIG_H4_RSI_LONG_RANGE[0] <= row["H4_rsi"] <= SIG_H4_RSI_LONG_RANGE[1]),
        ("H1 trend: close > EMA200", row["H1_close"] > row["H1_e200"]),
        ("H1 trend: EMA20 > EMA50", row["H1_e20"] > row["H1_e50"]),
        (f"H4 VR >= {SIG_H4_VR_MIN}", row["H4_vr"] >= SIG_H4_VR_MIN),
        (f"H1 VR >= {SIG_H1_VR_MIN}", row["H1_vr"] >= SIG_H1_VR_MIN),
        (f"H4 ATRP >= {SIG_H4_ATRP_MIN}", row["H4_atrp"] >= SIG_H4_ATRP_MIN),
        (f"H4 EG >= {SIG_H4_EG_LONG_MIN}", row["H4_eg"] >= SIG_H4_EG_LONG_MIN),
        (f"H1 RSI >= {SIG_H1_RSI_LONG_MIN}", row["H1_rsi"] >= SIG_H1_RSI_LONG_MIN),
    ]


def _short_checks(row):
    return [
        ("H4 trend: close < EMA200", row["H4_close"] < row["H4_e200"]),
        ("H4 trend: EMA20 < EMA50", row["H4_e20"] < row["H4_e50"]),
        (f"H4 RSI in [{SIG_H4_RSI_SHORT_RANGE[0]}, {SIG_H4_RSI_SHORT_RANGE[1]}]",
         SIG_H4_RSI_SHORT_RANGE[0] <= row["H4_rsi"] <= SIG_H4_RSI_SHORT_RANGE[1]),
        ("H1 trend: close < EMA200", row["H1_close"] < row["H1_e200"]),
        ("H1 trend: EMA20 < EMA50", row["H1_e20"] < row["H1_e50"]),
        (f"H4 VR >= {SIG_H4_VR_MIN}", row["H4_vr"] >= SIG_H4_VR_MIN),
        (f"H1 VR >= {SIG_H1_VR_MIN}", row["H1_vr"] >= SIG_H1_VR_MIN),
        (f"H4 ATRP >= {SIG_H4_ATRP_MIN}", row["H4_atrp"] >= SIG_H4_ATRP_MIN),
        (f"H4 EG <= {SIG_H4_EG_SHORT_MAX}", row["H4_eg"] <= SIG_H4_EG_SHORT_MAX),
        (f"H1 RSI <= {SIG_H1_RSI_SHORT_MAX}", row["H1_rsi"] <= SIG_H1_RSI_SHORT_MAX),
    ]

# ============================================================
# DEMO BALANCE / POSITIONS / ORDERS
# ============================================================
def print_demo_balance():
    try:
        data = authenticated_get(BALANCE_PATH)
        print()
        print("-" * 72)
        print("WEEX DEMO ACCOUNT")
        print("-" * 72)
        if isinstance(data, list):
            for item in data:
                print(
                    f"{item.get('asset', '?'):8} "
                    f"BAL={item.get('balance', '?')} "
                    f"AVAILABLE={item.get('availableBalance', '?')} "
                    f"FROZEN={item.get('frozen', '?')} "
                    f"UPNL={item.get('unrealizePnl', '?')}"
                )
        else:
            print(data)
        print("-" * 72)
    except Exception as e:
        print("DEMO BALANCE ERROR:", repr(e))


_DEMO_POSITIONS_CACHE = {"ts": 0.0, "data": None}


def get_demo_positions(force_refresh=False):
    now = time.time()
    cached = _DEMO_POSITIONS_CACHE.get("data")
    if (not force_refresh and cached is not None
            and now - _DEMO_POSITIONS_CACHE.get("ts", 0.0) < POSITION_CACHE_TTL):
        return cached
    try:
        data = authenticated_get(POSITION_PATH)
        data = data if isinstance(data, list) else []
        _DEMO_POSITIONS_CACHE["data"] = data
        _DEMO_POSITIONS_CACHE["ts"] = time.time()
        return data
    except Exception as e:
        print("POSITION ERROR:", repr(e))
        return cached if isinstance(cached, list) else []


def _parse_order_rows(payload):
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("data", "result", "rows", "orders"):
            raw = payload.get(key)
            if isinstance(raw, list):
                return [x for x in raw if isinstance(x, dict)]
    return []


def get_recent_demo_orders(limit=ORDER_HISTORY_LOOKBACK_LIMIT):
    try:
        payload = authenticated_get(ORDER_HISTORY_PATH, {
            "symbol": DEMO_SYMBOL,
            "limit": int(max(1, min(100, limit))),
        })
        return _parse_order_rows(payload)
    except Exception as e:
        print("ORDER HISTORY FETCH ERROR:", repr(e))
        return []


def _order_time_utc(order):
    for key in ("time", "createTime", "createdTime", "updateTime"):
        raw = order.get(key)
        if raw is None:
            continue
        try:
            v = float(raw)
            if v < 10_000_000_000:
                v *= 1000
            ts = pd.Timestamp(v, unit="ms", tz="UTC")
            if ts.year >= 2020:
                return ts
        except Exception:
            pass
    return None


def _order_client_id(order):
    return str(order.get("clientOrderId") or order.get("newClientOrderId") or order.get("clientAlgoId") or "")


def _order_position_side(order):
    return str(order.get("positionSide") or "").upper()


def _order_fill_price(order):
    return _extract_filled_price(order)


def _order_executed_qty(order):
    for key in ("executedQty", "filledQty", "cumExecQty", "cumQty", "qty"):
        raw = order.get(key)
        if raw is not None:
            try:
                q = float(raw)
                if np.isfinite(q) and q >= 0:
                    return q
            except Exception:
                pass
    return 0.0


def _infer_order_reason(order, item):
    cid = _order_client_id(order).upper()
    for token, reason in (("-TP1-", "TP1"), ("-TP2-", "TP2"), ("-TRAIL-", "TRAIL"),
                          ("-BE-", "BE"), ("-TIMEEX", "TIMEEXIT"), ("-SL-", "SL")):
        if token in cid:
            return reason
    price = _order_fill_price(order)
    hard_sl = float((item or {}).get("hard_sl_price", 0) or 0)
    if price and hard_sl and float((item or {}).get("risk_distance", 0) or 0) > 0:
        if abs(price - hard_sl) <= float((item or {}).get("risk_distance", 0)) * EXTERNAL_CLOSE_PRICE_TOLERANCE_R:
            return "SL_EXTERNAL"
    return "EXTERNAL_CLOSE"


def _find_recent_close_order(item, side, opened_at=None):
    if not CLOSE_RECONCILIATION_ENABLED:
        return None
    orders = get_recent_demo_orders()
    opened = _normalize_utc_timestamp(opened_at) if opened_at else None
    candidates = []
    for order in orders:
        if str(order.get("symbol", "")).upper() != DEMO_SYMBOL.upper():
            continue
        if _order_position_side(order) not in {"", side.upper()}:
            continue
        status = str(order.get("status") or order.get("orderStatus") or "").upper()
        if status not in {"FILLED", "CLOSED", "PARTIALLY_FILLED", "PARTIAL_FILLED"}:
            continue
        qty = _order_executed_qty(order)
        if qty <= 0:
            continue
        ots = _order_time_utc(order)
        if opened is not None and ots is not None and ots < opened:
            continue
        if opened is not None and ots is not None:
            age_ms = (pd.Timestamp.now(tz="UTC") - ots).total_seconds() * 1000
            if age_ms > EXTERNAL_CLOSE_LOOKBACK_MS:
                continue
        candidates.append((ots or pd.Timestamp(1970, unit="s", tz="UTC"), order))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _recover_exit_stages_from_history(item, position):
    """Recover TP1/TP2/close stages from bot order IDs/client IDs after restart."""
    try:
        opened_at = item.get("opened_at")
        orders = get_recent_demo_orders()
        side = _position_side(position)
        relevant = []
        opened = _normalize_utc_timestamp(opened_at) if opened_at else None
        for order in orders:
            if str(order.get("symbol", "")).upper() != DEMO_SYMBOL.upper():
                continue
            if _order_position_side(order) not in {"", side}:
                continue
            ots = _order_time_utc(order)
            if opened and ots and ots < opened:
                continue
            cid = _order_client_id(order).upper()
            if "-TP1-" in cid or "-TP2-" in cid:
                relevant.append(order)
        for order in sorted(relevant, key=lambda x: _order_time_utc(x) or pd.Timestamp(1970, unit="s", tz="UTC")):
            cid = _order_client_id(order).upper()
            px = _order_fill_price(order)
            if "-TP1-" in cid:
                item["tp1_done"] = True
                if px:
                    item["tp1_price"] = px
            elif "-TP2-" in cid:
                item["tp2_done"] = True
                if px:
                    item["tp2_price"] = px
        return item
    except Exception as e:
        print("⚠️ EXIT STAGE HISTORY RECOVERY ERROR:", repr(e))
        return item


def _get_order_fill_info(order_id, fallback_result=None):
    info = {
        "price": _extract_filled_price(fallback_result),
        "executed_qty": 0.0,
        "status": None,
    }
    if not order_id:
        return info
    try:
        data = authenticated_get(ORDER_HISTORY_PATH, {
            "symbol": DEMO_SYMBOL,
            "limit": 10,
        })
        rows = _parse_order_rows(data)
        for order in rows:
            if not isinstance(order, dict):
                continue
            oid = order.get("orderId") or order.get("id")
            if str(oid) != str(order_id):
                continue
            price = _extract_filled_price(order)
            if price is not None:
                info["price"] = price
            for key in ("executedQty", "filledQty", "cumExecQty", "cumQty"):
                raw = order.get(key)
                if raw is not None:
                    try:
                        q = float(raw)
                        if np.isfinite(q) and q >= 0:
                            info["executed_qty"] = q
                            break
                    except Exception:
                        pass
            info["status"] = order.get("status") or order.get("orderStatus")
            return info
    except Exception as e:
        print("ORDER FILL INFO LOOKUP ERROR:", repr(e))
    return info


def print_demo_positions():
    positions = get_demo_positions()
    print()
    print("-" * 72)
    print("DEMO POSITIONS")
    print("-" * 72)
    if not positions:
        print("NO ACTIVE POSITION")
    else:
        for position in positions:
            print(position)
    print("-" * 72)


def demo_any_position_exists(force_refresh=False):
    for position in get_demo_positions(force_refresh=force_refresh):
        symbol = str(position.get("symbol", "")).upper()
        size = float(position.get("size", 0) or 0)
        if symbol == DEMO_SYMBOL.upper() and size > 0:
            return True
    return False


def get_demo_available_balance():
    data = authenticated_get(BALANCE_PATH)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected demo balance response: {data}")
    for item in data:
        if str(item.get("asset", "")).upper() == "SUSDT":
            value = float(item.get("availableBalance", 0) or 0)
            if not np.isfinite(value) or value <= 0:
                raise RuntimeError(f"Invalid available SUSDT balance: {value}")
            return value
    raise RuntimeError(f"SUSDT balance not found: {data}")


# ============================================================
# POSITION HELPERS
# ============================================================
def _position_key(position):
    return f"{position.get('symbol','').upper()}_{position.get('side','').upper()}"


def _position_entry_price(position):
    size = float(position.get("size", 0) or 0)
    open_value = float(position.get("openValue", 0) or 0)
    if size > 0 and open_value > 0:
        return open_value / size
    for key in ("avgPrice", "entryPrice", "openPrice"):
        raw = position.get(key)
        if raw is not None:
            try:
                value = float(raw)
                if value > 0:
                    return value
            except Exception:
                pass
    return None


def _position_side(position):
    return str(position.get("side", "")).upper()


def _position_size(position):
    return float(position.get("size", 0) or 0)


def _position_created_ts_utc(position):
    try:
        raw = position.get("createdTime")
        if raw is not None:
            ms = float(raw)
            if ms > 0:
                return pd.Timestamp(ms, unit="ms", tz="UTC")
    except Exception:
        pass
    return None


# ============================================================
# V167.8.9 — EFFECTIVE_OPENED_AT
# ============================================================
def _get_effective_opened_at(position, item=None):
    """V167.8.9: only trusted timestamps. Never fabricate age with now()."""
    if item:
        raw = item.get("opened_at")
        ts = _normalize_utc_timestamp(raw) if raw is not None else None
        if ts is not None:
            return ts, "STATE"
    if position:
        try:
            ct = _position_created_ts_utc(position)
            if ct is not None and ct.year >= 2020:
                return ct, "POSITION"
        except Exception as e:
            print(f"⚠️ OPENED_AT position parse failed: {e!r}")
    return None, "UNKNOWN"


# ============================================================
# ROUND PRICE + POSITION SIZE
# ============================================================
def round_price_step(price, step=0.1):
    if not np.isfinite(price) or price <= 0:
        raise ValueError(f"Invalid trigger price: {price}")
    return float(round(np.round(price / step) * step, 1))


def calculate_position_size(entry_price, stop_loss_price, account_balance, risk_percent=None):
    if risk_percent is None:
        risk_percent = ACCOUNT_RISK_PERCENT
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0

    risk_amount = account_balance * risk_percent
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit <= 0:
        return 0

    quantity = risk_amount / risk_per_unit

    safe_notional = account_balance * MAX_LEVERAGE * MAX_MARGIN_UTILIZATION
    max_qty = safe_notional / entry_price

    if quantity > max_qty:
        print(
            f"⚠️ POSITION SIZE CAPPED BY LEVERAGE: risk-implied qty={quantity:.4f} "
            f"> max_qty={max_qty:.4f} "
            f"(MAX_LEVERAGE={MAX_LEVERAGE}x × "
            f"MAX_MARGIN_UTILIZATION={MAX_MARGIN_UTILIZATION:.2f}). "
            f"Effective risk on this trade will be LESS than the configured "
            f"{risk_percent * 100:.2f}% of account."
        )

    quantity = min(quantity, max_qty)
    qty = np.floor(quantity / QTY_STEP) * QTY_STEP
    qty = float(round(qty, 4))
    if qty < MIN_QTY:
        raise RuntimeError(f"Calculated quantity {qty} is below MIN_QTY {MIN_QTY}.")
    return qty


# ============================================================
# TP/SL CALCULATIONS
# ============================================================
def calculate_demo_tp_sl(signal, df, entry_price):
    if not TP_SL_ENABLED:
        return None, None, None, "DISABLED"
    signal = str(signal).upper()
    if signal not in {"LONG", "SHORT"}:
        raise ValueError(f"Invalid signal for TP/SL: {signal}")
    if df is None or len(df) < 50:
        raise RuntimeError("Not enough 15M candles for dynamic exit.")
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_current = float(tr.rolling(14).mean().iloc[-1])
    atr_50_avg = float(tr.rolling(50).mean().iloc[-1])
    if not np.isfinite(atr_current) or atr_current <= 0:
        raise RuntimeError(f"Invalid ATR: {atr_current}")
    volatility_ratio = atr_current / atr_50_avg if atr_50_avg > 0 else 1.0
    tp_mult = EXIT_TP_ATR_MULT_MIN + (EXIT_TP_ATR_MULT_MAX - EXIT_TP_ATR_MULT_MIN) * min(volatility_ratio, 1.0)
    risk = EXIT_SL_ATR_MULT_BASE * atr_current
    reward = tp_mult * risk
    if signal == "LONG":
        sl, tp = entry_price - risk, entry_price + reward
    else:
        sl, tp = entry_price + risk, entry_price - reward
    sl = round_price_step(sl, 0.1)
    tp = round_price_step(tp, 0.1)
    print(f"📊 V167.8.9 DYNAMIC EXIT: SL_ATR={EXIT_SL_ATR_MULT_BASE:.2f} | "
          f"TP_ATR={tp_mult:.2f}R | VOL_RATIO={volatility_ratio:.2f}")
    return atr_current, sl, tp, f"DYNAMIC_{tp_mult:.2f}R"


def calculate_exchange_failsafe_tp_sl(signal, entry_price, risk_distance):
    if entry_price <= 0 or risk_distance <= 0:
        raise ValueError("Invalid entry_price/risk_distance for failsafe TP/SL.")
    failsafe_sl_distance = risk_distance * FAILSAFE_SL_R
    failsafe_tp_distance = risk_distance * FAILSAFE_TP_R
    if signal == "LONG":
        sl = entry_price - failsafe_sl_distance
        tp = entry_price + failsafe_tp_distance
    else:
        sl = entry_price + failsafe_sl_distance
        tp = entry_price - failsafe_tp_distance
    sl = round_price_step(sl, 0.1)
    tp = round_price_step(tp, 0.1)
    return sl, tp


# ============================================================
# V167.8.9-3 NEW: MODIFY EXCHANGE TP/SL AFTER ACTUAL FILL
# ============================================================
def _modify_exchange_tpsl(symbol, position_side, new_tp, new_sl):
    """V167.8.9-3: modify existing exchange TP/SL after actual fill.

    This is a best-effort call. If the exchange rejects the modification
    (e.g. endpoint not supported in demo), we log a warning and continue.
    The failsafe placed at order-time remains valid in that case.
    """
    body = {
        "symbol": symbol,
        "positionSide": position_side,
        "tpTriggerPrice": f"{float(new_tp):.1f}",
        "slTriggerPrice": f"{float(new_sl):.1f}",
        "TpWorkingType": _weex_working_type(),
        "SlWorkingType": _weex_working_type(),
    }
    return authenticated_post(EXCHANGE_TPSL_MODIFY_PATH, body)


# ============================================================
# RISK GUARD
# ============================================================
def risk_guard(signal, df, entry_price, atr, sl, tp):
    if not RISK_GUARD_ENABLED:
        return True, "DISABLED"
    reasons = []

    if RISK_SINGLE_POSITION and demo_any_position_exists(force_refresh=True):
        reasons.append("ACTIVE POSITION EXISTS")

    in_danger_window = is_in_high_risk_session()
    if in_danger_window:
        dangerous, danger_reason = is_high_risk_conditions(df, atr)
        if dangerous:
            reasons.append(
                f"{high_risk_session_label()} → {danger_reason}"
            )
        else:
            print()
            print("=" * 72)
            print("V167.8.9 DANGER WINDOW CHECK")
            print("=" * 72)
            print(f"  Window  : {high_risk_session_label()}")
            print(f"  State   : {danger_reason} → ALLOW entry")
            print("=" * 72)

    if df is None or len(df) < max(20, RISK_ATR_MEDIAN_WINDOW + 14):
        reasons.append("INSUFFICIENT HISTORY")
        return False, "; ".join(reasons)

    x = df.copy()
    for c in ("open", "high", "low", "close"):
        x[c] = x[c].astype(float)
    last = x.iloc[-1]
    body = abs(float(last["close"]) - float(last["open"]))
    if np.isfinite(atr) and atr > 0:
        body_atr = body / atr
        if body_atr > RISK_MAX_BODY_ATR:
            reasons.append(f"EXHAUSTION BODY {body_atr:.2f}ATR")
        ema20 = float(x["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        ema_distance = abs(float(entry_price) - ema20) / atr
        if ema_distance > RISK_MAX_EMA_DISTANCE_ATR:
            reasons.append(f"ENTRY EXTENSION {ema_distance:.2f}ATR")
        close = x["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean()
        valid_atr = atr_series.dropna()
        if len(valid_atr) >= RISK_ATR_MEDIAN_WINDOW:
            baseline = float(valid_atr.iloc[-RISK_ATR_MEDIAN_WINDOW:].median())
            if baseline > 0:
                shock_ratio = atr / baseline
                if shock_ratio > RISK_MAX_ATR_SHOCK_RATIO:
                    reasons.append(f"ATR SHOCK {shock_ratio:.2f}x")
    risk = abs(float(entry_price) - float(sl))
    reward = abs(float(tp) - float(entry_price))
    rr = reward / risk if risk > 0 else 0.0
    if not np.isfinite(rr) or rr < RISK_MIN_RR:
        reasons.append(f"RR {rr:.2f} < MIN {RISK_MIN_RR:.2f}")
    if reasons:
        print()
        print("=" * 72)
        print("V167.8.9 PRE-TRADE RISK GUARD")
        print("=" * 72)
        print("RISK DECISION: REJECTED")
        for reason in reasons:
            print(f"  [BLOCK] {reason}")
        print("=" * 72)
        return False, "; ".join(reasons)
    print()
    print("=" * 72)
    print("V167.8.9 PRE-TRADE RISK GUARD")
    print("=" * 72)
    print("RISK DECISION: ACCEPTED")
    print("=" * 72)
    return True, "RISK_GUARD_PASS"


# ============================================================
# PLACE DEMO MARKET ORDER (with V167.8.9-3 exchange TP/SL modify)
# ============================================================
def place_demo_market_order(signal, candle, df, row):
    if not DEMO_ONLY:
        raise RuntimeError("DEMO_ONLY must remain True.")
    if not DEMO_ORDERS_ENABLED:
        print("DEMO ORDER: DISABLED")
        return None
    signal = str(signal).upper()
    if signal not in {"LONG", "SHORT"}:
        return None

    if not can_send_order(candle):
        print("🔒 ORDER BLOCKED: Another system already processed this candle")
        return None

    if is_circuit_breaker_active(candle):
        print("⛔ DEMO ORDER: BLOCKED BY CIRCUIT BREAKER")
        return None

    scores = calculate_bidirectional_scores(row, df)
    is_reduced = scores.get('is_reduced_size', False)

    entry_price = float(df["close"].iloc[-1])
    atr, sl, tp, protection_method = calculate_demo_tp_sl(signal, df, entry_price)

    risk_ok, risk_reason = risk_guard(signal, df, entry_price, atr, sl, tp)
    if not risk_ok:
        print("DEMO ORDER: BLOCKED BY RISK GUARD")
        print("REASON:", risk_reason)
        return None

    if not can_send_order(candle):
        print("🔒 ORDER BLOCKED (LATE CHECK): Another system already processed this candle")
        return None
    if demo_any_position_exists(force_refresh=True):
        print("🔒 ORDER BLOCKED (LATE CHECK): Active position now exists")
        return None

    available_balance = get_demo_available_balance()

    if is_reduced:
        effective_risk = ACCOUNT_RISK_PERCENT * REDUCED_SIZE_RATIO
        print(f"🟡 REDUCED SIZE: {REDUCED_SIZE_RATIO * 100:.0f}% of normal position")
    else:
        effective_risk = ACCOUNT_RISK_PERCENT

    qty = calculate_position_size(entry_price, sl, available_balance, effective_risk)

    actual_margin = (qty * entry_price) / MAX_LEVERAGE
    actual_notional = qty * entry_price

    max_allowed_margin = available_balance * MAX_MARGIN_UTILIZATION
    if actual_margin > max_allowed_margin:
        safe_qty = np.floor(
            (max_allowed_margin * MAX_LEVERAGE / entry_price) / QTY_STEP
        ) * QTY_STEP
        safe_qty = float(round(safe_qty, 4))
        print(
            f"🛡️ MARGIN PRE-FLIGHT: required={actual_margin:.4f} > "
            f"allowed={max_allowed_margin:.4f} | shrinking {qty:.4f} → {safe_qty:.4f}"
        )
        if safe_qty < MIN_QTY:
            raise RuntimeError(
                f"Available {available_balance:.4f} SUSDT too small for "
                f"MIN_QTY={MIN_QTY} at {MAX_LEVERAGE}x with "
                f"utilization={MAX_MARGIN_UTILIZATION:.2f}."
            )
        qty = safe_qty
        actual_margin = (qty * entry_price) / MAX_LEVERAGE
        actual_notional = qty * entry_price

    effective_risk_pct = (qty * abs(entry_price - sl)) / available_balance * 100
    print(
        f"📐 EFFECTIVE RISK: {effective_risk_pct:.2f}% "
        f"(target was {effective_risk * 100:.2f}%)"
    )

    side = "BUY" if signal == "LONG" else "SELL"
    client_id = ("V167P-" + signal[0] + "-" + pd.Timestamp(candle).strftime("%Y%m%d%H%M"))[:36]

    risk_distance = abs(entry_price - sl)
    failsafe_sl, failsafe_tp = calculate_exchange_failsafe_tp_sl(signal, entry_price, risk_distance)

    weex_working_type = _weex_working_type()

    body = {
        "symbol": DEMO_SYMBOL,
        "side": side,
        "positionSide": signal,
        "type": "MARKET",
        "quantity": f"{qty:.4f}",
        "newClientOrderId": client_id,
        "tpTriggerPrice": f"{failsafe_tp:.1f}",
        "slTriggerPrice": f"{failsafe_sl:.1f}",
        "TpWorkingType": weex_working_type,
        "SlWorkingType": weex_working_type,
    }

    print()
    print("=" * 72)
    print("SENDING WEEX DEMO ORDER + TP/SL")
    print("=" * 72)
    print(f"  SYSTEM          : {SYSTEM_ID} ({OPERATION_MODE})")
    print("SIGNAL        :", signal)
    print("QTY           :", f"{qty:.4f} BTC")
    print(f"RISK PER TRADE: {effective_risk * 100:.1f}% of account (target)")
    if is_reduced:
        print("SIZE MODE     : REDUCED (50%)")
    print("MAX LEVERAGE  :", f"{MAX_LEVERAGE}x")
    print("MARGIN BUFFER :", f"{MAX_MARGIN_UTILIZATION:.2f} × balance max")
    print("AVAILABLE     :", f"{available_balance:.4f} SUSDT")
    print("MARGIN USED   :", f"{actual_margin:.4f} SUSDT "
                            f"({actual_margin / available_balance * 100:.1f}% of balance)")
    print("NOTIONAL      :", f"{actual_notional:.4f} SUSDT")
    print("ENTRY REF     :", f"{entry_price:.2f}")
    print("15M ATR(14)   :", f"{atr:.2f}")
    print("INTERNAL SL   :", f"{sl:.2f}  (managed by exit engine / hard-SL watchdog)")
    print("INTERNAL TP   :", f"{tp:.2f}  (managed by TP1/TP2/trailing)")
    print("EXCHANGE SL   :", f"{failsafe_sl:.2f}  (crash fail-safe, {FAILSAFE_SL_R:.2f}R, {weex_working_type})")
    print("EXCHANGE TP   :", f"{failsafe_tp:.2f}  (crash fail-safe, {FAILSAFE_TP_R:.2f}R, {weex_working_type})")
    print("PROTECTION    :", protection_method)
    print("TRIGGER TYPE  :", weex_working_type)
    print("CLIENT ID     :", client_id)

    result = authenticated_post(DEMO_ORDER_PATH, body)
    print("DEMO ORDER RESPONSE:", result)

    accepted = isinstance(result, dict) and result.get("success") is True
    if not accepted:
        raise RuntimeError(f"WEEX Demo order rejected: {result}")

    print("✅ DEMO ORDER ACCEPTED")

    # ============================================================
    # V167.8.9-3: POST-FILL RECONCILE + EXCHANGE TP/SL MODIFY
    # ============================================================
    try:
        time.sleep(0.5)
        fresh_positions = get_demo_positions(force_refresh=True)
        opened_pos = None
        for p in fresh_positions:
            if (str(p.get("symbol", "")).upper() == DEMO_SYMBOL.upper()
                    and _position_size(p) > 0):
                opened_pos = p
                break

        if opened_pos is not None:
            actual_entry = _position_entry_price(opened_pos)
            created_ms = opened_pos.get("createdTime")

            if actual_entry and actual_entry > 0:
                actual_rd = EXIT_SL_ATR_MULT_BASE * float(atr)
                actual_fsl, actual_ftp = calculate_exchange_failsafe_tp_sl(
                    signal, actual_entry, actual_rd
                )

                if created_ms:
                    _save_entry_atr_snapshot(
                        symbol=DEMO_SYMBOL,
                        side=signal,
                        created_ts_ms=int(float(created_ms)),
                        entry_atr=float(atr),
                        entry_price=float(actual_entry),
                        risk_distance=float(actual_rd),
                    )

                _exit_log_event({
                    "event": "ENTRY_FILLED_RECONCILED",
                    "side": signal,
                    "signal_entry_reference": float(entry_price),
                    "actual_entry_price": float(actual_entry),
                    "entry_slippage": float(actual_entry - entry_price),
                    "entry_atr": float(atr),
                    "risk_distance": float(actual_rd),
                    "expected_exchange_failsafe_sl": float(actual_fsl),
                    "expected_exchange_failsafe_tp": float(actual_ftp),
                })

                print(
                    f"🔧 POST-FILL RECONCILE: signal={entry_price:.2f} "
                    f"actual={actual_entry:.2f} "
                    f"slippage={actual_entry - entry_price:+.2f} "
                    f"risk_distance={actual_rd:.2f}"
                )

                # ---- V167.8.9-3: modify exchange-side TP/SL ----
                try:
                    modify_result = _modify_exchange_tpsl(
                        symbol=DEMO_SYMBOL,
                        position_side=signal,
                        new_tp=float(actual_ftp),
                        new_sl=float(actual_fsl),
                    )
                    print(
                        f"🔧 V167.8.9 EXCHANGE TP/SL MODIFIED: "
                        f"TP={actual_ftp:.1f} SL={actual_fsl:.1f}"
                    )
                    _exit_log_event({
                        "event": "EXCHANGE_TPSL_MODIFIED",
                        "side": signal,
                        "actual_entry": float(actual_entry),
                        "new_tp": float(actual_ftp),
                        "new_sl": float(actual_fsl),
                        "modify_result": modify_result,
                    })
                except Exception as mod_err:
                    print(
                        f"⚠️ V167.8.9 EXCHANGE TP/SL MODIFY FAILED "
                        f"(failsafe remains at signal price): {mod_err!r}"
                    )
                    _exit_log_event({
                        "event": "EXCHANGE_TPSL_MODIFY_FAILED",
                        "side": signal,
                        "actual_entry": float(actual_entry),
                        "error": repr(mod_err),
                    })
        else:
            print("⚠️ V167.8.9: could not read newly-opened position for entry_atr snapshot")
    except Exception as e:
        print(f"⚠️ V167.8.9: entry_atr snapshot save failed: {repr(e)}")

    patterns = get_candle_patterns(df)
    if patterns["detected"]:
        print(f"🕯️ PATTERNS IN THIS CANDLE: {', '.join(patterns['detected'])}")

    save_shared_state(last_candle=candle, status="ORDER_SENT", mode=OPERATION_MODE)

    print("=" * 72)
    return result


# ============================================================
# EXCHANGE PROTECTION RECONCILIATION
# ============================================================
def get_open_demo_orders():
    try:
        payload = authenticated_get(EXCHANGE_OPEN_ORDERS_PATH, {"symbol": DEMO_SYMBOL, "limit": 100, "page": 0})
        return _parse_order_rows(payload)
    except Exception as e:
        print("OPEN ORDERS FETCH ERROR:", repr(e))
        return []


def _extract_trigger_price(order):
    for key in ("triggerPrice", "stopPrice", "price"):
        raw = order.get(key)
        if raw is not None:
            try:
                v = float(raw)
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass
    return None


def reconcile_exchange_protection(position, item):
    if not EXCHANGE_PROTECTION_RECONCILIATION_ENABLED:
        return item
    try:
        orders = get_open_demo_orders()
        side = _position_side(position)
        expected_sl = float(item.get("exchange_failsafe_sl", 0) or 0)
        expected_tp = float(item.get("exchange_failsafe_tp", 0) or 0)
        found_sl = None
        found_tp = None
        for order in orders:
            if str(order.get("symbol", "")).upper() != DEMO_SYMBOL.upper():
                continue
            if _order_position_side(order) not in {"", side}:
                continue
            cid = _order_client_id(order).upper()
            plan = str(order.get("planType") or order.get("algoType") or "").upper()
            px = _extract_trigger_price(order)
            if "STOP" in plan or "SL" in cid:
                found_sl = px or found_sl
            if "TAKE" in plan or "TP" in cid:
                found_tp = px or found_tp
        item["exchange_protection_last_check"] = pd.Timestamp.now(tz="UTC").isoformat()
        item["exchange_protection_sl_observed"] = found_sl
        item["exchange_protection_tp_observed"] = found_tp
        if expected_sl > 0 and found_sl is None:
            item["exchange_protection_status"] = "UNKNOWN_NOT_EXPOSED"
        elif expected_tp > 0 and found_tp is None:
            item["exchange_protection_status"] = "UNKNOWN_NOT_EXPOSED"
        else:
            item["exchange_protection_status"] = "OBSERVED"
        return item
    except Exception as e:
        item["exchange_protection_status"] = "CHECK_ERROR"
        item["exchange_protection_error"] = repr(e)
        print("⚠️ EXCHANGE PROTECTION RECONCILIATION ERROR:", repr(e))
        return item


# ============================================================
# MARK PRICE + LAST PRICE (MULTI-ENDPOINT)
# ============================================================
def get_demo_mark_price(force_refresh=False):
    now = time.time()
    cached_price = _MARK_PRICE_CACHE.get("price")
    cached_ts = float(_MARK_PRICE_CACHE.get("ts", 0.0) or 0.0)
    if (not force_refresh and cached_price is not None
            and now - cached_ts < _MARK_PRICE_CACHE_TTL):
        return float(cached_price)
    r = SESSION.get(BASE_URL + MARK_PRICE_PATH,
                    params={"symbol": "BTCUSDT", "priceType": "MARK"},
                    timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict) or data.get("price") is None:
        raise RuntimeError(f"Unexpected mark-price response: {data}")
    price = float(data["price"])
    _MARK_PRICE_CACHE["price"] = price
    _MARK_PRICE_CACHE["ts"] = now
    return price


_LAST_PRICE_ERR_LAST_LOG = {"ts": 0.0}

_LAST_PRICE_TELEMETRY = {
    "stats": {},
    "current_selected_path": None,
    "last_flush_ts": 0.0,
}


def _last_price_telemetry_record(path, success, latency_ms=None):
    try:
        stats = _LAST_PRICE_TELEMETRY["stats"].setdefault(path, {
            "attempts": 0, "successes": 0, "failures": 0,
            "last_latency_ms": None, "avg_latency_ms": None,
        })
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1
            stats["last_latency_ms"] = round(latency_ms, 1) if latency_ms is not None else None
            if latency_ms is not None:
                prev_avg = stats["avg_latency_ms"]
                stats["avg_latency_ms"] = round(
                    latency_ms if prev_avg is None else (prev_avg * 0.9 + latency_ms * 0.1), 1
                )
        else:
            stats["failures"] += 1
    except Exception as e:
        print(f"⚠️ LAST_PRICE TELEMETRY RECORD ERROR (non-fatal): {repr(e)}")


def _last_price_telemetry_flush(force=False):
    if not LAST_PRICE_TELEMETRY_ENABLED:
        return
    now = time.time()
    if not force and now - _LAST_PRICE_TELEMETRY["last_flush_ts"] < LAST_PRICE_TELEMETRY_FLUSH_SECONDS:
        return
    try:
        tmp = LAST_PRICE_TELEMETRY_FILE + ".tmp"
        payload = {
            "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "selected_path": _LAST_PRICE_TELEMETRY["current_selected_path"],
            "stats": _LAST_PRICE_TELEMETRY["stats"],
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, LAST_PRICE_TELEMETRY_FILE)
        _LAST_PRICE_TELEMETRY["last_flush_ts"] = now
    except Exception as e:
        print(f"⚠️ LAST_PRICE TELEMETRY FLUSH ERROR (non-fatal): {repr(e)}")


def get_demo_last_price(force_refresh=False):
    now = time.time()
    cached_price = _LAST_PRICE_CACHE.get("price")
    cached_ts = float(_LAST_PRICE_CACHE.get("ts", 0.0) or 0.0)
    if (not force_refresh and cached_price is not None
            and now - cached_ts < _MARK_PRICE_CACHE_TTL):
        return float(cached_price)

    last_error = None
    for path, params, key in WEEX_LAST_PRICE_PATHS:
        t0 = time.time()
        try:
            r = SESSION.get(BASE_URL + path, params=params, timeout=REQUEST_TIMEOUT)
            if not r.ok:
                last_error = f"{path} HTTP {r.status_code}"
                if LAST_PRICE_TELEMETRY_ENABLED:
                    _last_price_telemetry_record(path, False)
                continue
            data = r.json()
            if not isinstance(data, dict):
                last_error = f"{path} non-dict response"
                if LAST_PRICE_TELEMETRY_ENABLED:
                    _last_price_telemetry_record(path, False)
                continue
            if key not in data and isinstance(data.get("data"), dict):
                data = data["data"]
            raw = data.get(key)
            if raw is None:
                last_error = f"{path} missing key={key}"
                if LAST_PRICE_TELEMETRY_ENABLED:
                    _last_price_telemetry_record(path, False)
                continue
            price = float(raw)
            if not np.isfinite(price) or price <= 0:
                last_error = f"{path} invalid price={raw}"
                if LAST_PRICE_TELEMETRY_ENABLED:
                    _last_price_telemetry_record(path, False)
                continue

            if LAST_PRICE_TELEMETRY_ENABLED:
                latency_ms = (time.time() - t0) * 1000.0
                _last_price_telemetry_record(path, True, latency_ms)
                if _LAST_PRICE_TELEMETRY["current_selected_path"] != path:
                    print(f"📡 V167.8.9 LAST_PRICE PATH CHANGED: now using '{path}' "
                          f"(latency={latency_ms:.0f}ms)")
                    _LAST_PRICE_TELEMETRY["current_selected_path"] = path
                _last_price_telemetry_flush()

            _LAST_PRICE_CACHE["price"] = price
            _LAST_PRICE_CACHE["ts"] = now
            return price
        except Exception as e:
            last_error = f"{path} {type(e).__name__}: {e}"
            if LAST_PRICE_TELEMETRY_ENABLED:
                _last_price_telemetry_record(path, False)
            continue

    if LAST_PRICE_TELEMETRY_ENABLED:
        _last_price_telemetry_flush(force=True)

    if now - _LAST_PRICE_ERR_LAST_LOG["ts"] > 60.0:
        print(f"⚠️ LAST PRICE all endpoints failed ({last_error}); falling back to MARK")
        _LAST_PRICE_ERR_LAST_LOG["ts"] = now
    try:
        return get_demo_mark_price(force_refresh=force_refresh)
    except Exception:
        return None


def calculate_15m_atr_series(df):
    if df is None or len(df) < 15:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def get_active_poll_interval(current_r):
    try:
        r = float(current_r)
    except Exception:
        return POLL_INTERVAL_NORMAL
    if r < -0.6:
        return POLL_INTERVAL_TIGHT
    if r < -0.3:
        return POLL_INTERVAL_NEAR
    if r > 0.5:
        return POLL_INTERVAL_PROFIT
    return POLL_INTERVAL_NORMAL

# ============================================================
# CIRCUIT BREAKER (hardened)
# ============================================================
def _atomic_json_write(path, payload):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"🚨 ATOMIC JSON WRITE ERROR [{path}]: {e!r}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def get_circuit_breaker_state():
    """V167.8.9: corrupt CB state is quarantined, never silently reset."""
    path = Path(CIRCUIT_BREAKER_FILE)
    default = {
        "loss_streak": 0,
        "cooldown_until": None,
        "last_update": None,
        "state_valid": True,
    }
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("circuit breaker JSON is not an object")
        streak = int(state.get("loss_streak", 0))
        if streak < 0 or streak > 100:
            raise ValueError(f"invalid loss_streak={streak}")
        cd = state.get("cooldown_until")
        if cd is not None and _normalize_utc_timestamp(cd) is None:
            raise ValueError(f"invalid cooldown_until={cd!r}")
        state["loss_streak"] = streak
        state["state_valid"] = True
        return state
    except Exception as e:
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        quarantine = path.with_name(path.name + f".CORRUPT.{stamp}")
        try:
            os.replace(path, quarantine)
            print(f"🚨 CIRCUIT BREAKER CORRUPT: quarantined to {quarantine.name}")
        except Exception as move_err:
            print(f"🚨 CIRCUIT BREAKER CORRUPT AND QUARANTINE FAILED: {move_err!r}")
        print(f"🚨 CIRCUIT BREAKER FAIL-SAFE: reason={e!r}")
        safe = dict(default)
        safe["state_valid"] = False
        safe["last_reset_reason"] = "CORRUPT_STATE_QUARANTINED"
        return safe


def _reset_circuit_breaker_state(reason="UNKNOWN"):
    state = {
        "loss_streak": 0,
        "cooldown_until": None,
        "last_update": pd.Timestamp.now(tz="UTC").isoformat(),
        "last_reset_reason": reason,
        "state_valid": True,
    }
    _atomic_json_write(CIRCUIT_BREAKER_FILE, state)
    return state


def update_circuit_breaker(is_loss, current_candle_time):
    state = get_circuit_breaker_state()
    if not state.get("state_valid", True):
        state = _reset_circuit_breaker_state("CORRUPT_STATE_REINITIALIZED")
    ts = _normalize_utc_timestamp(current_candle_time)
    if ts is None:
        raise ValueError(f"Circuit breaker requires a valid UTC timestamp, got {current_candle_time!r}")
    if is_loss:
        state["loss_streak"] = int(state.get("loss_streak", 0)) + 1
        if state["loss_streak"] >= CIRCUIT_MAX_LOSSES:
            cooldown_until = ts + pd.Timedelta(minutes=15 * CIRCUIT_COOLDOWN_CANDLES)
            state["cooldown_until"] = cooldown_until.isoformat()
            print(f"⛔ CIRCUIT BREAKER ACTIVE until {cooldown_until} (loss_streak={state['loss_streak']})")
    else:
        state["loss_streak"] = 0
        state["cooldown_until"] = None
    state["last_update"] = pd.Timestamp.now(tz="UTC").isoformat()
    state["state_valid"] = True
    if not _atomic_json_write(CIRCUIT_BREAKER_FILE, state):
        raise RuntimeError("Circuit breaker state could not be persisted")
    return state


def is_circuit_breaker_active(current_candle_time):
    state = get_circuit_breaker_state()
    if not state.get("state_valid", True):
        print("⛔ CIRCUIT BREAKER: state was corrupt and has been quarantined; SAFE HOLD")
        return True
    cooldown_raw = state.get("cooldown_until")
    if cooldown_raw is None:
        return False
    cooldown_until = _normalize_utc_timestamp(cooldown_raw)
    current_ts = _normalize_utc_timestamp(current_candle_time)
    if cooldown_until is None or current_ts is None:
        print("⛔ CIRCUIT BREAKER: invalid timestamp state/input; SAFE HOLD")
        return True
    if current_ts >= cooldown_until:
        state["cooldown_until"] = None
        state["loss_streak"] = 0
        state["last_update"] = pd.Timestamp.now(tz="UTC").isoformat()
        _atomic_json_write(CIRCUIT_BREAKER_FILE, state)
        return False
    return True


# ============================================================
# EXIT STATE MANAGEMENT
# ============================================================
def _exit_load_state():
    try:
        path = Path(EXIT_STATE_FILE)
        if not path.exists():
            return {}
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            print("⚠️ EXIT STATE FILE EMPTY — treating as fresh")
            return {}
        data = json.loads(content)
        if not isinstance(data, dict):
            print("⚠️ EXIT STATE FILE NOT A DICT — treating as fresh")
            return {}
        return data
    except json.JSONDecodeError as e:
        print(f"⚠️ EXIT STATE CORRUPTED ({e}) — backing up and rebuilding")
        try:
            backup = EXIT_STATE_FILE + f".corrupt.{int(time.time())}"
            Path(EXIT_STATE_FILE).rename(backup)
            print(f"   Corrupted file backed up to: {backup}")
        except Exception as backup_err:
            print(f"   Failed to backup corrupted file: {repr(backup_err)}")
        return {}
    except Exception as e:
        print("EXIT STATE LOAD ERROR:", repr(e))
        return {}


def _exit_save_state(state):
    try:
        tmp = EXIT_STATE_FILE + ".tmp"
        content = json.dumps(state, ensure_ascii=False, indent=2)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, EXIT_STATE_FILE)
    except Exception as e:
        print("🚨 EXIT STATE SAVE ERROR (exit tracking may be stale):", repr(e))


def _exit_log_event(event):
    try:
        path = Path(EXIT_LOG_FILE)
        event = dict(event)
        event["timestamp_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as e:
        print("EXIT LOG ERROR:", repr(e))


# ============================================================
# EXIT QUANTITIES
# ============================================================
def _round_qty(qty):
    if qty <= 0:
        return 0.0
    rounded = np.floor(qty / QTY_STEP) * QTY_STEP
    return float(round(rounded, 4))


def _calculate_exit_quantities(total_size):
    if total_size <= 0:
        return 0.0, 0.0, 0.0
    tp1_qty = _round_qty(total_size * TP1_CLOSE_PCT)
    tp2_qty = _round_qty(total_size * TP2_CLOSE_PCT)
    runner_qty = _round_qty(total_size - tp1_qty - tp2_qty)
    diff = _round_qty(total_size - (tp1_qty + tp2_qty + runner_qty))
    if diff != 0.0:
        runner_qty = _round_qty(runner_qty + diff)
    return tp1_qty, tp2_qty, runner_qty


def _extract_order_id(result):
    if not isinstance(result, dict):
        return None
    data = result.get("data")
    if not isinstance(data, dict):
        data = {}
    return result.get("orderId") or data.get("orderId") or result.get("id") or data.get("id")


def _extract_filled_price(result):
    if not isinstance(result, dict):
        return None
    data = result.get("data")
    if not isinstance(data, dict):
        data = {}
    for key in ("avgPrice", "filledPrice", "fillPrice", "executedPrice", "price"):
        value = result.get(key, data.get(key))
        if value is not None:
            try:
                value = float(value)
                if np.isfinite(value) and value > 0:
                    return value
            except Exception:
                pass
    return None


def _find_active_position_for_key(positions, key):
    for p in positions or []:
        if _position_key(p) == key and _position_size(p) > 0:
            return p
    return None


def _verify_exit_execution(position, qty, reason, result,
                           max_wait=EXIT_VERIFY_MAX_WAIT, expected_before_size=None):
    requested_qty = max(0.0, float(qty or 0.0))
    before_size = _position_size(position)
    if expected_before_size is not None:
        try:
            before_size = float(expected_before_size)
        except Exception:
            pass
    key = _position_key(position)
    order_id = _extract_order_id(result)
    filled_price = _extract_filled_price(result)
    deadline = time.time() + min(max(0.25, float(max_wait)), EXIT_VERIFY_MAX_WAIT)
    after_size = before_size

    while True:
        try:
            current_positions = get_demo_positions(force_refresh=True)
            current = _find_active_position_for_key(current_positions, key)
            after_size = _position_size(current) if current is not None else 0.0
            reduction = max(0.0, before_size - after_size)
            if reduction >= requested_qty - QTY_STEP / 2:
                break
            if time.time() >= deadline:
                break
        except Exception:
            if time.time() >= deadline:
                break
        time.sleep(min(EXIT_VERIFY_POLL_SECONDS, max(0.05, deadline - time.time())))

    fill_info = _get_order_fill_info(order_id, result) if order_id else {
        "price": filled_price, "executed_qty": 0.0, "status": None
    }
    filled_price = fill_info.get("price")
    history_qty = float(fill_info.get("executed_qty", 0.0) or 0.0)
    reduction = max(0.0, before_size - after_size)
    effective_filled_qty = max(reduction, history_qty)
    fully_requested = reduction >= requested_qty - QTY_STEP / 2
    partially_filled = reduction >= QTY_STEP / 2 and not fully_requested
    verified = bool(fully_requested)

    event = {
        "reason": reason,
        "requested_qty": requested_qty,
        "before_position_size": before_size,
        "after_position_size": after_size,
        "position_reduction": reduction,
        "effective_filled_qty": effective_filled_qty,
        "history_executed_qty": history_qty,
        "order_id": order_id,
        "filled_price": filled_price,
        "order_status": fill_info.get("status"),
        "execution_verified": bool(verified),
        "partial_fill": bool(partially_filled),
        "fill_price_verified": filled_price is not None,
        "raw_response": result,
    }
    _exit_log_event(event)

    state_label = "PASS" if verified else "PARTIAL" if partially_filled else "FAIL"
    print(
        f"🔎 EXIT VERIFY: reason={reason} order_id={order_id} "
        f"before={before_size:.4f} after={after_size:.4f} "
        f"reduced={reduction:.4f} filled={filled_price} execution={state_label}"
    )
    return verified, order_id, filled_price, after_size


def _send_close_order(position, qty, reason, client_suffix=""):
    if qty <= 0:
        return None, False
    side = _position_side(position)
    if side not in {"LONG", "SHORT"}:
        return None, False

    close_side = "SELL" if side == "LONG" else "BUY"
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H%M%S")
    client_id = (f"V167-{reason[:6]}-{side[0]}-{ts}{client_suffix}")[:36]

    body = {
        "symbol": DEMO_SYMBOL,
        "side": close_side,
        "positionSide": side,
        "type": "MARKET",
        "quantity": f"{qty:.4f}",
        "newClientOrderId": client_id,
    }

    print(f"📤 SENDING {reason} DEMO CLOSE ORDER: qty={qty:.4f} side={close_side} | positionSide={side}")
    result = authenticated_post(DEMO_ORDER_PATH, body)
    accepted = isinstance(result, dict) and result.get("success") is True
    if not accepted:
        _exit_log_event({
            "reason": reason,
            "requested_qty": qty,
            "side": close_side,
            "order_id": _extract_order_id(result),
            "filled_price": _extract_filled_price(result),
            "execution_verified": False,
            "fill_price_verified": False,
            "raw_response": result,
        })
        print(f"❌ {reason} ORDER REJECTED: {result}")
        return result, False

    verified, order_id, filled_price, after_size = _verify_exit_execution(
        position, qty, reason, result
    )
    print(
        f"✅ {reason} ORDER: id={order_id} filled={filled_price} "
        f"remaining={after_size:.4f} verified={verified}"
    )
    return result, verified


# ============================================================
# V167.8.9-2: RISK_DISTANCE SANITY CHECK + REBUILD (symmetric helper)
# ============================================================
def _reconstruct_entry_atr_if_needed(item, position, df):
    entry_price = _position_entry_price(position)
    if entry_price is None or entry_price <= 0:
        return item

    side = _position_side(position)
    created_ts = _position_created_ts_utc(position)
    created_ms = None
    if created_ts is not None:
        try:
            created_ms = int(created_ts.value // 1_000_000)
        except Exception:
            created_ms = None

    snap = None
    if created_ms is not None:
        snap = _lookup_entry_atr_snapshot(DEMO_SYMBOL, side, created_ms)
    if snap is not None:
        snap_atr = float(snap.get("entry_atr", 0) or 0)
        snap_rd = float(snap.get("risk_distance", 0) or 0)
        if snap_atr > 0 and snap_rd > 0:
            state_rd = float(item.get("risk_distance", 0) or 0)
            # V167.8.9-2: use the symmetric helper
            if state_rd <= 0 or not _risk_distance_is_sane(state_rd, snap_rd):
                print(f"🔧 V167.8.9 SNAPSHOT-BASED REBUILD: state_rd={state_rd:.2f} "
                      f"snapshot_rd={snap_rd:.2f} (symmetric sanity) → using snapshot")
                item["risk_distance"] = snap_rd
                item["entry_atr"] = snap_atr
                item["hard_sl_price"] = round_price_step(
                    entry_price + snap_rd if side == "SHORT" else entry_price - snap_rd,
                    0.1,
                )
                item["hard_sl_source"] = "SNAPSHOT"
                item["entry_atr_unknown"] = False
                return item

    history_atr = _rebuild_entry_atr_from_history(df, created_ts) if created_ts else None
    if history_atr is not None and history_atr > 0:
        history_rd = EXIT_SL_ATR_MULT_BASE * history_atr
        state_rd = float(item.get("risk_distance", 0) or 0)
        # V167.8.9-2: use the symmetric helper
        if state_rd <= 0 or not _risk_distance_is_sane(state_rd, history_rd):
            print(f"🔧 V167.8.9 HISTORY-BASED REBUILD: state_rd={state_rd:.2f} "
                  f"history_atr={history_atr:.2f} history_rd={history_rd:.2f} "
                  f"(symmetric sanity) → using history")
            item["risk_distance"] = history_rd
            item["entry_atr"] = history_atr
            item["hard_sl_price"] = round_price_step(
                entry_price + history_rd if side == "SHORT" else entry_price - history_rd,
                0.1,
            )
            item["hard_sl_source"] = "HISTORY_REBUILD"
            item["entry_atr_unknown"] = False
            if created_ms is not None:
                _save_entry_atr_snapshot(DEMO_SYMBOL, side, created_ms,
                                         history_atr, entry_price, history_rd)
            return item

    state_rd = float(item.get("risk_distance", 0) or 0)
    if state_rd <= 0:
        print("🚫 V167.8.9 ENTRY_ATR_UNKNOWN — SAFE HOLD")
        print("   Position will be managed by exchange-side SL/TP only.")
        print("   No internal R/SL/TP will be computed.")
        item["entry_atr_unknown"] = True
        item["hard_sl_source"] = "UNKNOWN_HOLD"
        item["unknown_reason"] = "no_snapshot_no_history"
    else:
        item["entry_atr_unknown"] = False
    return item


def _ensure_exit_state_for_position(item, position, df):
    if item:
        if "entry_atr" not in item:
            rd = float(item.get("risk_distance", 0) or 0)
            item["entry_atr"] = (rd / EXIT_SL_ATR_MULT_BASE) if rd > 0 and EXIT_SL_ATR_MULT_BASE > 0 else 0.0
        if "tp1_r_initial" not in item:
            item["tp1_r_initial"] = TP1_R
        if "tp2_r_initial" not in item:
            item["tp2_r_initial"] = TP2_R
        if "tp1_r_effective" not in item:
            item["tp1_r_effective"] = TP1_R
        if "tp2_r_effective" not in item:
            item["tp2_r_effective"] = TP2_R
        if "tp_compressed" not in item:
            item["tp_compressed"] = False
        if "entry_atr_unknown" not in item:
            item["entry_atr_unknown"] = False
        if "thesis_status" not in item:
            item["thesis_status"] = "UNKNOWN"
        if "thesis_valid" not in item:
            item["thesis_valid"] = True
        if "thesis_invalidated_at" not in item:
            item["thesis_invalidated_at"] = None
        if "thesis_age_hours" not in item:
            item["thesis_age_hours"] = None
        if "thesis_invalidated_r" not in item:
            item["thesis_invalidated_r"] = None

        effective_opened_at, opened_at_source = _get_effective_opened_at(position, item)
        old_opened_raw = item.get("opened_at")
        try:
            old_ts = pd.Timestamp(old_opened_raw) if old_opened_raw else None
            if old_ts is not None and old_ts.tzinfo is None:
                old_ts = old_ts.tz_localize("UTC")
        except Exception:
            old_ts = None
        if old_ts is None or old_ts != effective_opened_at:
            item["opened_at"] = effective_opened_at.isoformat()
            item["opened_at_source"] = opened_at_source
            print(f"🔧 V167.8.9 OPENED_AT RECONCILED: "
                  f"{old_opened_raw!r} → {item['opened_at']} (source={opened_at_source})")
        else:
            item["opened_at_source"] = opened_at_source

        item = _reconstruct_entry_atr_if_needed(item, position, df)
        return item

    entry = _position_entry_price(position)
    if entry is None or entry <= 0:
        return None

    side = _position_side(position)
    created_ts = _position_created_ts_utc(position)
    created_ms = None
    if created_ts is not None:
        try:
            created_ms = int(created_ts.value // 1_000_000)
        except Exception:
            created_ms = None

    entry_atr = None
    risk_distance = None
    source = "UNKNOWN_HOLD"
    if created_ms is not None:
        snap = _lookup_entry_atr_snapshot(DEMO_SYMBOL, side, created_ms)
        if snap is not None:
            snap_atr = float(snap.get("entry_atr", 0) or 0)
            snap_rd = float(snap.get("risk_distance", 0) or 0)
            if snap_atr > 0 and snap_rd > 0:
                entry_atr = snap_atr
                risk_distance = snap_rd
                source = "SNAPSHOT"
                print(f"🔧 V167.8.9 FRESH STATE: using SNAPSHOT "
                      f"entry_atr={entry_atr:.2f} risk_distance={risk_distance:.2f}")

    if entry_atr is None or risk_distance is None:
        hist_atr = _rebuild_entry_atr_from_history(df, created_ts) if created_ts else None
        if hist_atr is not None and hist_atr > 0:
            entry_atr = hist_atr
            risk_distance = EXIT_SL_ATR_MULT_BASE * hist_atr
            source = "HISTORY_REBUILD"
            print(f"🔧 V167.8.9 FRESH STATE: using HISTORY_REBUILD "
                  f"entry_atr={entry_atr:.2f} risk_distance={risk_distance:.2f}")

    if risk_distance is None or risk_distance <= 0:
        size = _position_size(position)
        hard_sl = None
        effective_opened_at, opened_at_source = _get_effective_opened_at(position, None)
        opened_at_str = effective_opened_at.isoformat() if effective_opened_at is not None else None
        print("🚫 V167.8.9 FRESH STATE: ENTRY_ATR_UNKNOWN — SAFE HOLD")
        print(f"   position={DEMO_SYMBOL} {side} size={size}")
        print("   No internal R/SL/TP will be computed.")
        return {
            "entry_price": entry,
            "risk_distance": 0.0,
            "entry_atr": 0.0,
            "hard_sl_price": None,
            "hard_sl_source": "UNKNOWN_HOLD",
            "side": side,
            "initial_size": size,
            "tp1_done": False,
            "tp2_done": False,
            "be_armed": False,
            "trailing_active": False,
            "trailing_peak": entry,
            "opened_at": opened_at_str,
            "opened_at_source": opened_at_source,
            "pending_exit": None,
            "recovered": True,
            "tp1_r_initial": TP1_R,
            "tp2_r_initial": TP2_R,
            "tp1_r_effective": TP1_R,
            "tp2_r_effective": TP2_R,
            "tp_compressed": False,
            "entry_atr_unknown": True,
            "unknown_reason": "fresh_state_no_source",
            "thesis_status": "UNKNOWN",
            "thesis_valid": True,
            "thesis_invalidated_at": None,
            "thesis_age_hours": None,
            "thesis_invalidated_r": None,
        }

    size = _position_size(position)
    hard_sl = entry - risk_distance if side == "LONG" else entry + risk_distance
    hard_sl = round_price_step(hard_sl, 0.1)

    effective_opened_at, opened_at_source = _get_effective_opened_at(position, None)
    opened_at_str = effective_opened_at.isoformat() if effective_opened_at is not None else None

    if created_ms is not None and source == "HISTORY_REBUILD":
        _save_entry_atr_snapshot(DEMO_SYMBOL, side, created_ms,
                                 entry_atr, entry, risk_distance)

    return {
        "entry_price": entry,
        "risk_distance": risk_distance,
        "entry_atr": entry_atr,
        "hard_sl_price": hard_sl,
        "hard_sl_source": source,
        "side": side,
        "initial_size": size,
        "tp1_done": False,
        "tp2_done": False,
        "be_armed": False,
        "trailing_active": False,
        "trailing_peak": entry,
        "opened_at": opened_at_str,
        "opened_at_source": opened_at_source,
        "pending_exit": None,
        "recovered": True,
        "tp1_r_initial": TP1_R,
        "tp2_r_initial": TP2_R,
        "tp1_r_effective": TP1_R,
        "tp2_r_effective": TP2_R,
        "tp_compressed": False,
        "entry_atr_unknown": False,
        "thesis_status": "UNKNOWN",
        "thesis_valid": True,
        "thesis_invalidated_at": None,
        "thesis_age_hours": None,
        "thesis_invalidated_r": None,
    }


def _mark_exit_stage_done(item, reason, mark, current_r):
    if reason == "TP1":
        item["tp1_done"] = True
        item["tp1_price"] = mark
        item["tp1_r"] = current_r
    elif reason == "TP2":
        item["tp2_done"] = True
        item["tp2_price"] = mark
        item["tp2_r"] = current_r


def _clear_pending_after_partial(item, pending_reason, after_size, requested_qty, before_size):
    reduction = max(0.0, float(before_size) - float(after_size))
    item["pending_exit"] = None
    item["last_partial_exit"] = {
        "reason": pending_reason,
        "requested_qty": requested_qty,
        "filled_qty": reduction,
        "remaining_requested": max(0.0, requested_qty - reduction),
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    return reduction


def _record_close_for_circuit_breaker(item, reason, position, closed_at=None, exit_price=None):
    """V167.8.9: closed_at and exit_price are separate, typed values."""
    closed_ts = _normalize_utc_timestamp(closed_at) or pd.Timestamp.now(tz="UTC")
    if exit_price is None and position:
        exit_price = _position_entry_price(position)
    try:
        exit_price_f = float(exit_price) if exit_price is not None else None
        if exit_price_f is not None and (not np.isfinite(exit_price_f) or exit_price_f <= 0):
            exit_price_f = None
    except Exception:
        exit_price_f = None
    entry = float((item or {}).get("entry_price", 0) or 0)
    rd = float((item or {}).get("risk_distance", 0) or 0)
    side = str((item or {}).get("side", "")).upper()
    final_r = None
    if entry > 0 and rd > 0 and exit_price_f is not None:
        final_r = ((exit_price_f - entry) / rd) if side == "LONG" else ((entry - exit_price_f) / rd)
    was_invalidated = bool((item or {}).get("thesis_invalidated_at"))
    _exit_log_event({
        "event": "TRADE_CLOSED",
        "reason": reason,
        "side": side,
        "entry_price": entry if entry > 0 else None,
        "exit_price": exit_price_f,
        "final_r": round(final_r, 3) if final_r is not None else None,
        "closed_at": closed_ts.isoformat(),
        "opened_at": (item or {}).get("opened_at"),
        "opened_at_source": (item or {}).get("opened_at_source"),
        "tp1_done": bool((item or {}).get("tp1_done", False)),
        "tp2_done": bool((item or {}).get("tp2_done", False)),
        "tp_compressed": bool((item or {}).get("tp_compressed", False)),
        "tp_compression_atr_ratio": (item or {}).get("tp_compression_atr_ratio"),
        "tp_compression_age_hours": (item or {}).get("tp_compression_age_hours"),
        "thesis_status": (item or {}).get("thesis_status"),
        "thesis_invalidated_at": (item or {}).get("thesis_invalidated_at"),
        "thesis_age_hours": (item or {}).get("thesis_age_hours"),
        "thesis_invalidated_r": (item or {}).get("thesis_invalidated_r"),
        "thesis_final_reason": reason if was_invalidated else None,
        "thesis_final_r": round(final_r, 3) if (final_r is not None and was_invalidated) else None,
    })
    try:
        if reason in {"SL", "SL_EXTERNAL"}:
            update_circuit_breaker(True, closed_ts)
            print("🧮 CIRCUIT BREAKER: loss recorded (SL)")
        elif reason in {"TRAIL", "TIMEEXIT"}:
            update_circuit_breaker(False, closed_ts)
            print("🧮 CIRCUIT BREAKER: win recorded (TRAIL/TIMEEXIT)")
        elif reason == "BE":
            print("🧮 CIRCUIT BREAKER: BE scratch (streak unchanged)")
    except Exception as e:
        print("CIRCUIT BREAKER RECORD ERROR:", repr(e))
    return final_r


# ============================================================
# MANAGE POSITION EXITS (V167.8.9-4: trailing tied to tp2_r_effective)
# ============================================================
def manage_position_exits(df=None):
    if not DEMO_ONLY or not DEMO_ORDERS_ENABLED:
        return False

    try:
        positions = get_demo_positions(force_refresh=False)
        active = [
            p for p in positions
            if str(p.get("symbol", "")).upper() == DEMO_SYMBOL.upper()
            and _position_size(p) > 0
        ]
        state = _exit_load_state()
        active_keys = {_position_key(p) for p in active}
        for k in list(state):
            if k not in active_keys:
                stale_item = state.get(k) or {}
                side_stale = str(stale_item.get("side", "")).upper()
                opened_stale = stale_item.get("opened_at")
                close_order = _find_recent_close_order(stale_item, side_stale, opened_stale)
                if close_order is not None:
                    close_price = _order_fill_price(close_order)
                    close_time = _order_time_utc(close_order) or pd.Timestamp.now(tz="UTC")
                    close_reason = _infer_order_reason(close_order, stale_item)
                    print(f"🔎 EXTERNAL CLOSE RECONCILIATION: key={k} reason={close_reason} price={close_price}")
                    _record_close_for_circuit_breaker(stale_item, close_reason, {}, close_time, close_price)
                else:
                    print(f"🔎 EXTERNAL CLOSE RECONCILIATION: key={k} no matching fill found; logging UNKNOWN close")
                    _record_close_for_circuit_breaker(stale_item, "EXTERNAL_CLOSE_UNKNOWN", {}, pd.Timestamp.now(tz="UTC"), None)
                state.pop(k, None)
        if not active:
            _exit_save_state(state)
            try:
                snap_keys = set()
                for p in positions:
                    s = _position_created_ts_utc(p)
                    if s is not None:
                        ms = int(s.value // 1_000_000)
                        snap_keys.add(_entry_atr_key(DEMO_SYMBOL, _position_side(p), ms))
                _cleanup_entry_atr_snapshots(snap_keys)
            except Exception:
                pass
            return False

        mark = get_demo_mark_price()
        last = None
        try:
            last = get_demo_last_price()
        except Exception as e:
            print(f"⚠️ LAST PRICE FETCH ERROR (will fall back to mark only): {repr(e)}")

        print(f"🧭 EXIT MANAGER: ACTIVE | positions={len(active)} | mark={mark:.2f}"
              + (f" | last={last:.2f}" if last is not None else ""))

        for live_position in active:
            key = _position_key(live_position)
            side = _position_side(live_position)
            size = _position_size(live_position)
            item = state.get(key, {})
            item = _ensure_exit_state_for_position(item, live_position, df)
            item = _recover_exit_stages_from_history(item, live_position)
            if not item:
                print(f"⚠️ EXIT RECOVERY: unable to initialize state for {key}")
                continue
            if key not in state:
                state[key] = item
                _exit_save_state(state)
                print(f"🔄 EXIT RECOVERY: state initialized for {key}")

            fresh_positions = get_demo_positions(force_refresh=True)
            position = _find_active_position_for_key(fresh_positions, key)
            if position is None:
                state.pop(key, None)
                _exit_save_state(state)
                continue
            size = _position_size(position)
            side = _position_side(position)

            entry_price = float(item["entry_price"])
            risk_distance = float(item.get("risk_distance", 0) or 0)

            if item.get("entry_atr_unknown", False) or risk_distance <= 0:
                print(f"🚫 EXIT HOLD: entry_atr unknown for {key} — "
                      f"exchange-side SL/TP is the sole protection")
                state[key] = item
                _exit_save_state(state)
                continue

            if entry_price <= 0 or size <= 0:
                continue

            hard_sl = item.get("hard_sl_price")
            if hard_sl is None or not np.isfinite(float(hard_sl)) or float(hard_sl) <= 0:
                hard_sl = entry_price - risk_distance if side == "LONG" else entry_price + risk_distance
                hard_sl = round_price_step(hard_sl, 0.1)
                item["hard_sl_price"] = hard_sl
                item["hard_sl_source"] = "INITIAL_RISK_DISTANCE"
                state[key] = item
                _exit_save_state(state)

            item = reconcile_exchange_protection(position, item)
            state[key] = item
            _exit_save_state(state)

            pending = item.get("pending_exit")
            if pending:
                pending_qty = max(0.0, float(pending.get("qty", 0) or 0))
                pending_reason = str(pending.get("reason", "EXIT"))
                pending_started = float(pending.get("created_ts", 0) or 0)
                pending_age = time.time() - pending_started if pending_started else 0.0
                verify_wait = min(EXIT_VERIFY_MAX_WAIT, max(0.25, PENDING_EXIT_TIMEOUT - max(0.0, pending_age)))
                before_size = float(pending.get("before_size", size) or size)

                verified, order_id, filled_price, after_size = _verify_exit_execution(
                    position,
                    pending_qty,
                    pending_reason,
                    pending.get("result", {}),
                    max_wait=verify_wait,
                    expected_before_size=before_size,
                )
                reduction = max(0.0, before_size - after_size)

                if verified:
                    _mark_exit_stage_done(item, pending_reason, mark, 0.0)
                    if pending_reason == "SL":
                        item["sl_triggered"] = True
                        item["sl_trigger_price"] = float(item.get("hard_sl_price", 0.0) or 0.0)
                        item["sl_trigger_mark"] = mark
                        item["sl_trigger_r"] = pending.get("sl_trigger_r", item.get("sl_trigger_r"))
                        item["sl_filled"] = True
                        item["sl_fill_price"] = filled_price
                    item["last_exit_order_id"] = order_id
                    item["last_exit_filled_price"] = filled_price
                    item["pending_exit"] = None
                    state[key] = item
                    _exit_save_state(state)
                    print(f"✅ EXIT RECOVERY: {pending_reason} fully confirmed")
                    if pending_reason in {"TRAIL", "BE", "TIMEEXIT"}:
                        _record_close_for_circuit_breaker(
                            item, pending_reason, position, pd.Timestamp.now(tz="UTC"), filled_price
                        )
                        state.pop(key, None)
                        _exit_save_state(state)
                        continue
                    fresh = get_demo_positions(force_refresh=True)
                    position = _find_active_position_for_key(fresh, key)
                    if position is None:
                        _record_close_for_circuit_breaker(
                            item, pending_reason, position or {}, pd.Timestamp.now(tz="UTC"), filled_price
                        )
                        state.pop(key, None)
                        _exit_save_state(state)
                        continue
                    size = _position_size(position)
                elif reduction >= QTY_STEP / 2:
                    _clear_pending_after_partial(item, pending_reason, after_size, pending_qty, before_size)
                    item["last_exit_order_id"] = order_id
                    item["last_exit_filled_price"] = filled_price
                    state[key] = item
                    _exit_save_state(state)
                    print(f"🟠 EXIT PARTIAL: {pending_reason} reduced {reduction:.4f}; re-evaluating remainder")
                    fresh = get_demo_positions(force_refresh=True)
                    position = _find_active_position_for_key(fresh, key)
                    if position is None:
                        state.pop(key, None)
                        _exit_save_state(state)
                        continue
                    size = _position_size(position)
                elif pending_age >= PENDING_EXIT_TIMEOUT:
                    pending_order_id = _extract_order_id(pending.get("result", {}))
                    order_info = _get_order_fill_info(pending_order_id, pending.get("result", {})) if pending_order_id else {"price": None, "executed_qty": 0.0, "status": None}
                    order_status = str(order_info.get("status") or "").upper()
                    terminal_reject = order_status in {"CANCELED", "CANCELLED", "REJECTED", "EXPIRED"}
                    if order_status in {"FILLED", "PARTIALLY_FILLED", "PARTIAL_FILLED", "CLOSED"}:
                        item["pending_last_status"] = order_status
                        item["pending_last_check_ts"] = time.time()
                        state[key] = item
                        _exit_save_state(state)
                        print(f"🔎 PENDING RECONCILE: {pending_reason} status={order_status}; waiting for position delta")
                        continue
                    if terminal_reject:
                        item["pending_exit"] = None
                        item["last_exit_reject_status"] = order_status
                        item["last_exit_reject_ts"] = pd.Timestamp.now(tz="UTC").isoformat()
                        state[key] = item
                        _exit_save_state(state)
                        print(f"↩️ PENDING EXIT TERMINAL {order_status}: safe to re-evaluate")
                        fresh = get_demo_positions(force_refresh=True)
                        position = _find_active_position_for_key(fresh, key)
                        if position is None:
                            state.pop(key, None)
                            _exit_save_state(state)
                            continue
                        size = _position_size(position)
                    else:
                        item["pending_last_status"] = order_status or "UNKNOWN"
                        item["pending_last_check_ts"] = time.time()
                        item["pending_exit_deadline"] = time.time() + 10.0
                        state[key] = item
                        _exit_save_state(state)
                        print(f"🛡️ PENDING EXIT HELD: {pending_reason} status={order_status or 'UNKNOWN'}; NO DUPLICATE")
                        continue
                else:
                    state[key] = item
                    _exit_save_state(state)
                    print(f"⏳ EXIT PENDING: {pending_reason} age={pending_age:.1f}s")
                    continue

            current_r = ((mark - entry_price) / risk_distance) if side == "LONG" else ((entry_price - mark) / risk_distance)
            if last is not None:
                current_r_last = ((last - entry_price) / risk_distance) if side == "LONG" else ((entry_price - last) / risk_distance)
            else:
                current_r_last = current_r

            if side == "LONG":
                item["trailing_peak"] = max(float(item.get("trailing_peak", entry_price)), mark)
            else:
                item["trailing_peak"] = min(float(item.get("trailing_peak", entry_price)), mark)

            # ============================================================
            # THESIS OBSERVATION (LOG ONLY)
            # ============================================================
            thesis_valid_obs = item.get("thesis_valid", True)
            if THESIS_OBSERVATION_ENABLED and not item.get("tp1_done", False) and risk_distance > 0:
                opened_at_obs, opened_at_src_obs = _get_effective_opened_at(position, item)
                age_hours_obs = (pd.Timestamp.now(tz="UTC") - opened_at_obs).total_seconds() / 3600
                if opened_at_src_obs != "STATE":
                    print(f"🔧 V167.8.9 THESIS_AGE source={opened_at_src_obs} "
                          f"opened_at={opened_at_obs.isoformat()} age={age_hours_obs:.2f}h")
                item["opened_at_source"] = opened_at_src_obs
                if item.get("opened_at") != opened_at_obs.isoformat():
                    item["opened_at"] = opened_at_obs.isoformat()
                    state[key] = item
                    _exit_save_state(state)

                try:
                    if df is not None and len(df) >= 1:
                        current_row_obs = build_features(df)
                        if current_row_obs is not None:
                            thesis_valid_obs = is_entry_thesis_valid(item, current_row_obs)
                except Exception as e:
                    print(f"⚠️ V167.8.9 THESIS OBSERVATION error (ignoring, log only): {repr(e)}")
                    thesis_valid_obs = item.get("thesis_valid", True)

                item["thesis_valid"] = thesis_valid_obs
                item["thesis_valid_last_check"] = pd.Timestamp.now(tz="UTC").isoformat()
                item["thesis_status"] = "VALID" if thesis_valid_obs else "INVALID"

                if not thesis_valid_obs and item.get("thesis_invalidated_at") is None:
                    item["thesis_invalidated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
                    item["thesis_age_hours"] = round(age_hours_obs, 2)
                    item["thesis_invalidated_r"] = round(current_r, 3)
                    print(
                        f"🔎 V167.8.9 THESIS INVALIDATED (log only, no exit-behavior change): "
                        f"key={key} age={age_hours_obs:.1f}h R={current_r:.2f}"
                    )
                    _exit_log_event({
                        "event": "THESIS_INVALIDATED",
                        "reason": "THESIS_OBSERVATION",
                        "position_key": key,
                        "side": side,
                        "thesis_age_hours": round(age_hours_obs, 2),
                        "thesis_invalidated_r": round(current_r, 3),
                    })
                    state[key] = item
                    _exit_save_state(state)

            # ============================================================
            # DYNAMIC TP COMPRESSION — every iteration
            # ============================================================
            if (TP_COMPRESSION_ENABLED 
                    and not item.get("tp_compressed", False)
                    and not item.get("tp1_done", False)
                    and risk_distance > 0):
                opened_at, opened_at_src = _get_effective_opened_at(position, item)
                if opened_at is None:
                    item["opened_at_source"] = "UNKNOWN"
                    item["age_status"] = "UNKNOWN_SAFE_HOLD"
                    print("🛡️ TP COMPRESSION HOLD: opened_at is unknown")
                    age_hours = None
                else:
                    age_hours = ((pd.Timestamp.now(tz="UTC") - opened_at).total_seconds() / 3600 if opened_at is not None else None)
                if opened_at_src != "STATE" and opened_at is not None:
                    print(f"🔧 V167.8.9 TP_COMPRESSION_AGE source={opened_at_src} "
                          f"opened_at={opened_at.isoformat()} age={age_hours:.2f}h")
                item["opened_at_source"] = opened_at_src
                if opened_at is not None and item.get("opened_at") != opened_at.isoformat():
                    item["opened_at"] = opened_at.isoformat()
                    state[key] = item
                    _exit_save_state(state)

                entry_atr = float(item.get("entry_atr", 0.0) or 0.0)
                if entry_atr <= 0 and EXIT_SL_ATR_MULT_BASE > 0:
                    entry_atr = risk_distance / EXIT_SL_ATR_MULT_BASE
                current_atr = entry_atr
                if df is not None and len(df) >= 15:
                    atr_series = calculate_15m_atr_series(df)
                    valid = atr_series.dropna() if atr_series is not None else pd.Series(dtype=float)
                    if not valid.empty:
                        current_atr = float(valid.iloc[-1])
                atr_ratio = (current_atr / entry_atr) if entry_atr > 0 else 1.0

                if THESIS_OBSERVATION_ENABLED:
                    thesis_valid = thesis_valid_obs
                else:
                    thesis_valid = True
                    try:
                        if df is not None and len(df) >= 1:
                            current_row = build_features(df)
                            if current_row is not None:
                                thesis_valid = is_entry_thesis_valid(item, current_row)
                                item["thesis_valid_last_check"] = pd.Timestamp.now(tz="UTC").isoformat()
                                item["thesis_valid"] = thesis_valid
                    except Exception as e:
                        print(f"⚠️ Thesis check error (ignoring): {repr(e)}")
                        thesis_valid = True

                if age_hours is not None and age_hours >= TP_COMPRESSION_MIN_AGE_HOURS and atr_ratio < TP_COMPRESSION_ATR_RATIO:
                    if thesis_valid:
                        item["tp_compressed"] = True
                        item["tp1_r_effective"] = TP_COMPRESSION_TP1_R
                        item["tp2_r_effective"] = TP_COMPRESSION_TP2_R
                        item["tp_compressed_at"] = pd.Timestamp.now(tz="UTC").isoformat()
                        item["tp_compression_age_hours"] = age_hours
                        item["tp_compression_atr_ratio"] = atr_ratio
                        state[key] = item
                        _exit_save_state(state)
                        print(
                            f"📉 V167.8.9 TP COMPRESSED: age={age_hours:.1f}h "
                            f"atr_ratio={atr_ratio:.3f} (< {TP_COMPRESSION_ATR_RATIO:.2f}) "
                            f"thesis=VALID → TP1={TP_COMPRESSION_TP1_R}R TP2={TP_COMPRESSION_TP2_R}R"
                        )
                    else:
                        item["compression_skipped_reason"] = "THESIS_INVALID"
                        item["compression_last_check_ts"] = pd.Timestamp.now(tz="UTC").isoformat()
                        state[key] = item
                        _exit_save_state(state)
                        print(
                            f"⚠️ V167.8.9 TP COMPRESSION SKIPPED: age={age_hours:.1f}h "
                            f"atr_ratio={atr_ratio:.3f} thesis=INVALID"
                        )
                else:
                    if age_hours < TP_COMPRESSION_MIN_AGE_HOURS:
                        reason = f"AGE {age_hours:.1f}h < {TP_COMPRESSION_MIN_AGE_HOURS}h"
                    else:
                        reason = f"ATR_RATIO {atr_ratio:.3f} >= {TP_COMPRESSION_ATR_RATIO}"
                    if item.get("compression_skipped_reason") != reason:
                        item["compression_skipped_reason"] = reason
                        item["compression_last_check_ts"] = pd.Timestamp.now(tz="UTC").isoformat()
                        state[key] = item
                        _exit_save_state(state)

            tp1_r_eff = float(item.get("tp1_r_effective", TP1_R))
            tp2_r_eff = float(item.get("tp2_r_effective", TP2_R))

            hard_sl_display = float(item.get("hard_sl_price", 0.0) or 0.0)
            print(
                f"📊 EXIT STATE: {key} size={size:.4f} R={current_r:.2f} "
                f"R_last={current_r_last:.2f} "
                f"rd={risk_distance:.2f} "
                f"SL={hard_sl_display:.2f} "
                f"src={item.get('hard_sl_source', 'INIT')} "
                f"TP1={'Y' if item.get('tp1_done') else 'N'} "
                f"TP2={'Y' if item.get('tp2_done') else 'N'} "
                f"BE={'Y' if item.get('be_armed') else 'N'} "
                f"TRAIL={'Y' if item.get('trailing_active') else 'N'} "
                f"TP_COMP={'Y' if item.get('tp_compressed') else 'N'} "
                f"(TP1eff={tp1_r_eff:.2f}R TP2eff={tp2_r_eff:.2f}R)"
            )

            hard_sl = float(item.get("hard_sl_price", 0.0) or 0.0)
            trigger_buffer = max(0.0, float(HARD_SL_TRIGGER_TOLERANCE or 0.0))
            if HARD_SL_WATCHDOG_ENABLED and hard_sl > 0 and size >= MIN_QTY:
                if side == "LONG":
                    sl_hit_mark = mark <= hard_sl + trigger_buffer
                    sl_hit_last = (last is not None) and (last <= hard_sl + trigger_buffer)
                else:
                    sl_hit_mark = mark >= hard_sl - trigger_buffer
                    sl_hit_last = (last is not None) and (last >= hard_sl - trigger_buffer)
                sl_hit = sl_hit_mark or sl_hit_last
                if sl_hit_mark and sl_hit_last:
                    trigger_source = "MARK+LAST"
                elif sl_hit_mark:
                    trigger_source = "MARK"
                elif sl_hit_last:
                    trigger_source = "LAST"
                else:
                    trigger_source = "NONE"
            else:
                sl_hit = False
                trigger_source = "NONE"

            if HARD_SL_WATCHDOG_ENABLED and hard_sl > 0 and sl_hit and size >= MIN_QTY:
                print(f"🛑 HARD SL TRIGGERED | source={trigger_source} | side={side} "
                      f"mark={mark:.2f} last={last if last is not None else 'n/a'} "
                      f"stop={hard_sl:.2f} R={current_r:.2f}")
                item["sl_trigger_pending"] = True
                item["sl_trigger_price"] = hard_sl
                item["sl_trigger_mark"] = mark
                item["sl_trigger_last"] = last
                item["sl_trigger_source"] = trigger_source
                item["sl_trigger_r"] = current_r
                state[key] = item
                _exit_save_state(state)
                result, verified = _send_close_order(position, size, "SL")
                if verified:
                    item["sl_triggered"] = True
                    item["sl_trigger_pending"] = False
                    item["sl_filled"] = True
                    item["sl_fill_price"] = _extract_filled_price(result)
                    _record_close_for_circuit_breaker(
                        item, "SL", position, pd.Timestamp.now(tz="UTC"), _extract_filled_price(result)
                    )
                    state.pop(key, None)
                    _exit_save_state(state)
                    print("✅ HARD SL FILLED + POSITION CLOSED")
                else:
                    item["pending_exit"] = {"reason": "SL", "qty": size, "before_size": size,
                                            "result": result, "created_ts": time.time(),
                                            "sl_trigger_price": hard_sl,
                                            "sl_trigger_mark": mark,
                                            "sl_trigger_last": last,
                                            "sl_trigger_source": trigger_source,
                                            "sl_trigger_r": current_r}
                    state[key] = item
                    _exit_save_state(state)
                    print("⚠️ HARD SL SENT BUT FILL NOT YET VERIFIED")
                continue

            if TIME_EXIT_ENABLED:
                opened_at_te, opened_at_src_te = _get_effective_opened_at(position, item)
                if opened_at_te is None:
                    print("🛡️ TIME EXIT HOLD: opened_at unknown, refusing to fabricate trade age")
                    hours_open = None
                else:
                    hours_open = (pd.Timestamp.now(tz="UTC") - opened_at_te).total_seconds() / 3600
                if opened_at_src_te != "STATE" and opened_at_te is not None:
                    print(f"🔧 V167.8.9 TIME_EXIT_AGE source={opened_at_src_te} "
                          f"opened_at={opened_at_te.isoformat()} hours_open={hours_open:.2f}h")
                    item["opened_at"] = opened_at_te.isoformat()
                    item["opened_at_source"] = opened_at_src_te
                    state[key] = item
                    _exit_save_state(state)
                if hours_open is not None and hours_open >= TIME_EXIT_HOURS and current_r >= TIME_EXIT_MIN_PROFIT_R:
                    print(f"⏰ TIME EXIT: {hours_open:.1f}h >= {TIME_EXIT_HOURS}h, R={current_r:.2f}")
                    result, verified = _send_close_order(position, size, "TIMEEXIT")
                    if verified:
                        _record_close_for_circuit_breaker(
                            item, "TIMEEXIT", position, pd.Timestamp.now(tz="UTC"), _extract_filled_price(result)
                        )
                        state.pop(key, None)
                    else:
                        item["pending_exit"] = {"reason": "TIMEEXIT", "qty": size,
                                                "before_size": size, "result": result,
                                                "created_ts": time.time()}
                        state[key] = item
                    _exit_save_state(state)
                    continue

            if not item.get("tp1_done", False) and current_r >= tp1_r_eff:
                tp1_qty, _, _ = _calculate_exit_quantities(float(item.get("initial_size", size)))
                tp1_qty = min(tp1_qty, _round_qty(size))
                if tp1_qty >= MIN_QTY:
                    print(f"🎯 TP1 HIT @ {current_r:.2f}R (effective target {tp1_r_eff:.2f}R): closing {tp1_qty:.4f}")
                    result, verified = _send_close_order(position, tp1_qty, "TP1")
                    if verified:
                        _mark_exit_stage_done(item, "TP1", mark, current_r)
                        item["pending_exit"] = None
                        state[key] = item
                        _exit_save_state(state)
                    else:
                        item["pending_exit"] = {"reason": "TP1", "qty": tp1_qty,
                                                "before_size": size, "result": result,
                                                "created_ts": time.time()}
                        state[key] = item
                        _exit_save_state(state)
                    continue

            if item.get("tp1_done", False) and not item.get("tp2_done", False) and current_r >= tp2_r_eff:
                _, tp2_qty, _ = _calculate_exit_quantities(float(item.get("initial_size", size)))
                tp2_qty = min(tp2_qty, _round_qty(size))
                if tp2_qty >= MIN_QTY:
                    print(f"🎯 TP2 HIT @ {current_r:.2f}R (effective target {tp2_r_eff:.2f}R): closing {tp2_qty:.4f}")
                    result, verified = _send_close_order(position, tp2_qty, "TP2")
                    if verified:
                        _mark_exit_stage_done(item, "TP2", mark, current_r)
                        item["pending_exit"] = None
                        state[key] = item
                        _exit_save_state(state)
                    else:
                        item["pending_exit"] = {"reason": "TP2", "qty": tp2_qty,
                                                "before_size": size, "result": result,
                                                "created_ts": time.time()}
                        state[key] = item
                        _exit_save_state(state)
                    continue

            # ============================================================
            # V167.8.9-4: TRAILING tied to effective TP2 (not hard-coded 1.75)
            # ============================================================
            if TRAILING_ENABLED and item.get("tp2_done", False):
                trailing_activation_eff = max(tp2_r_eff, float(TRAILING_ACTIVATION_R))
                if not item.get("trailing_active", False) and current_r >= trailing_activation_eff:
                    item["trailing_active"] = True
                    print(
                        f"🔄 TRAILING ACTIVATED @ {current_r:.2f}R "
                        f"(threshold={trailing_activation_eff:.2f}R, "
                        f"tp2_eff={tp2_r_eff:.2f}R)"
                    )
                if item.get("trailing_active", False):
                    peak = float(item.get("trailing_peak", entry_price))
                    if TRAILING_MODE == "R_BASED":
                        trail_distance = TRAILING_DISTANCE_R * risk_distance
                    elif TRAILING_MODE == "ATR" and df is not None:
                        atr_series = calculate_15m_atr_series(df)
                        valid = atr_series.dropna() if atr_series is not None else pd.Series(dtype=float)
                        atr = float(valid.iloc[-1]) if not valid.empty else risk_distance
                        trail_distance = TRAILING_ATR_MULT * atr
                    else:
                        trail_distance = peak * TRAILING_PERCENT
                    trail_stop = peak - trail_distance if side == "LONG" else peak + trail_distance
                    hit = mark <= trail_stop if side == "LONG" else mark >= trail_stop
                    if hit and size >= MIN_QTY:
                        print(f"🛑 TRAILING STOP HIT @ {mark:.2f} (stop={trail_stop:.2f})")
                        result, verified = _send_close_order(position, size, "TRAIL")
                        if verified:
                            _record_close_for_circuit_breaker(
                                item, "TRAIL", position, pd.Timestamp.now(tz="UTC"), _extract_filled_price(result)
                            )
                            state.pop(key, None)
                        else:
                            item["pending_exit"] = {"reason": "TRAIL", "qty": size,
                                                    "before_size": size, "result": result,
                                                    "created_ts": time.time()}
                            state[key] = item
                        _exit_save_state(state)
                        continue

            if not item.get("be_armed", False) and current_r >= BREAKEVEN_TRIGGER_R:
                item["be_armed"] = True
                item["be_armed_at"] = mark
                item["be_armed_r"] = current_r
                print(f"🛡️ BE ARMED @ {current_r:.2f}R (threshold {BREAKEVEN_TRIGGER_R:.2f}R)")

            if item.get("be_armed", False):
                if side == "LONG":
                    be_trigger = entry_price * (1.0 + BREAKEVEN_FEE_BUFFER_PCT)
                    be_hit = mark <= be_trigger
                else:
                    be_trigger = entry_price * (1.0 - BREAKEVEN_FEE_BUFFER_PCT)
                    be_hit = mark >= be_trigger
                if be_hit and size >= MIN_QTY:
                    print(f"🛡️ BE EXIT: price returned to {be_trigger:.2f} "
                          f"(entry {entry_price:.2f} ± fee buffer "
                          f"{BREAKEVEN_FEE_BUFFER_PCT*100:.3f}%)")
                    result, verified = _send_close_order(position, size, "BE")
                    if verified:
                        _record_close_for_circuit_breaker(
                            item, "BE", position, pd.Timestamp.now(tz="UTC"), _extract_filled_price(result)
                        )
                        state.pop(key, None)
                    else:
                        item["pending_exit"] = {"reason": "BE", "qty": size,
                                                "before_size": size, "result": result,
                                                "created_ts": time.time()}
                        state[key] = item
                    _exit_save_state(state)
                    continue

            state[key] = item
            _exit_save_state(state)

        return bool(active)

    except KeyboardInterrupt:
        raise
    except Exception as e:
        print("EXIT MANAGER ERROR:", repr(e))
        import traceback
        traceback.print_exc()
        return bool(active) if "active" in locals() else False

# ============================================================
# MAIN (ONLINE DEMO RUNNER)
# ============================================================
def main():
    try:
        if DEMO_ONLY is not True:
            raise RuntimeError("SAFETY ERROR: DEMO_ONLY must remain True.")

        check_credentials()

        print()
        print("=" * 72)
        print("V167.8.9 PROFESSIONAL BOT  [ONLINE DEMO MODE]")
        print("  + SINGLE SOURCE OF TRUTH: _normalize_utc_timestamp")
        print("  + SYMMETRIC RISK_DISTANCE SANITY (_risk_distance_is_sane)")
        print("  + POST-FILL EXCHANGE TP/SL MODIFY (single source of truth)")
        print("  + TRAILING ACTIVATION tied to effective TP2")
        print("  + ONLINE DEMO DEFAULT (offline via V167_OFFLINE_TEST=1)")
        print("=" * 72)
        print(f"  OPERATION MODE : {OPERATION_MODE}")
        print(f"  SYSTEM ID      : {SYSTEM_ID}")
        print("=" * 72)
        print("V167.6.1 ENTRY     : FROZEN (UNCHANGED)")
        print("V167.7.8 EXIT      : API-GUARDED + HARD SL CUSHION + SAFE PENDING RECOVERY")
        print("V167.7.9 ADDITIONS : MARGIN UTILIZATION BUFFER + DETERMINISTIC REJECTION GUARD")
        print("V167.7.10 ADDITIONS: FEE-AWARE BREAKEVEN (1.00R ARM, 20 bps BUFFER)")
        print("V167.7.11 ADDITIONS: DUAL-PRICE HARD-SL + LAST_PRICE FAILSAFE + ADAPTIVE POLLING")
        print("V167.7.12 ADDITIONS: SESSION FILTER inside risk_guard() (time-based)")
        print("V167.7.13 ADDITIONS: CONDITION-AWARE Session Filter")
        print("V167.7.13.1 PATCH  : WEEX TpWorkingType/SlWorkingType → CONTRACT_PRICE")
        print("V167.7.13.2 PATCH  : LAST PRICE multi-endpoint fallback")
        print("V167.8 ADDITIONS   : DYNAMIC TP COMPRESSION (Exit Optimization)")
        print("V167.8.1 ADDITIONS : EXIT STATE RESILIENCE")
        print("V167.8.2 ADDITIONS : PER-ITERATION COMPRESSION RESTORE")
        print("V167.8.3 ADDITIONS : PERSISTENT ENTRY_ATR + RISK_DISTANCE REBUILD")
        print("V167.8.4 ADDITIONS : HISTORICAL ENTRY-ATR VERIFICATION + SAFE HOLD")
        print("V167.8.5 ADDITIONS : OBSERVATION / TELEMETRY (log-only)")
        print("V167.8.6 ADDITIONS : CIRCUIT BREAKER TIMESTAMP FIX")
        print("V167.8.8 ADDITIONS : EFFECTIVE_OPENED_AT (single source of truth)")
        print("V167.8.9 ADDITIONS : 5 fixes (see top-of-file header)")
        print("WEEX FUTURES       : DEMO")
        print("SYMBOL             :", MARKET_SYMBOL)
        print("TIMEFRAME          :", TIMEFRAME)
        print("LOOKBACK           :", f"{LOOKBACK_DAYS} DAYS")
        print("DEMO ONLY          :", DEMO_ONLY)
        print("DEMO ORDERS        : ENABLED" if OPERATION_MODE == "PRIMARY" else "DISABLED (BACKUP)")
        print("REAL ORDERS        : DISABLED")
        print("SAFETY             : DEMO ONLY / REAL ORDERS HARD-DISABLED")
        print("RISK PER TRADE     :", f"{ACCOUNT_RISK_PERCENT * 100:.1f}% of account (target)")
        print("MAX LEVERAGE       :", f"{MAX_LEVERAGE}x")
        print("MARGIN UTILIZATION :", f"MAX {MAX_MARGIN_UTILIZATION * 100:.0f}% of available balance")
        print("BREAK-EVEN         :", f"ARM @ +{BREAKEVEN_TRIGGER_R:.2f}R -> "
                                    f"EXIT @ entry ± {BREAKEVEN_FEE_BUFFER_PCT*100:.3f}% (fee-aware)")
        print("TP1                :", f"{TP1_R}R -> CLOSE {TP1_CLOSE_PCT*100:.0f}% (compressed: {TP_COMPRESSION_TP1_R}R)")
        print("TP2                :", f"{TP2_R}R -> CLOSE {TP2_CLOSE_PCT*100:.0f}% (compressed: {TP_COMPRESSION_TP2_R}R)")
        print("TP COMPRESSION     :", "ENABLED" if TP_COMPRESSION_ENABLED else "DISABLED",
              f"(ATR ratio < {TP_COMPRESSION_ATR_RATIO}, age >= {TP_COMPRESSION_MIN_AGE_HOURS}h)")
        print("ENTRY_ATR SNAPSHOT : ENABLED (file: {})".format(ENTRY_ATR_FILE))
        print("RISK_DIST REBUILD  : SYMMETRIC (min 0.5x, max 1.5x)")
        print("ATR SANITY BOUNDS  : [{:.3%}, {:.3%}]".format(ATR_SANITY_MIN_PCT, ATR_SANITY_MAX_PCT))
        print("SAFE HOLD          : ENABLED (unknown entry_atr → exchange SL/TP only)")
        print("THESIS OBSERVATION :", "ENABLED (log-only)" if THESIS_OBSERVATION_ENABLED else "DISABLED")
        print("LAST_PRICE TELEM.  :", "ENABLED (log-only)" if LAST_PRICE_TELEMETRY_ENABLED else "DISABLED")
        print("CIRCUIT BREAKER    : HARDENED (quarantine + atomic + safe-hold)")
        print("EFFECTIVE_OPENED_AT: ENABLED (STATE → POSITION → UNKNOWN)")
        print("EXCHANGE TP/SL MOD : ENABLED (post-fill reconcile)")
        print("RUNNER             :", f"Trailing (activates at max(TP2_eff, {TRAILING_ACTIVATION_R}R))")
        print("TRAILING MODE      :", TRAILING_MODE, f"(distance={TRAILING_DISTANCE_R}R)")
        print("TIME EXIT          :", "ENABLED" if TIME_EXIT_ENABLED else "DISABLED",
              f"({TIME_EXIT_HOURS}h)" if TIME_EXIT_ENABLED else "")
        print("EXIT VERIFICATION  :", "ENABLED" if EXIT_VERIFICATION_ENABLED else "DISABLED")
        print("EXCHANGE TP/SL     : FAILSAFE ONLY (SL {:.2f}R, TP {:.2f}R, {})".format(
              FAILSAFE_SL_R, FAILSAFE_TP_R, _weex_working_type()))
        print("HARD-SL WATCHDOG   : MARK OR LAST (dual-price, tolerance {:.2f})".format(HARD_SL_TRIGGER_TOLERANCE))
        print("ADAPTIVE POLLING   : TIGHT {:.2f}s | NEAR {:.2f}s | PROFIT {:.2f}s | NORMAL {:.2f}s".format(
              POLL_INTERVAL_TIGHT, POLL_INTERVAL_NEAR, POLL_INTERVAL_PROFIT, POLL_INTERVAL_NORMAL))
        print("SESSION FILTER     : {} [{}:00, {}:00) UTC (CONDITION-AWARE)".format(
              "ENABLED" if NY_SESSION_FILTER_ENABLED else "DISABLED",
              NY_SESSION_FILTER_START_HOUR_UTC, NY_SESSION_FILTER_END_HOUR_UTC))
        print("  Danger thresholds (only inside window):")
        print("    ATR Shock > {:.2f}".format(DANGER_WINDOW_ATR_SHOCK))
        print("    Body/ATR  > {:.2f}".format(DANGER_WINDOW_BODY_ATR))
        print("    3-candle range > {:.2f} ATR".format(DANGER_WINDOW_3CANDLE_ATR))
        print("CIRCUIT BREAKER    :", f"ACTIVE (max_losses={CIRCUIT_MAX_LOSSES}, cooldown={CIRCUIT_COOLDOWN_CANDLES} candles)")
        print("CONFLUENCE SCORING :", "ENABLED" if CONFLUENCE_SCORING_ENABLED else "DISABLED",
              f"(max +{CONFLUENCE_MAX_BONUS})")
        print(f"ENTRY THRESHOLD    : {ENTRY_THRESHOLD} (FROZEN)")
        print(f"MIN EDGE GAP       : {MIN_EDGE_GAP} (FROZEN)")
        print(f"MIN CONFIDENCE     : {MIN_CONFIDENCE}% (FROZEN)")
        print("=" * 72)

        save_runtime_state(status="STARTING")
        save_shared_state(status="RUNNING", mode=OPERATION_MODE)
        server_now = safe_get_server_utc()
        print()
        print("WEEX SERVER TIME:", server_now)

        local_df = load_local_dataset()
        if local_df is None:
            print("LOCAL DATA: NOT FOUND/INVALID")
            print("ACTION: INITIAL 90D BACKFILL")
            df = fetch_initial_history()
        else:
            df = sync_from_local_dataset(local_df)
            try:
                test_row = build_features(df)
            except Exception:
                test_row = None
            if test_row is None:
                print("LOCAL DATA: FEATURE BUILD FAILED")
                print("ACTION: FULL 90D BACKFILL")
                df = fetch_initial_history()

        print()
        print("Building H1/H4 feature matrix...")
        row = build_features(df)
        if row is None:
            raise RuntimeError("Unable to build initial H1/H4 features.")

        signal = final_signal(row, df)
        print_state(row, signal, df)
        print_confluence_diagnostic(df, test_d_signal(row, df))

        print(f"🔍 DEBUG: signal={signal}, type={type(signal)}")
        print(f"🔍 DEBUG: signal in LONG/SHORT? {signal in {'LONG', 'SHORT'}}")

        if signal in {"LONG", "SHORT"}:
            if OPERATION_MODE == "PRIMARY":
                print(f"🚀 ORDER TRIGGERED: {signal}")
                try:
                    place_demo_market_order(signal, row["candle"], df, row)
                except DeterministicOrderRejection as drej:
                    print("🚫 DETERMINISTIC REJECTION — NOT RETRYING SAME ORDER")
                    print(repr(drej))
                    save_runtime_state(row["candle"], status="ORDER_REJECTED")
                    save_shared_state(last_candle=row["candle"], status="ORDER_REJECTED")
            else:
                print(f"🔒 BACKUP MODE: Signal {signal} detected but orders disabled")
        else:
            print(f"💤 NO ORDER: signal={signal}")

        print_demo_balance()
        print_demo_positions()
        exit_active_position = bool(manage_position_exits(df))

        last_candle = pd.Timestamp(row["candle"])
        save_shared_state(last_candle=last_candle)

        print()
        print("=" * 72)
        print("LIVE DEMO MONITOR STARTED")
        print(f"  MODE            : {OPERATION_MODE}")
        print(f"  SYSTEM ID       : {SYSTEM_ID}")
        print("INITIAL BACKFILL: COMPLETE")
        print("MODE: INCREMENTAL 15M ONLY")
        print("WAITING FOR NEW CLOSED 15M CANDLE")
        print("=" * 72)

        last_heartbeat_time = time.time()
        last_candle_poll_time = time.time()
        exit_active_position = bool(manage_position_exits(df))

        while True:
            try:
                if exit_active_position:
                    try:
                        _state = _exit_load_state()
                        _r_for_poll = 0.0
                        if _state:
                            _first_item = next(iter(_state.values()))
                            _entry = float(_first_item.get("entry_price", 0) or 0)
                            _risk = float(_first_item.get("risk_distance", 0) or 0)
                            _side = _first_item.get("side", "")
                            if _entry > 0 and _risk > 0:
                                _mark_for_poll = get_demo_mark_price()
                                if _side == "LONG":
                                    _r_for_poll = (_mark_for_poll - _entry) / _risk
                                else:
                                    _r_for_poll = (_entry - _mark_for_poll) / _risk
                        poll_sleep = get_active_poll_interval(_r_for_poll)
                    except Exception:
                        poll_sleep = EXIT_ACTIVE_POSITION_POLL_SECONDS
                else:
                    poll_sleep = EXIT_IDLE_POLL_SECONDS

                time.sleep(poll_sleep)
                exit_active_position = bool(manage_position_exits(df))

                if OPERATION_MODE == "PRIMARY":
                    current_time = time.time()
                    if current_time - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                        save_shared_state(last_candle=last_candle, status="RUNNING", mode="PRIMARY")
                        last_heartbeat_time = current_time

                current_time = time.time()
                if current_time - last_candle_poll_time < CANDLE_POLL_SECONDS:
                    continue
                last_candle_poll_time = current_time

                try:
                    new_df, newest_closed = fetch_new_closed_candles(last_candle)
                except KeyboardInterrupt:
                    raise
                except Exception as net_err:
                    print("[NETWORK ERROR]", repr(net_err))
                    save_runtime_state(last_candle, status="NETWORK_DOWN")
                    network_wait("LIVE NETWORK RECOVERY")
                    new_df, newest_closed = fetch_new_closed_candles(last_candle)

                if new_df is None or new_df.empty:
                    print(f"[WAIT] server={safe_get_server_utc()} | last_closed={newest_closed} | tracked={last_candle}")
                    save_runtime_state(last_candle, status="WAITING")
                    continue

                old_last = last_candle
                df = append_new_candles(df, new_df)
                last_candle = pd.Timestamp(df["timestamp"].iloc[-1])
                save_runtime_state(last_candle, status="CANDLE_UPDATED")

                print()
                print(f"[NEW CLOSED CANDLE] {old_last} -> {last_candle}")
                print(f"[INCREMENTAL FETCH] {len(new_df)} candle(s)")

                row = build_features(df)
                if row is None:
                    print("FEATURE ERROR: insufficient/invalid H1/H4 data.")
                    continue

                signal = final_signal(row, df)
                print_state(row, signal, df)
                print_confluence_diagnostic(df, test_d_signal(row, df))

                print(f"🔍 DEBUG: signal={signal}, type={type(signal)}")
                print(f"🔍 DEBUG: signal in LONG/SHORT? {signal in {'LONG', 'SHORT'}}")

                if signal in {"LONG", "SHORT"}:
                    if OPERATION_MODE == "PRIMARY":
                        print(f"🚀 ORDER TRIGGERED: {signal}")
                        try:
                            place_demo_market_order(signal, row["candle"], df, row)
                        except DeterministicOrderRejection as drej:
                            print("🚫 DETERMINISTIC REJECTION — NOT RETRYING SAME ORDER")
                            print(repr(drej))
                            save_runtime_state(last_candle, status="ORDER_REJECTED")
                            save_shared_state(last_candle=last_candle, status="ORDER_REJECTED")
                    else:
                        print(f"🔒 BACKUP MODE: Signal {signal} detected but orders disabled")
                else:
                    print(f"💤 NO ORDER: signal={signal}")

                print_demo_balance()
                print_demo_positions()
                exit_active_position = bool(manage_position_exits(df))

            except KeyboardInterrupt:
                print()
                print("=" * 72)
                print("V167.8.9 DEMO BOT STOPPED")
                print("=" * 72)
                break
            except Exception as e:
                print()
                print("=" * 72)
                print("LIVE LOOP ERROR")
                print("=" * 72)
                print(repr(e))
                import traceback
                traceback.print_exc()
                print("=" * 72)
                print("Retrying in 10 seconds...")
                print("=" * 72)
                time.sleep(10)

    except Exception as e:
        print()
        print("=" * 72)
        print("🚨 FATAL ERROR - PROGRAM WILL NOT CLOSE")
        print("=" * 72)
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR: {repr(e)}")
        print("-" * 72)
        import traceback
        traceback.print_exc()
        print("=" * 72)
        print("Press ENTER to exit...")
        input()


# ============================================================
# OFFLINE SELF TEST (opt-in via V167_OFFLINE_TEST=1)
# ============================================================
def _offline_self_tests():
    results = []

    def check(name, condition):
        results.append((name, bool(condition)))

    ts = _normalize_utc_timestamp("2026-09-12T19:30:00Z")
    check("UTC timestamp normalization",
          ts is not None and ts.tzinfo is not None)

    check("Risk distance 0.50x accepted",
          _risk_distance_is_sane(500, 1000))
    check("Risk distance 1.50x accepted",
          _risk_distance_is_sane(1500, 1000))
    check("Risk distance >1.50x rejected",
          not _risk_distance_is_sane(1601, 1000))
    check("Risk distance <0.50x rejected",
          not _risk_distance_is_sane(499, 1000))

    try:
        cb = get_circuit_breaker_state()
        check("Circuit breaker state readable", isinstance(cb, dict))
    except Exception:
        check("Circuit breaker state readable", False)

    try:
        ast.parse(Path(__file__).read_text(encoding="utf-8"))
        check("Generated source AST valid", True)
    except Exception:
        check("Generated source AST valid", False)

    print("\n" + "=" * 72)
    print("V167.8.9 OFFLINE SELF TEST")
    print("=" * 72)
    passed = 0
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        passed += int(ok)
    print("-" * 72)
    print(f"RESULT: {passed}/{len(results)} tests passed")
    print("=" * 72)
    return all(ok for _, ok in results)


# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    # DEFAULT = ONLINE DEMO RUNNER (WEEX demo orders enabled).
    # Set V167_OFFLINE_TEST=1 only for local self-tests (no exchange).
    if os.getenv("V167_OFFLINE_TEST", "0").strip().lower() in {"1", "true", "yes", "on"}:
        print("=" * 72)
        print("V167.8.9 OFFLINE SELF-TEST MODE")
        print("EXCHANGE ORDERS : DISABLED")
        print("REAL ORDERS     : DISABLED")
        print("=" * 72)
        raise SystemExit(0 if _offline_self_tests() else 1)

    # Normal/default path: run the actual WEEX DEMO bot.
    main()
