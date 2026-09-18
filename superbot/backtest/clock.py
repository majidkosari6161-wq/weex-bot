"""
Deterministic virtual clock for backtesting.

The live bot calls `pd.Timestamp.now(tz="UTC")` in ~29 places. During
backtest, we must replace every such call with a controllable virtual
time so that:

    - opened_at / age_hours / thesis_age are consistent
    - circuit_breaker timestamps do not mix real and virtual time
    - trade ledger filled_at matches the simulated bar time
    - TP compression can actually trigger when age > threshold

Approach:
    Monkey-patch `pandas.Timestamp.now` with a callable that reads the
    current virtual time.

    NOTE: `datetime.datetime.now` cannot be patched in Python 3.11+
    because datetime is an immutable built-in type. The live bot only
    uses `pd.Timestamp.now(tz='UTC')`, so patching pandas is sufficient.

Usage:
    clock = VirtualClock(start_utc="2026-06-17T00:00:00Z")
    with clock.installed():
        # inside this block, every pd.Timestamp.now(tz="UTC") returns
        # the virtual time
        ...
        clock.set("2026-06-17T00:15:00Z")
        ...
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

BAR_15M = pd.Timedelta(minutes=15)
BAR_1H = pd.Timedelta(hours=1)
BAR_4H = pd.Timedelta(hours=4)


# ------------------------------------------------------------
# VirtualClock
# ------------------------------------------------------------

@dataclass
class VirtualClock:
    """
    A controllable virtual clock that can be installed over pandas.

    Thread-safety:
        Internal state is protected by a reentrant lock. However, install()
        globally patches pandas — do NOT call install() from multiple threads.

    Attributes:
        _now: current virtual UTC time.
        _installed: whether patches are currently applied.
        _lock: reentrant lock.
        _orig_pd_now: original pd.Timestamp.now (saved on first install).
    """

    _now: pd.Timestamp
    _installed: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _orig_pd_now: object = None

    # -------- construction --------

    @classmethod
    def from_iso(cls, start_utc: str | pd.Timestamp) -> "VirtualClock":
        ts = _to_utc_ts(start_utc)
        return cls(_now=ts)

    # -------- time access --------

    def now(self) -> pd.Timestamp:
        """Return current virtual time as tz-aware UTC Timestamp."""
        with self._lock:
            return self._now

    def now_iso(self) -> str:
        """Return current virtual time as ISO8601 string."""
        return self.now().isoformat()

    def epoch_ms(self) -> int:
        """Return current virtual time as ms since epoch."""
        return int(self.now().value // 1_000_000)

    def datetime(self) -> datetime:
        """Return current virtual time as stdlib datetime (UTC)."""
        return self.now().to_pydatetime()

    # -------- time mutation --------

    def set(self, when: str | pd.Timestamp | datetime) -> None:
        """Set virtual time to an absolute value."""
        with self._lock:
            ts = _to_utc_ts(when)
            if ts < self._now:
                logger.warning(
                    "VirtualClock.set going backwards: %s -> %s", self._now, ts
                )
            self._now = ts

    def advance(self, delta: pd.Timedelta | str) -> pd.Timestamp:
        """Advance virtual time by a timedelta. Returns new time."""
        with self._lock:
            d = pd.Timedelta(delta)
            if d < pd.Timedelta(0):
                raise ValueError(f"Cannot advance by negative delta: {d}")
            self._now = self._now + d
            return self._now

    def advance_bars(self, n: int, bar: pd.Timedelta = BAR_15M) -> pd.Timestamp:
        """Advance by n bars of the given size."""
        return self.advance(bar * int(n))

    # -------- patching --------

    def install(self) -> None:
        """
        Patch pd.Timestamp.now globally.

        NOTE: datetime.datetime.now cannot be patched in Python 3.11+
        because datetime is an immutable built-in type. The live bot only
        uses pd.Timestamp.now(tz='UTC'), so patching pandas is sufficient.

        Idempotent: calling install() twice is safe.
        """
        with self._lock:
            if self._installed:
                logger.debug("VirtualClock already installed; skipping")
                return
            # Save original (only once)
            if self._orig_pd_now is None:
                self._orig_pd_now = pd.Timestamp.now

            clock = self

            def _pd_now(tz=None):  # type: ignore[no-untyped-def]
                if tz is None:
                    # Live bot only ever calls with tz="UTC" but be safe:
                    return clock.now().tz_localize(None)
                return clock.now().tz_convert(tz)

            # Type: ignore because we're deliberately replacing a classmethod.
            pd.Timestamp.now = staticmethod(_pd_now)  # type: ignore[assignment]
            self._installed = True
            logger.info("VirtualClock installed at %s", self._now)

    def uninstall(self) -> None:
        """Restore original pd.Timestamp.now."""
        with self._lock:
            if not self._installed:
                return
            if self._orig_pd_now is not None:
                pd.Timestamp.now = self._orig_pd_now  # type: ignore[assignment]
            self._installed = False
            logger.info("VirtualClock uninstalled")

    @property
    def is_installed(self) -> bool:
        return self._installed

    @contextmanager
    def installed(self) -> Iterator["VirtualClock"]:
        """Context manager: install on enter, uninstall on exit."""
        self.install()
        try:
            yield self
        finally:
            self.uninstall()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _to_utc_ts(value: str | pd.Timestamp | datetime) -> pd.Timestamp:
    """Convert str/Timestamp/datetime to tz-aware UTC Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def utc_now() -> pd.Timestamp:
    """
    Safe accessor for current time.

    If a VirtualClock is installed, returns virtual time.
    Otherwise, returns real UTC time.

    Use this in library code instead of pd.Timestamp.now(tz='UTC')
    when you need to be explicit about backtest-awareness.
    """
    return pd.Timestamp.now(tz="UTC")


# ------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    print("Before install:")
    real_now = pd.Timestamp.now(tz="UTC")
    print(f"  pd.Timestamp.now(UTC)  = {real_now}")
    print(f"  datetime.now(UTC)      = {datetime.now(timezone.utc)}")

    clock = VirtualClock.from_iso("2026-01-01T00:00:00Z")

    with clock.installed():
        print("\nInside install (virtual):")
        print(f"  pd.Timestamp.now(UTC)  = {pd.Timestamp.now(tz='UTC')}")
        print(f"  datetime.now(UTC)      = {datetime.now(timezone.utc)}  (not patched, OK)")
        clock.advance_bars(1)
        print(f"  after +1 bar           = {pd.Timestamp.now(tz='UTC')}")
        clock.advance("4h")
        print(f"  after +4h              = {pd.Timestamp.now(tz='UTC')}")
        clock.set("2026-06-15T12:34:56Z")
        print(f"  after set(2026-06-15)  = {pd.Timestamp.now(tz='UTC')}")

    print("\nAfter uninstall (real again):")
    print(f"  pd.Timestamp.now(UTC)  = {pd.Timestamp.now(tz='UTC')}")
    assert pd.Timestamp.now(tz="UTC") >= real_now, "Time went backwards!"
    print("\nOK: clock round-trip verified")