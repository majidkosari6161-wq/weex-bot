"""
Live ML filter for SuperBot.

This module is the ONLY integration point between the ML layer and the
live bot. It reads the trained model and produces a decision for each
candidate signal.

Principles:
    - ADDITIVE: does not modify, override, or re-derive any signal.
    - FAIL-OPEN: on any error, allows the trade with full size.
    - BOUNDED: size multipliers are clamped to [0.25, 1.5].
    - LOGGED: every decision is recorded with p_win and top SHAP features.

Policies:
    advisory_only     — log only; no impact on execution
    reject_if_below   — if p_win < reject_threshold, skip the trade
    reduce_if_below   — if p_win < reduce_threshold, halve size
    boost_if_above    — if p_win > boost_threshold, multiply size (capped)

Multiple policies may be active; they are evaluated in this order:
    reject_if_below > reduce_if_below > boost_if_above
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from superbot.ml.feature_builder import build_feature_vector, ALL_FEATURES

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

ACTION_ALLOW = "ALLOW"
ACTION_REJECT = "REJECT"
ACTION_REDUCE = "REDUCE"
ACTION_BOOST = "BOOST"
ACTION_ADVISORY = "ADVISORY"

VALID_POLICIES = {
    "advisory_only",
    "reject_if_below",
    "reduce_if_below",
    "boost_if_above",
}

# Hard clamp on size multipliers
SIZE_MULT_MIN = 0.25
SIZE_MULT_MAX = 1.50


# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------

@dataclass
class FilterDecision:
    """Result of evaluating one candidate signal."""
    action: str                        # ALLOW | REJECT | REDUCE | BOOST | ADVISORY
    p_win: float
    size_multiplier: float = 1.0
    reason: str = ""
    policy: str = ""
    top_features: dict[str, float] = field(default_factory=dict)
    model_loaded: bool = False
    error: str | None = None

    @property
    def blocked(self) -> bool:
        return self.action == ACTION_REJECT

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "p_win": self.p_win,
            "size_multiplier": self.size_multiplier,
            "reason": self.reason,
            "policy": self.policy,
            "top_features": self.top_features,
            "model_loaded": self.model_loaded,
            "error": self.error,
        }

    def log_line(self) -> str:
        base = f"ML[{self.action}] p_win={self.p_win:.3f} mult={self.size_multiplier:.2f}"
        if self.reason:
            base += f" reason={self.reason}"
        if self.error:
            base += f" error={self.error}"
        return base


# ------------------------------------------------------------
# MLFilter
# ------------------------------------------------------------

class MLFilter:
    """
    Live ML filter that gates / sizes candidate signals.

    Loaded once at startup, kept in memory, and called per signal.

    Usage:
        f = MLFilter.from_config(config)
        decision = f.evaluate(signal="SHORT", row=row, df=df_window)
        if decision.blocked:
            return
        effective_risk *= decision.size_multiplier
    """

    def __init__(
        self,
        model: Any | None,
        policy: str,
        reject_threshold: float = 0.35,
        reduce_threshold: float = 0.45,
        boost_threshold: float = 0.70,
        fail_safe_allow_on_error: bool = True,
        feature_names: list[str] | None = None,
        top_k_features: int = 3,
    ) -> None:
        self.model = model
        self.policy = policy
        self.reject_threshold = float(reject_threshold)
        self.reduce_threshold = float(reduce_threshold)
        self.boost_threshold = float(boost_threshold)
        self.fail_safe_allow_on_error = bool(fail_safe_allow_on_error)
        self.feature_names = list(feature_names) if feature_names else list(ALL_FEATURES)
        self.top_k_features = int(top_k_features)
        self._loaded = model is not None

        if self.policy not in VALID_POLICIES:
            logger.warning("Unknown policy %r; defaulting to advisory_only", self.policy)
            self.policy = "advisory_only"

        if not self._loaded:
            if self.fail_safe_allow_on_error:
                logger.warning("MLFilter: no model loaded; will allow all trades")
            else:
                logger.error("MLFilter: no model loaded and fail_safe disabled — all trades will be rejected")

    # ==========================================================
    # Construction
    # ==========================================================

    @classmethod
    def from_config(cls, config: dict, strict: bool = False) -> "MLFilter":
        """
        Build a filter from config.

        If ml.enabled is False, returns a stub filter in advisory_only mode
        with no model.
        """
        ml = config.get("ml", {})
        if not ml.get("enabled", False):
            logger.info("MLFilter: ml.enabled=False; running in advisory_only with no model")
            return cls(
                model=None,
                policy="advisory_only",
                fail_safe_allow_on_error=True,
            )

        model_path = Path(ml.get("model_path", ""))
        meta_path = Path(ml.get("metadata_path", ""))

        if not model_path.exists():
            msg = f"Model not found at {model_path}"
            if strict:
                raise FileNotFoundError(msg)
            logger.warning("%s — MLFilter will be fail-open", msg)
            return cls(
                model=None,
                policy=ml.get("policy", "advisory_only"),
                reject_threshold=ml.get("reject_threshold", 0.35),
                reduce_threshold=ml.get("reduce_threshold", 0.45),
                boost_threshold=ml.get("boost_threshold", 0.70),
                fail_safe_allow_on_error=ml.get("fail_safe_allow_on_error", True),
            )

        try:
            from superbot.ml.trainer import Trainer
            model = Trainer.load(model_path.parent)
            logger.info("MLFilter loaded model from %s", model_path.parent)
        except Exception as e:
            if strict:
                raise
            logger.error("Failed to load model: %r — MLFilter will be fail-open", e)
            return cls(
                model=None,
                policy=ml.get("policy", "advisory_only"),
                fail_safe_allow_on_error=ml.get("fail_safe_allow_on_error", True),
            )

        feature_names = list(model.feature_names) if model.feature_names else list(ALL_FEATURES)

        return cls(
            model=model,
            policy=ml.get("policy", "advisory_only"),
            reject_threshold=ml.get("reject_threshold", 0.35),
            reduce_threshold=ml.get("reduce_threshold", 0.45),
            boost_threshold=ml.get("boost_threshold", 0.70),
            fail_safe_allow_on_error=ml.get("fail_safe_allow_on_error", True),
            feature_names=feature_names,
        )

    # ==========================================================
    # Inference
    # ==========================================================

    def evaluate(
        self,
        signal: str,
        row: dict[str, Any],
        df_15m: pd.DataFrame,
    ) -> FilterDecision:
        """
        Evaluate a single candidate signal.

        Args:
            signal: "LONG" | "SHORT"
            row: live bot's row dict.
            df_15m: 15m OHLCV up to and including the current bar.

        Returns:
            FilterDecision; on any error, action=ALLOW with full size.
        """
        # Advisory-only fast path
        if self.policy == "advisory_only":
            p = self._safe_predict(row, df_15m)
            return FilterDecision(
                action=ACTION_ADVISORY,
                p_win=p,
                size_multiplier=1.0,
                reason="advisory_only",
                policy=self.policy,
                model_loaded=self._loaded,
            )

        # No model → fail-open
        if self.model is None:
            if self.fail_safe_allow_on_error:
                return FilterDecision(
                    action=ACTION_ALLOW,
                    p_win=float("nan"),
                    size_multiplier=1.0,
                    reason="no_model_fail_open",
                    policy=self.policy,
                    model_loaded=False,
                )
            return FilterDecision(
                action=ACTION_REJECT,
                p_win=float("nan"),
                size_multiplier=0.0,
                reason="no_model_fail_closed",
                policy=self.policy,
                model_loaded=False,
            )

        # Build features
        try:
            fv = build_feature_vector(row, df_15m, strict=True)
        except Exception as e:
            return self._fail_open(f"feature_build_failed: {e!r}")

        # Predict
        try:
            X = self._to_array(fv.values)
            p_win = float(self.model.predict_proba(X)[0])
            if not np.isfinite(p_win):
                raise ValueError(f"non-finite p_win: {p_win}")
            p_win = float(np.clip(p_win, 0.0, 1.0))
        except Exception as e:
            return self._fail_open(f"inference_failed: {e!r}")

        # Evaluate policy chain
        return self._apply_policy(p_win, fv.values)

    def evaluate_batch(
        self,
        signals: list[dict[str, Any]],
        df_15m: pd.DataFrame,
    ) -> list[FilterDecision]:
        """
        Batch evaluation for backtest efficiency.

        Each element of `signals` must have:
            {"signal": "LONG"/"SHORT", "row": row_dict, "bar_ts": ts}
        """
        out: list[FilterDecision] = []
        for s in signals:
            bar_ts = pd.Timestamp(s["bar_ts"])
            df_window = df_15m.loc[:bar_ts]
            out.append(self.evaluate(s["signal"], s["row"], df_window))
        return out

    # ==========================================================
    # Internal
    # ==========================================================

    def _safe_predict(self, row: dict[str, Any], df_15m: pd.DataFrame) -> float:
        """Return p_win, or NaN on any error. Never raises."""
        if self.model is None:
            return float("nan")
        try:
            fv = build_feature_vector(row, df_15m, strict=True)
            X = self._to_array(fv.values)
            p = float(self.model.predict_proba(X)[0])
            return float(np.clip(p, 0.0, 1.0)) if np.isfinite(p) else float("nan")
        except Exception:
            return float("nan")

    def _apply_policy(
        self,
        p_win: float,
        feature_values: dict[str, float],
    ) -> FilterDecision:
        top = self._top_features(feature_values)

        # 1) reject
        if self.policy == "reject_if_below" and p_win < self.reject_threshold:
            return FilterDecision(
                action=ACTION_REJECT,
                p_win=p_win,
                size_multiplier=0.0,
                reason=f"p_win < reject_threshold ({self.reject_threshold:.2f})",
                policy=self.policy,
                top_features=top,
                model_loaded=True,
            )

        # 2) reduce
        if self.policy == "reduce_if_below" and p_win < self.reduce_threshold:
            mult = 0.5
            return FilterDecision(
                action=ACTION_REDUCE,
                p_win=p_win,
                size_multiplier=mult,
                reason=f"p_win < reduce_threshold ({self.reduce_threshold:.2f})",
                policy=self.policy,
                top_features=top,
                model_loaded=True,
            )

        # 3) boost
        if self.policy == "boost_if_above" and p_win > self.boost_threshold:
            mult = min(SIZE_MULT_MAX, 1.5)
            return FilterDecision(
                action=ACTION_BOOST,
                p_win=p_win,
                size_multiplier=mult,
                reason=f"p_win > boost_threshold ({self.boost_threshold:.2f})",
                policy=self.policy,
                top_features=top,
                model_loaded=True,
            )

        return FilterDecision(
            action=ACTION_ALLOW,
            p_win=p_win,
            size_multiplier=1.0,
            reason="within bounds",
            policy=self.policy,
            top_features=top,
            model_loaded=True,
        )

    def _top_features(self, values: dict[str, float]) -> dict[str, float]:
        """
        Return top-K features by absolute value.

        This is a lightweight proxy for SHAP; for exact attribution, use
        superbot.ml.explain (Phase 2.5).
        """
        try:
            items = [(k, float(v)) for k, v in values.items() if np.isfinite(v)]
            items.sort(key=lambda kv: -abs(kv[1]))
            return dict(items[: self.top_k_features])
        except Exception:
            return {}

    def _to_array(self, values: dict[str, float]) -> np.ndarray:
        """
        Convert a values dict to a numpy array in the canonical feature order.

        If a feature is missing, raises (which triggers fail-open).
        Extra features are ignored with a warning.
        """
        missing = [f for f in self.feature_names if f not in values]
        if missing:
            raise ValueError(f"missing features: {missing[:5]}...")
        extra = [f for f in values if f not in self.feature_names]
        if extra:
            logger.debug("Ignoring %d extra features: %s", len(extra), extra[:5])
        arr = np.array([values[f] for f in self.feature_names], dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("non-finite feature values")
        return arr.reshape(1, -1)

    def _fail_open(self, reason: str) -> FilterDecision:
        if self.fail_safe_allow_on_error:
            return FilterDecision(
                action=ACTION_ALLOW,
                p_win=float("nan"),
                size_multiplier=1.0,
                reason=reason,
                policy=self.policy,
                model_loaded=self._loaded,
                error=reason,
            )
        return FilterDecision(
            action=ACTION_REJECT,
            p_win=float("nan"),
            size_multiplier=0.0,
            reason=reason,
            policy=self.policy,
            model_loaded=self._loaded,
            error=reason,
        )

    # ==========================================================
    # Status
    # ==========================================================

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def status_line(self) -> str:
        return (
            f"MLFilter(policy={self.policy}, loaded={self._loaded}, "
            f"reject<{self.reject_threshold:.2f}, reduce<{self.reduce_threshold:.2f}, "
            f"boost>{self.boost_threshold:.2f}, fail_open={self.fail_safe_allow_on_error})"
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
        print("Usage: python -m superbot.ml.filter <config.yaml>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from superbot.backtest.data_loader import load_market_data
    from superbot.backtest.engine import load_bot_module

    md = load_market_data(cfg["paths"]["data_csv"], use_cache=False)
    bot = load_bot_module(cfg["paths"]["bot_file"])
    row = bot.build_features(md.df_15m)
    if row is None:
        print("Could not build features")
        sys.exit(1)

    f = MLFilter.from_config(cfg)
    print(f.status_line())
    d = f.evaluate("SHORT", row, md.df_15m)
    print(d.log_line())
    print(json.dumps(d.to_dict(), indent=2, default=str))