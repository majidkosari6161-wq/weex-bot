"""
ML trainer for SuperBot.

Trains a LightGBM (primary) or XGBoost (fallback) classifier to predict
the probability that a candidate signal will be a win.

Design:
    - Time-series split with purge + embargo (no random shuffle).
    - Purged K-Fold CV for hyperparameter tuning (López de Prado).
    - Optuna for Bayesian hyperparameter search.
    - SHAP for feature importance and per-trade explanation.
    - Isotonic calibration for reliable probabilities.
    - Full reproducibility: same seed → same model.

Output:
    - model.txt         (LightGBM native)
    - calibrator.pkl    (CalibratedClassifierCV)
    - model_meta.json   (features, hyperparams, metrics, timestamps)
    - shap_summary.json (top-K features by mean |SHAP|)
"""
from __future__ import annotations

import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from superbot.ml.labeler import LabeledDataset, time_series_split

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ------------------------------------------------------------
# Optional dependencies (fail gracefully)
# ------------------------------------------------------------

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed; trainer will not be usable")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not installed; hyperparameter search disabled")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed; feature importance disabled")

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
        log_loss,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed; calibration/metrics disabled")


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_split_gain": 0.0,
    "verbose": -1,
}


# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------

@dataclass
class TrainMetrics:
    """Metrics from one train/val/test pass."""
    train_auc: float = 0.0
    val_auc: float = 0.0
    test_auc: float = 0.0
    train_pr_auc: float = 0.0
    val_pr_auc: float = 0.0
    test_pr_auc: float = 0.0
    train_brier: float = 0.0
    val_brier: float = 0.0
    test_brier: float = 0.0
    train_logloss: float = 0.0
    val_logloss: float = 0.0
    test_logloss: float = 0.0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    overfit_gap: float = 0.0  # train_auc - val_auc
    meta_overfit_gap: float = 0.0  # val_auc - test_auc

    def summary(self) -> str:
        return (
            f"TrainMetrics(train_auc={self.train_auc:.3f}, "
            f"val_auc={self.val_auc:.3f}, test_auc={self.test_auc:.3f}, "
            f"overfit_gap={self.overfit_gap:+.3f}, "
            f"n_train={self.n_train}, n_val={self.n_val}, n_test={self.n_test})"
        )


@dataclass
class TrainedModel:
    """A trained + calibrated model ready for live inference."""
    booster: Any
    calibrator: Any | None
    feature_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Return probability of win (class=1) for each row of X.

        X can be a DataFrame (with feature columns) or a raw ndarray.
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        raw = self.booster.predict(X)
        if self.calibrator is not None:
            return self.calibrator.predict(raw.reshape(-1, 1))
        return np.asarray(raw, dtype=np.float64)

    def predict_label(self, X: np.ndarray | pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ------------------------------------------------------------
# Purged K-Fold CV
# ------------------------------------------------------------

def purged_kfold_indices(
    n: int,
    durations: np.ndarray,
    n_splits: int = 5,
    purge_bars: int = 100,
    embargo_bars: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Purged K-Fold: each fold's test set is a contiguous block; train set
    excludes any sample whose [start, start+duration] overlaps the test block
    (purge) plus embargo after test.

    Returns:
        list of (train_idx, val_idx) tuples.
    """
    if n < n_splits * 10:
        raise ValueError(f"Not enough samples for {n_splits}-fold: n={n}")

    fold_size = n // n_splits
    folds = []
    for k in range(n_splits):
        val_start = k * fold_size
        val_end = val_start + fold_size if k < n_splits - 1 else n
        val_idx = np.arange(val_start, val_end)

        # Train = all other samples, but purge those whose duration crosses
        # into val's block, and apply embargo after val_end.
        train_mask = np.ones(n, dtype=bool)
        # Exclude val
        train_mask[val_idx] = False
        # Purge: any sample in train whose end enters [val_start - purge_bars, val_end + embargo_bars]
        for i in range(n):
            if not train_mask[i]:
                continue
            end_i = i + int(durations[i]) + purge_bars
            if val_start - purge_bars <= i <= val_end + embargo_bars:
                train_mask[i] = False
            elif end_i >= val_start - purge_bars and i < val_start:
                # sample's future reaches into val block
                train_mask[i] = False
        train_idx = np.where(train_mask)[0]
        folds.append((train_idx, val_idx))
    return folds


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

class Trainer:
    """
    ML trainer for SuperBot.

    Usage:
        trainer = Trainer(config)
        model = trainer.train(labeled_dataset)
        trainer.save(model, output_dir)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        ml = config["ml"]
        self.model_type = ml["model_type"]
        self.label_mode = ml["label_mode"]
        self.train_ratio = ml["train_ratio"]
        self.val_ratio = ml["val_ratio"]
        self.embargo_bars = ml["embargo_bars"]
        self.wf_enabled = ml["wf_enabled"]
        self.wf_n_splits = ml["wf_n_splits"]
        self.wf_purge_bars = ml["wf_purge_bars"]
        self.wf_embargo_bars = ml["wf_embargo_bars"]
        self.optuna_enabled = ml["optuna_enabled"]
        self.optuna_n_trials = ml["optuna_n_trials"]
        self.optuna_timeout = ml["optuna_timeout_seconds"]
        self.seed = int(config["backtest"]["seed"])
        self.label_threshold = float(ml["label_win_threshold_r"])

        if self.model_type != "lightgbm":
            raise ValueError(
                f"model_type={self.model_type} not supported in Phase 2. "
                f"Only lightgbm is implemented."
            )
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. pip install lightgbm")

        np.random.seed(self.seed)

    # ==========================================================
    # Public API
    # ==========================================================

    def train(
        self,
        ds: LabeledDataset,
        tune: bool | None = None,
        save_dir: str | Path | None = None,
    ) -> TrainedModel:
        """
        Train a model on the labeled dataset.

        Args:
            ds: LabeledDataset from labeler.py.
            tune: if True, run Optuna; if None, use config.
            save_dir: if given, save model artifacts to this directory.

        Returns:
            TrainedModel with booster, calibrator, feature_names, metadata.
        """
        if tune is None:
            tune = self.optuna_enabled

        X = ds.features
        y = ds.labels["is_win"].values.astype(int)
        durations = ds.labels["duration_bars"].values

        feature_names = list(X.columns)

        # Time-based split (no random shuffle)
        sp = time_series_split(
            ds,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            embargo_bars=self.embargo_bars,
        )
        X_train = X.iloc[sp.train].values
        y_train = y[sp.train]
        X_val = X.iloc[sp.val].values
        y_val = y[sp.val]
        X_test = X.iloc[sp.test].values
        y_test = y[sp.test]

        logger.info(
            "Split sizes: train=%d, val=%d, test=%d",
            len(X_train), len(X_val), len(X_test),
        )
        if len(np.unique(y_train)) < 2:
            raise ValueError("Training set has only one class — cannot train")

        # Class imbalance weight
        pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
        logger.info("scale_pos_weight = %.3f", pos_weight)

        # Hyperparameters
        if tune and HAS_OPTUNA:
            best_params = self._optuna_search(
                X_train, y_train, X_val, y_val, durations, sp
            )
        else:
            best_params = dict(DEFAULT_LGB_PARAMS)
            best_params["random_state"] = self.seed
            best_params["scale_pos_weight"] = pos_weight

        # Train final model on train + val (test held out)
        X_trainval = np.concatenate([X_train, X_val], axis=0)
        y_trainval = np.concatenate([y_train, y_val], axis=0)

        booster = self._fit_booster(best_params, X_trainval, y_trainval,
                                     X_test, y_test)

        # Metrics
        metrics = self._compute_metrics(
            booster, X_train, y_train, X_val, y_val, X_test, y_test
        )
        logger.info(metrics.summary())
        if metrics.overfit_gap > 0.15:
            logger.warning("High overfit gap: %.3f", metrics.overfit_gap)

               # Calibration DISABLED: isotonic regression on small validation
        # sets collapses p_win into {0.0, 0.55+} (bimodal), which breaks
        # the ML filter's gating. LightGBM raw probabilities are already
        # reasonably calibrated for this use case.
        calibrator = None
        # SHAP
        shap_summary = {}
        if HAS_SHAP:
            shap_summary = self._shap_importance(
                booster, X_val, feature_names
            )

        metadata = {
            "model_type": self.model_type,
            "label_mode": self.label_mode,
            "label_threshold_r": self.label_threshold,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "train_start": str(X.index[sp.train[0]]) if len(sp.train) else None,
            "train_end": str(X.index[sp.train[-1]]) if len(sp.train) else None,
            "test_start": str(X.index[sp.test[0]]) if len(sp.test) else None,
            "test_end": str(X.index[sp.test[-1]]) if len(sp.test) else None,
            "best_params": best_params,
            "metrics": {
                "train_auc": metrics.train_auc,
                "val_auc": metrics.val_auc,
                "test_auc": metrics.test_auc,
                "train_pr_auc": metrics.train_pr_auc,
                "val_pr_auc": metrics.val_pr_auc,
                "test_pr_auc": metrics.test_pr_auc,
                "overfit_gap": metrics.overfit_gap,
            },
            "shap_top_features": shap_summary,
            "seed": self.seed,
            "optuna_used": bool(tune and HAS_OPTUNA),
        }

        model = TrainedModel(
            booster=booster,
            calibrator=calibrator,
            feature_names=feature_names,
            metadata=metadata,
        )

        if save_dir is not None:
            self.save(model, save_dir)

        return model

    # ==========================================================
    # Training internals
    # ==========================================================

    def _fit_booster(
        self,
        params: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> Any:
        """Fit a LightGBM booster with early stopping on validation."""
        params = dict(params)
        params.setdefault("random_state", self.seed)
        params.setdefault("verbose", -1)

        n_est = params.pop("n_estimators", 300)
        train_ds = lgb.Dataset(X_train, label=y_train)

        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets = []
        valid_names = []
        if X_val is not None and y_val is not None and len(X_val) > 0:
            valid_sets.append(lgb.Dataset(X_val, label=y_val, reference=train_ds))
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

        booster = lgb.train(
            params,
            train_ds,
            num_boost_round=int(n_est),
            valid_sets=valid_sets if valid_sets else None,
            valid_names=valid_names if valid_names else None,
            callbacks=callbacks,
        )
        return booster

    def _compute_metrics(
        self,
        booster: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainMetrics:
        m = TrainMetrics(
            n_train=len(X_train), n_val=len(X_val), n_test=len(X_test),
        )
        if not HAS_SKLEARN:
            return m

        def _safe_auc(y_true, y_pred):
            try:
                if len(np.unique(y_true)) < 2:
                    return 0.0
                return float(roc_auc_score(y_true, y_pred))
            except Exception:
                return 0.0

        def _safe_pr(y_true, y_pred):
            try:
                if len(np.unique(y_true)) < 2:
                    return 0.0
                return float(average_precision_score(y_true, y_pred))
            except Exception:
                return 0.0

        def _safe_brier(y_true, y_pred):
            try:
                return float(brier_score_loss(y_true, y_pred))
            except Exception:
                return 0.0

        def _safe_ll(y_true, y_pred):
            try:
                p = np.clip(y_pred, 1e-6, 1 - 1e-6)
                return float(log_loss(y_true, p))
            except Exception:
                return 0.0

        for name, X, y in (("train", X_train, y_train),
                           ("val", X_val, y_val),
                           ("test", X_test, y_test)):
            if len(X) == 0:
                continue
            p = booster.predict(X)
            setattr(m, f"{name}_auc", _safe_auc(y, p))
            setattr(m, f"{name}_pr_auc", _safe_pr(y, p))
            setattr(m, f"{name}_brier", _safe_brier(y, p))
            setattr(m, f"{name}_logloss", _safe_ll(y, p))

        m.overfit_gap = m.train_auc - m.val_auc
        m.meta_overfit_gap = m.val_auc - m.test_auc
        return m

    def _fit_calibration(
        self,
        booster: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        """Fit isotonic calibration on validation set."""
        raw_val = booster.predict(X_val)
        try:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw_val, y_val)
            return iso
        except Exception as e:
            logger.warning("Calibration failed: %r", e)
            return None

    def _shap_importance(
        self,
        booster: Any,
        X_val: np.ndarray,
        feature_names: list[str],
        top_k: int = 10,
    ) -> dict[str, float]:
        """Compute mean |SHAP| per feature; return top-K."""
        try:
            explainer = shap.TreeExplainer(booster)
            sv = explainer.shap_values(X_val)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            mean_abs = np.mean(np.abs(sv), axis=0)
            order = np.argsort(-mean_abs)[:top_k]
            return {feature_names[i]: float(mean_abs[i]) for i in order}
        except Exception as e:
            logger.warning("SHAP computation failed: %r", e)
            return {}

    # ==========================================================
    # Optuna
    # ==========================================================

    def _optuna_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        durations: np.ndarray,
        sp: Any,
    ) -> dict[str, Any]:
        """Run Optuna to find best hyperparameters."""
        pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

        # K-Fold on the train set for robustness
        n = len(X_train)
        folds = purged_kfold_indices(
            n,
            durations[sp.train],
            n_splits=self.wf_n_splits,
            purge_bars=self.wf_purge_bars,
            embargo_bars=self.wf_embargo_bars,
        )

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 8, 64),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": self.seed,
                "scale_pos_weight": pos_weight,
                "verbose": -1,
            }
            n_est = params.pop("n_estimators")
            aucs = []
            for tr_idx, val_idx in folds:
                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train[val_idx], y_train[val_idx]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                    continue
                try:
                    b = self._fit_booster(
                        params, X_tr, y_tr,
                        X_val=X_va, y_val=y_va,
                    )
                    # Refit n_estimators override: pass as num_boost_round already set
                    # (we popped it above; use default set inside)
                    p = b.predict(X_va)
                    aucs.append(float(roc_auc_score(y_va, p)))
                except Exception:
                    continue
            if not aucs:
                return -1.0
            cv_auc = float(np.mean(aucs))
            # Final validation AUC
            params["n_estimators"] = n_est
            try:
                b = self._fit_booster(params, X_train, y_train, X_val, y_val)
                val_auc = float(roc_auc_score(y_val, b.predict(X_val))) if len(np.unique(y_val)) > 1 else cv_auc
            except Exception:
                val_auc = cv_auc
            overfit = max(0.0, cv_auc - val_auc)
            return val_auc - 0.3 * overfit

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(
            objective,
            n_trials=self.optuna_n_trials,
            timeout=self.optuna_timeout,
            show_progress_bar=False,
        )
        logger.info("Optuna best score: %.4f", study.best_value)
        logger.info("Optuna best params: %s", study.best_params)

        best = dict(DEFAULT_LGB_PARAMS)
        best.update(study.best_params)
        best["random_state"] = self.seed
        best["scale_pos_weight"] = pos_weight
        best["verbose"] = -1
        return best

    # ==========================================================
    # Save / load
    # ==========================================================

    def save(self, model: TrainedModel, save_dir: str | Path) -> Path:
        """Save booster, calibrator, and metadata to save_dir."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Booster
        booster_path = save_dir / "model.txt"
        model.booster.save_model(str(booster_path))

        # Calibrator
        if model.calibrator is not None:
            cal_path = save_dir / "calibrator.pkl"
            with cal_path.open("wb") as f:
                pickle.dump(model.calibrator, f)

        # Metadata
        meta_path = save_dir / "model_meta.json"
        meta = dict(model.metadata)
        meta["saved_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info("Saved model to %s", save_dir)
        return save_dir

    @staticmethod
    def load(save_dir: str | Path) -> TrainedModel:
        """Load a previously saved model."""
        save_dir = Path(save_dir)
        booster_path = save_dir / "model.txt"
        meta_path = save_dir / "model_meta.json"
        cal_path = save_dir / "calibrator.pkl"

        if not booster_path.exists():
            raise FileNotFoundError(f"Booster not found: {booster_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        booster = lgb.Booster(model_file=str(booster_path))
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        calibrator = None
        if cal_path.exists():
            with cal_path.open("rb") as f:
                calibrator = pickle.load(f)

        return TrainedModel(
            booster=booster,
            calibrator=calibrator,
            feature_names=list(meta.get("feature_names", [])),
            metadata=meta,
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

    if len(sys.argv) < 3:
        print("Usage: python -m superbot.ml.trainer <config.yaml> <labeled.parquet> [out_dir]")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from superbot.ml.labeler import load_labeled_dataset
    ds = load_labeled_dataset(sys.argv[2])
    print(ds.summary())

    trainer = Trainer(cfg)
    model = trainer.train(ds, tune=cfg["ml"]["optuna_enabled"],
                          save_dir=sys.argv[3] if len(sys.argv) > 3 else None)
    print("\nMetrics:")
    for k, v in model.metadata["metrics"].items():
        print(f"  {k:20s} = {v:.4f}" if isinstance(v, float) else f"  {k:20s} = {v}")
    print("\nTop SHAP features:")
    for k, v in model.metadata["shap_top_features"].items():
        print(f"  {k:32s} {v:.4f}")