"""PID LightGBM configuration, training, and evaluation logic.

Data loading, plots, artifact writing, and command-line orchestration remain in
the analysis entry point.  In particular, ``subsample`` is preserved exactly as
configured by the historical workflow: because ``subsample_freq`` is not set,
LightGBM keeps its default value of zero and row subsampling is inactive.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import paths, tabular
from .smearing import TimeSmearing

LOGGER = logging.getLogger("pid_lgbm")
CLASS_NAMES = {"pi": r"$\pi^-$", "proton": "p", "kaon": r"$K^0_L$"}

__all__ = [
    "CLASS_NAMES",
    "Config",
    "build_classifier",
    "evaluate",
    "search_hyperparameters",
    "set_global_seeds",
    "train",
]


@dataclass
class Config:
    """Everything that defines a historical PID BDT run."""

    tree_name: str = "paramsTree"
    pdg_col: str = "particlePDG"
    particles: Sequence[str] = field(default_factory=lambda: list(paths.active_particles()))
    feature_cols: Sequence[str] = field(
        default_factory=lambda: [
            "Thr1",
            "Thr2",
            "Thr3",
            "Begin",
            "Radius",
            "Density",
            "NClusters",
            "Zbary",
            "Zrms",
            "PctHitsFirst10",
            "PlanesWithClusmore2",
            "AvgClustSize",
            "MaxClustSize",
            "lambda1",
            "lambda2",
            "tMin",
            "tMax",
            "tMean",
            "Nmax",
            "z0_fit",
            "Xmax",
            "lambda",
            "nTrackSegments",
            "eccentricity3D",
        ]
    )
    test_size: float = 0.20
    val_size: float = 0.10
    random_state: int = 42
    exclude_holdout: bool = False
    use_smote: bool = True
    class_weight: str | None = None
    smear_sigma_ns: float | None = 0.1
    smear_columns: Sequence[str] = ("tMin", "tMax", "tMean")
    learning_rate: float = 0.01
    n_estimators: int = 1000
    num_leaves: int = 63
    max_depth: int = -1
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    min_child_samples: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    grid_search: bool = False
    param_grid: dict[str, list] = field(
        default_factory=lambda: {
            "clf__num_leaves": [31, 63, 127],
            "clf__max_depth": [-1, 10, 20],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__n_estimators": [500, 1000],
        }
    )

    @property
    def target_pdg(self) -> list[int]:
        return [paths.PARTICLE_PDG[p] for p in self.particles]

    @property
    def labels(self) -> list[str]:
        return [CLASS_NAMES.get(p, p) for p in self.particles]

    def smearing(self) -> TimeSmearing:
        return TimeSmearing(
            sigma_ns=self.smear_sigma_ns,
            columns=tuple(self.smear_columns),
            seed=self.random_state,
        )

    def uses_timing(self) -> bool:
        return any(c.startswith("t") and c in self.feature_cols for c in self.smear_columns)

    def loggable(self) -> dict[str, object]:
        out: dict[str, object] = {
            "particles": ";".join(self.particles),
            "tree_name": self.tree_name,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "use_smote": int(self.use_smote),
            "exclude_holdout": int(self.exclude_holdout),
            "class_weight": str(self.class_weight),
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "early_stopping_rounds": self.early_stopping_rounds,
            "grid_search": int(self.grid_search),
            "timing_in_features": int(self.uses_timing()),
        }
        out.update(self.smearing().as_dict())
        return out


def set_global_seeds(seed: int) -> None:
    tabular.seed_python_numpy(seed)


def build_classifier(cfg: Config, seed: int) -> lgb.LGBMClassifier:
    n_classes = len(cfg.particles)
    multiclass = (
        {"objective": "multiclass", "num_class": n_classes} if n_classes > 2 else {"objective": "binary"}
    )
    return lgb.LGBMClassifier(
        **multiclass,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        class_weight=cfg.class_weight,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def search_hyperparameters(x_train: pd.DataFrame, y_train: np.ndarray, cfg: Config, seed: int) -> dict:
    """Search with SMOTE inside each cross-validation fold."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    steps = []
    if cfg.use_smote:
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("clf", build_classifier(cfg, seed)))
    grid = GridSearchCV(
        Pipeline(steps),
        cfg.param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(x_train, y_train)
    best = {k.removeprefix("clf__"): v for k, v in grid.best_params_.items()}
    LOGGER.info("Best parameters: %s (CV f1_macro=%.4f)", best, grid.best_score_)
    return best


def train(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: Config,
    seed: int,
):
    """Resample the training split only, then fit with validation early stopping."""
    best = search_hyperparameters(x_train, y_train, cfg, seed) if cfg.grid_search else {}
    if cfg.use_smote:
        from imblearn.over_sampling import SMOTE

        x_fit, y_fit = SMOTE(random_state=seed).fit_resample(x_train, y_train)
        x_fit = pd.DataFrame(x_fit, columns=x_train.columns)
        LOGGER.info("Training split after SMOTE: %s", np.bincount(y_fit).tolist())
    else:
        x_fit, y_fit = x_train, y_train
    model = build_classifier(cfg, seed)
    if best:
        model.set_params(**best)
    model.fit(
        x_fit,
        y_fit,
        eval_set=[(x_fit, y_fit), (x_val, y_val)],
        eval_names=["train", "valid"],
        eval_metric="multi_logloss" if len(cfg.particles) > 2 else "binary_logloss",
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    return model, best


def evaluate(model, x_test: pd.DataFrame, y_test: np.ndarray, cfg: Config):
    n_classes = len(cfg.particles)
    proba = model.predict_proba(x_test)
    y_pred = np.argmax(proba, axis=1)
    results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, proba, labels=np.arange(n_classes))),
        "n_test": int(y_test.size),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
    }
    if n_classes > 2:
        results["auc_ovr_macro"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
    else:
        results["auc"] = float(roc_auc_score(y_test, proba[:, 1]))
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
    for index, particle in enumerate(cfg.particles):
        true_count = cm[index].sum()
        results[f"efficiency_{particle}"] = (
            float(cm[index, index] / true_count) if true_count else float("nan")
        )
        predicted_count = cm[:, index].sum()
        results[f"purity_{particle}"] = (
            float(cm[index, index] / predicted_count) if predicted_count else float("nan")
        )
    LOGGER.info(
        "Test: accuracy=%.4f  balanced=%.4f  logloss=%.4f%s",
        results["accuracy"],
        results["balanced_accuracy"],
        results["log_loss"],
        f"  AUC={results['auc_ovr_macro']:.4f}" if n_classes > 2 else f"  AUC={results['auc']:.4f}",
    )
    LOGGER.info(
        "Classification report:\n%s",
        classification_report(y_test, y_pred, target_names=list(cfg.particles), digits=4, zero_division=0),
    )
    return results, cm, proba, y_pred
