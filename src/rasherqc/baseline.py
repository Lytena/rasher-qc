"""Classical baseline: hand-crafted features + random forest.

Two outputs, mirroring the detector's tray-level outputs:

* ``fail``: tray PASS (0) / FAIL (1);
* ``presence``: one binary label per defect class ("is at least one X on
  this tray?"), as a multi-label problem.

Random forests were chosen for the baseline because they need no feature
scaling, cope with correlated features, give probabilistic outputs that can
be thresholded exactly like detector confidences, and expose feature
importances for interpretation. Hyper-parameters are chosen by 5-fold
stratified cross-validation on the *training split only*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config
from .dataset import Record, tray_fail_label
from .features import FEAT_WH, extract_batch


def load_feature_images(records: list[Record]) -> list[np.ndarray]:
    """Read images and resize them to the baseline working resolution."""
    out = []
    for r in records:
        with Image.open(r.path) as im:
            out.append(np.asarray(im.convert("RGB").resize(FEAT_WH, Image.BOX)))
    return out


def labels(records: list[Record], n_classes: int, defect_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Tray FAIL labels (N,) and per-class presence labels (N, n_classes)."""
    y_fail = np.array([tray_fail_label(r, defect_ids) for r in records])
    y_pres = np.zeros((len(records), n_classes), int)
    for i, r in enumerate(records):
        y_pres[i, np.unique(r.classes)] = 1
    return y_fail, y_pres


@dataclass
class Baseline:
    """Fitted baseline models.

    Attributes:
        fail_model: Random forest for PASS/FAIL.
        presence_models: One random forest per class (``None`` if a class
            never or always occurs in training).
        cv_results: Cross-validation table of the PASS/FAIL grid search.
    """

    fail_model: RandomForestClassifier
    presence_models: list[RandomForestClassifier | None]
    cv_results: dict

    def predict_fail_proba(self, F: np.ndarray) -> np.ndarray:
        """Probability that each tray should FAIL."""
        return self.fail_model.predict_proba(F)[:, 1]

    def predict_presence_proba(self, F: np.ndarray) -> np.ndarray:
        """(N, n_classes) probability that each defect class is present."""
        cols = []
        for m in self.presence_models:
            cols.append(np.zeros(len(F)) if m is None else m.predict_proba(F)[:, 1])
        return np.stack(cols, 1)


def fit(F: np.ndarray, y_fail: np.ndarray, y_pres: np.ndarray, seed: int = config.SEED, n_jobs: int = 2) -> Baseline:
    """Fit the baseline with a small cross-validated grid search.

    Args:
        F: (N, d) training features.
        y_fail: (N,) PASS/FAIL labels.
        y_pres: (N, C) presence labels.
        seed: Random seed.
        n_jobs: Parallel jobs for the forest / search.

    Returns:
        A fitted :class:`Baseline`.
    """
    grid = {"n_estimators": [200, 500], "max_depth": [None, 12], "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", 0.3]}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    gs = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=n_jobs),
                      grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
    gs.fit(F, y_fail)
    best = gs.best_params_
    pres = []
    for c in range(y_pres.shape[1]):
        yc = y_pres[:, c]
        if yc.min() == yc.max():
            pres.append(None)
            continue
        m = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=n_jobs, **best)
        pres.append(m.fit(F, yc))
    cvr = {k: list(map(lambda v: v if isinstance(v, (int, float, str)) or v is None else str(v), vals))
           for k, vals in gs.cv_results_.items() if k.startswith(("param_", "mean_test", "std_test", "rank"))}
    cvr["best_params"] = best
    cvr["best_score"] = float(gs.best_score_)
    return Baseline(gs.best_estimator_, pres, cvr)


def features_for(records: list[Record], mode: str = "fast", n_jobs: int = 2) -> np.ndarray:
    """Load images of ``records`` and extract baseline features."""
    return extract_batch(load_feature_images(records), mode=mode, n_jobs=n_jobs, progress=True)
