"""
Isotonic regression calibration for Track B predictions.

fit_calibrators: fit DE and DIR calibrators from labeled dev_val predictions.
apply_calibrators: map raw (up, down) → calibrated (up, down).
save_calibrators / load_calibrators: persist to disk via joblib.

Usage (eval-train run):
    python track_b_submit.py --eval-train --calibrate --run-name devval_cal

Usage (test run with pre-fitted calibrators):
    python track_b_submit.py --calibrate \
        --calibrators-from outputs/track_b/devval_cal/calibrators.joblib
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def fit_calibrators(
    dev_val_df: pd.DataFrame,
    results: dict[str, dict],
) -> tuple[IsotonicRegression, IsotonicRegression]:
    """
    Fit DE and DIR isotonic calibrators from labeled dev_val predictions.

    iso_de:  (raw up + down)           → P(DE)
    iso_dir: (raw up / (up + down))    → P(up | DE), fitted on true-DE rows only

    Returns (iso_de, iso_dir).
    """
    rows = []
    for _, row in dev_val_df.iterrows():
        r = results.get(row["id"], {})
        rows.append({
            "label": row["label"],
            "up":    float(r.get("prediction_up",   0.31)),
            "down":  float(r.get("prediction_down",  0.14)),
        })
    df = pd.DataFrame(rows)

    # DE calibrator
    s_de = (df["up"] + df["down"]).values
    y_de = (df["label"] != "none").astype(float).values
    iso_de = IsotonicRegression(out_of_bounds="clip")
    iso_de.fit(s_de, y_de)

    # DIR calibrator (only on true-DE rows)
    de_mask = df["label"] != "none"
    de_df   = df[de_mask]
    denom   = (de_df["up"] + de_df["down"]).values
    denom   = np.where(denom == 0, 1.0, denom)
    s_dir   = de_df["up"].values / denom
    y_dir   = (de_df["label"] == "up").astype(float).values
    iso_dir = IsotonicRegression(out_of_bounds="clip")
    iso_dir.fit(s_dir, y_dir)

    # Monotonicity sanity check
    xs = np.linspace(0, 1, 200)
    assert np.all(np.diff(iso_de.predict(xs))  >= -1e-9), "iso_de is not monotone"
    assert np.all(np.diff(iso_dir.predict(xs)) >= -1e-9), "iso_dir is not monotone"

    return iso_de, iso_dir


def apply_calibrators(
    up: float,
    down: float,
    iso_de: IsotonicRegression,
    iso_dir: IsotonicRegression,
) -> tuple[float, float]:
    """
    Map raw (up, down) → calibrated (up, down).
      cal_up   = P(DE) × P(up  | DE)
      cal_down = P(DE) × P(down| DE) = P(DE) × (1 − P(up|DE))
    """
    p_de = float(iso_de.predict(np.array([up + down]))[0])
    denom = up + down
    s_dir = up / denom if denom > 0 else 0.5
    p_up_given_de = float(iso_dir.predict(np.array([s_dir]))[0])
    return p_de * p_up_given_de, p_de * (1.0 - p_up_given_de)


def save_calibrators(
    iso_de: IsotonicRegression,
    iso_dir: IsotonicRegression,
    path: Path,
) -> None:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"iso_de": iso_de, "iso_dir": iso_dir}, path)


def load_calibrators(path: Path) -> tuple[IsotonicRegression, IsotonicRegression]:
    import joblib
    d = joblib.load(Path(path))
    return d["iso_de"], d["iso_dir"]
