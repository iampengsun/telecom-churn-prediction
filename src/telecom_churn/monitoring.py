import numpy as np
import pandas as pd

from telecom_churn.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _distribution(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    smooth = counts.astype(float) + 0.5
    return smooth / smooth.sum()


def make_reference(frame: pd.DataFrame) -> dict:
    reference = {"numeric": {}, "categorical": {}, "rows": len(frame)}
    for column in NUMERIC_FEATURES:
        values = frame[column].to_numpy(dtype=float)
        interior = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        # Include extrema so a formerly constant feature can still register drift.
        interior = np.unique(np.r_[interior, np.nextafter(interior[-1], np.inf)])
        edges = np.r_[-np.inf, interior, np.inf]
        reference["numeric"][column] = {
            "cuts": interior.tolist(),
            "proportions": _distribution(values, edges).tolist(),
        }
    for column in CATEGORICAL_FEATURES:
        reference["categorical"][column] = frame[column].value_counts(normalize=True).to_dict()
    return reference


def drift_report(frame: pd.DataFrame, reference: dict) -> dict:
    if frame.empty:
        raise ValueError("Cannot monitor an empty dataset")
    results = []
    for column, baseline in reference["numeric"].items():
        actual = _distribution(
            frame[column].to_numpy(dtype=float), np.r_[-np.inf, baseline["cuts"], np.inf]
        )
        expected = np.asarray(baseline["proportions"])
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        results.append(
            {
                "feature": column,
                "metric": "psi",
                "value": psi,
                "status": "alert" if psi >= 0.25 else "watch" if psi >= 0.1 else "stable",
            }
        )
    for column, baseline in reference["categorical"].items():
        actual = frame[column].value_counts(normalize=True).to_dict()
        keys = set(actual) | set(baseline)
        distance = sum(abs(actual.get(k, 0) - baseline.get(k, 0)) for k in keys) / 2
        results.append(
            {
                "feature": column,
                "metric": "total_variation",
                "value": distance,
                "unknown_rate": sum(v for k, v in actual.items() if k not in baseline),
                "status": "alert" if distance >= 0.2 else "watch" if distance >= 0.1 else "stable",
            }
        )
    return {
        "rows": len(frame),
        "small_sample": len(frame) < 200,
        "alert_count": sum(row["status"] == "alert" for row in results),
        "features": results,
    }
