import json

import numpy as np
import pytest

from telecom_churn.metrics import classification_metrics
from telecom_churn.modeling import load_model, make_pipeline, temporal_split
from telecom_churn.monitoring import drift_report, make_reference
from telecom_churn.schema import FEATURES
from telecom_churn.scoring import score_customers


def test_time_split_purges_unmatured_labels(features):
    parts = temporal_split(features, "2025-08-01", "2025-10-01")
    assert parts["train"]["label_end_date"].max() <= parts["validation"]["snapshot_date"].min()
    assert parts["validation"]["label_end_date"].max() <= parts["test"]["snapshot_date"].min()
    # February 1 + 30 days extends into March: a March boundary must purge February.
    parts = temporal_split(features, "2025-03-01", "2025-08-01")
    assert set(parts["train"]["snapshot_date"].dt.month) == {1}


def test_invalid_split_rejected(features):
    with pytest.raises(ValueError):
        temporal_split(features, "2025-10-01", "2025-08-01")


def test_model_artifact_round_trip(model_dir, latest, bundle):
    scores = score_customers(latest, load_model(model_dir.parent))
    expected = bundle["model"].predict_proba(latest[FEATURES])[:, 1]
    np.testing.assert_allclose(scores["risk_score"], expected)
    assert scores["risk_score"].between(0, 1).all()
    assert set(scores["risk_level"]) <= {"low", "medium", "high"}
    assert bundle["metadata"]["test_metrics"]["roc_auc"] > 0.5
    assert bundle["metadata"]["partitions"]["train"]["end"] < "2025-08-01"


def test_artifact_checksum_rejected(model_dir, tmp_path):
    import shutil

    target = tmp_path / "model"
    shutil.copytree(model_dir, target)
    with (target / "model.joblib").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_model(target)


def test_path_traversal_rejected(tmp_path):
    (tmp_path / "latest.json").write_text(json.dumps({"model_version": "../../bad"}))
    with pytest.raises(ValueError, match="pointer"):
        load_model(tmp_path)


def test_unknown_region_supported(bundle, latest):
    latest["region"] = "new-region"
    assert len(score_customers(latest, bundle)) == len(latest)


def test_identifier_and_label_are_not_model_features(bundle):
    names = bundle["model"].named_steps["preprocess"].get_feature_names_out()
    assert "customer_id" not in names
    assert "churn" not in names
    assert "snapshot_date" not in names


def test_top_k_metrics_and_single_class():
    report = classification_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 0.75, 0.5)
    assert report["precision_at_k"] == 0.5
    assert report["recall_at_k"] == 0.5
    assert report["lift_at_k"] == 1
    assert report["confusion_matrix"] == [[1, 1], [1, 1]]
    no_positive = classification_metrics([0, 0], [0.2, 0.1], 0.5)
    assert no_positive["roc_auc"] is None
    assert no_positive["average_precision"] is None
    assert no_positive["lift_at_k"] is None


def test_drift_detects_shift(latest):
    reference = make_reference(latest)
    stable = drift_report(latest, reference)
    assert stable["alert_count"] == 0
    shifted = latest.copy()
    shifted["monthly_fee"] = 5000
    shifted["region"] = "unknown"
    report = drift_report(shifted, reference)
    assert report["alert_count"] >= 2


def test_constant_feature_drift(latest):
    latest["monthly_fee"] = 50
    reference = make_reference(latest)
    latest["monthly_fee"] = 100
    report = drift_report(latest, reference)
    assert next(r for r in report["features"] if r["feature"] == "monthly_fee")["status"] == "alert"


@pytest.mark.parametrize("algorithm", ["hist_gradient_boosting", "lightgbm"])
def test_alternative_algorithms(features, algorithm):
    if algorithm == "lightgbm":
        try:
            import lightgbm  # noqa: F401
        except (ImportError, OSError):
            pytest.skip("Optional LightGBM runtime not installed")
    model = make_pipeline(algorithm, 42)
    part = features.head(400)
    model.fit(part[FEATURES], part["churn"])
    scores = model.predict_proba(part[FEATURES])[:, 1]
    assert np.isfinite(scores).all()


def test_no_training_preprocessing_leakage(features):
    parts = temporal_split(features, "2025-08-01", "2025-10-01")
    model = make_pipeline("logistic", 42)
    model.fit(parts["train"][FEATURES], parts["train"]["churn"])
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"]
    assert scaler.mean_[1] == pytest.approx(parts["train"]["monthly_fee"].mean())
