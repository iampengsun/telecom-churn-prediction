import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telecom_churn.metrics import classification_metrics
from telecom_churn.monitoring import make_reference
from telecom_churn.schema import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    SCHEMA_VERSION,
    validate_features,
)


def temporal_split(frame: pd.DataFrame, validation_start: str, test_start: str) -> dict:
    frame = validate_features(frame, labeled=True)
    val_date, test_date = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    if val_date >= test_date:
        raise ValueError("validation_start must precede test_start")
    # Labels must have fully matured before the next partition's first scoring date.
    train = frame[(frame["snapshot_date"] < val_date) & (frame["label_end_date"] <= val_date)]
    validation = frame[
        (frame["snapshot_date"] >= val_date)
        & (frame["snapshot_date"] < test_date)
        & (frame["label_end_date"] <= test_date)
    ]
    test = frame[frame["snapshot_date"] >= test_date]
    parts = {"train": train, "validation": validation, "test": test}
    for name, part in parts.items():
        if part.empty or part["churn"].nunique() != 2:
            raise ValueError(f"{name} partition must contain both target classes")
    return parts


def make_pipeline(algorithm: str, seed: int) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    if algorithm == "logistic":
        estimator = LogisticRegression(C=0.5, max_iter=2000, random_state=seed)
    elif algorithm == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=180,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=2,
            early_stopping=False,
            random_state=seed,
        )
    elif algorithm == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except (ImportError, OSError) as exc:
            raise ValueError(
                "LightGBM runtime unavailable; install the lightgbm extra and libomp on macOS"
            ) from exc
        estimator = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=5,
            min_child_samples=30,
            reg_lambda=2,
            verbosity=-1,
            n_jobs=2,
            random_state=seed,
            deterministic=True,
            force_col_wise=True,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return Pipeline([("preprocess", preprocess), ("classifier", estimator)])


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def train(
    frame: pd.DataFrame,
    artifacts: Path,
    validation_start: str = "2025-08-01",
    test_start: str = "2025-10-01",
    algorithm: str = "lightgbm",
    seed: int = 42,
    top_fraction: float = 0.1,
    source_metadata: dict | None = None,
) -> Path:
    if not 0 < top_fraction < 0.5:
        raise ValueError("top_fraction must be between 0 and 0.5")
    parts = temporal_split(frame, validation_start, test_start)
    tr, val, test = (parts[n] for n in ["train", "validation", "test"])
    models, comparison = {}, {}
    for name in dict.fromkeys(["logistic", algorithm]):
        pipeline = make_pipeline(name, seed)
        pipeline.fit(tr[FEATURES], tr["churn"])
        scores = pipeline.predict_proba(val[FEATURES])[:, 1]
        threshold = float(np.quantile(scores, 1 - top_fraction))
        comparison[name] = classification_metrics(val["churn"], scores, threshold, top_fraction)
        models[name] = pipeline
    winner = max(comparison, key=lambda name: comparison[name]["average_precision"])
    model = models[winner]
    high = comparison[winner]["threshold"]
    medium = float(np.quantile(model.predict_proba(val[FEATURES])[:, 1], 0.5))
    test_scores = model.predict_proba(test[FEATURES])[:, 1]
    test_metrics = classification_metrics(test["churn"], test_scores, high, top_fraction)
    importance = permutation_importance(
        model,
        val[FEATURES],
        val["churn"],
        scoring="average_precision",
        n_repeats=3,
        random_state=seed,
        n_jobs=1,
    )
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    output = artifacts / version
    output.mkdir(parents=True, exist_ok=False)
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(
            frame.sort_values(["snapshot_date", "customer_id"])[
                ["customer_id", "snapshot_date", *FEATURES, "churn"]
            ],
            index=False,
        ).values.tobytes()
    ).hexdigest()
    versions = {
        name: importlib.metadata.version(name)
        for name in ["pandas", "numpy", "scikit-learn", "joblib"]
    }
    if "lightgbm" in models:
        versions["lightgbm"] = importlib.metadata.version("lightgbm")
    metadata = {
        "model_version": version,
        "schema_version": SCHEMA_VERSION,
        "selected_algorithm": winner,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "features": FEATURES,
        "thresholds": {"high": high, "medium": medium},
        "selection_metric": "validation_average_precision",
        "validation_comparison": comparison,
        "test_metrics": test_metrics,
        "test_by_month": {
            str(date.date()): classification_metrics(
                group["churn"], model.predict_proba(group[FEATURES])[:, 1], high, top_fraction
            )
            for date, group in test.groupby("snapshot_date")
        },
        "partitions": {
            name: {
                "rows": len(p),
                "start": str(p["snapshot_date"].min().date()),
                "end": str(p["snapshot_date"].max().date()),
                "positives": int(p["churn"].sum()),
            }
            for name, p in parts.items()
        },
        "purged_rows": len(frame) - sum(len(p) for p in parts.values()),
        "data_fingerprint": fingerprint,
        "data_source": source_metadata or {"source": "external_feature_table"},
        "python_version": platform.python_version(),
        "dependencies": versions,
        "permutation_importance": sorted(
            [
                {"feature": col, "mean": float(mean), "std": float(std)}
                for col, mean, std in zip(
                    FEATURES, importance.importances_mean, importance.importances_std, strict=True
                )
            ],
            key=lambda row: row["mean"],
            reverse=True,
        ),
    }
    bundle = {"model": model, "metadata": metadata, "reference": make_reference(tr)}
    model_path = output / "model.joblib"
    joblib.dump(bundle, model_path)
    metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    write_json(output / "metadata.json", metadata)
    scored_test = test[["customer_id", "snapshot_date", "churn"]].copy()
    scored_test["risk_score"] = test_scores
    scored_test.to_csv(output / "test_predictions.csv", index=False)
    # Promote only after all immutable artifacts have been written successfully.
    pointer = artifacts / f".latest-{uuid4().hex}.tmp"
    write_json(pointer, {"model_version": version})
    os.replace(pointer, artifacts / "latest.json")
    return output


def resolve_model(path: Path) -> Path:
    if path.is_file():
        if path.suffix != ".joblib":
            raise ValueError("Expected a model.joblib file or artifact directory")
        return path
    if (path / "model.joblib").is_file():
        return path / "model.joblib"
    pointer = json.loads((path / "latest.json").read_text(encoding="utf-8"))
    version = pointer["model_version"]
    if not isinstance(version, str) or Path(version).name != version or version in {".", ".."}:
        raise ValueError("Invalid model version pointer")
    return path / version / "model.joblib"


def load_model(path: Path) -> dict:
    model_path = resolve_model(path)
    metadata = json.loads(model_path.with_name("metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model artifact checksum mismatch")
    if metadata["schema_version"] != SCHEMA_VERSION or metadata["features"] != FEATURES:
        raise ValueError("Model feature schema does not match the application")
    # Joblib is an executable serialization format; artifact directories must be trusted.
    bundle = joblib.load(model_path)
    if bundle["metadata"]["model_version"] != metadata["model_version"]:
        raise ValueError("Model version mismatch")
    bundle["metadata"] = metadata
    return bundle
