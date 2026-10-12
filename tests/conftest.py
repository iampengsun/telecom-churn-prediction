from pathlib import Path

import pandas as pd
import pytest

from telecom_churn.features import build_dataset
from telecom_churn.modeling import load_model, train
from telecom_churn.sample_data import generate


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("raw")
    generate(path, customers=160, seed=42)
    return path


@pytest.fixture(scope="session")
def features(raw_dir) -> pd.DataFrame:
    return build_dataset(raw_dir, "2025-01-01", "2025-12-01", "2026-02-01")


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, features) -> Path:
    return train(features, tmp_path_factory.mktemp("artifacts"), algorithm="logistic")


@pytest.fixture(scope="session")
def bundle(model_dir):
    return load_model(model_dir)


@pytest.fixture
def latest(features):
    return features[features["snapshot_date"] == features["snapshot_date"].max()].copy()


@pytest.fixture
def payload(latest):
    rows = latest.head(3).drop(columns=["churn", "label_end_date"])
    rows["snapshot_date"] = rows["snapshot_date"].dt.strftime("%Y-%m-%d")
    return {"customers": rows.to_dict("records")}
