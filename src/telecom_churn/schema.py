from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"
HORIZON_DAYS = 30
NUMERIC_FEATURES = [
    "tenure_days",
    "monthly_fee",
    "contract_days_remaining",
    "usage_mb_30d",
    "usage_mb_90d",
    "usage_trend",
    "call_minutes_30d",
    "active_days_30d",
    "complaints_30d",
    "complaints_90d",
    "overdue_invoices",
    "bill_amount_90d",
]
CATEGORICAL_FEATURES = ["plan_type", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CHURN_TYPES = {"voluntary_close", "port_out"}


class CustomerSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    customer_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    snapshot_date: date
    tenure_days: int = Field(ge=0, le=36500)
    monthly_fee: float = Field(ge=0, le=10000)
    contract_days_remaining: int = Field(ge=0, le=36500)
    usage_mb_30d: float = Field(ge=0)
    usage_mb_90d: float = Field(ge=0)
    usage_trend: float = Field(ge=0, le=3.000001)
    call_minutes_30d: float = Field(ge=0, le=43200)
    active_days_30d: int = Field(ge=0, le=30)
    complaints_30d: int = Field(ge=0)
    complaints_90d: int = Field(ge=0)
    overdue_invoices: int = Field(ge=0)
    bill_amount_90d: float = Field(ge=0)
    plan_type: Literal["basic", "standard", "premium"]
    region: str = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def check_windows(self):
        if self.usage_mb_30d > self.usage_mb_90d + 1e-6:
            raise ValueError("usage_mb_30d must not exceed usage_mb_90d")
        expected = self.usage_mb_30d * 3 / self.usage_mb_90d if self.usage_mb_90d else 0.0
        if abs(self.usage_trend - expected) > 1e-5:
            raise ValueError("usage_trend must equal 3 * usage_mb_30d / usage_mb_90d (0 if empty)")
        if self.complaints_30d > self.complaints_90d:
            raise ValueError("complaints_30d must not exceed complaints_90d")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[CustomerSnapshot] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    customer_id: str
    snapshot_date: str
    risk_score: float
    risk_level: Literal["high", "medium", "low"]
    risk_signals: list[str]
    suggested_action: str
    model_version: str


def validate_features(frame: pd.DataFrame, labeled: bool = False) -> pd.DataFrame:
    required = {"customer_id", "snapshot_date", *FEATURES}
    if labeled:
        required |= {"churn", "label_end_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Feature dataset is empty")
    result = frame.copy()
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], errors="raise")
    if result["snapshot_date"].isna().any():
        raise ValueError("snapshot_date cannot be null")
    if result.duplicated(["customer_id", "snapshot_date"]).any():
        raise ValueError("Duplicate customer_id / snapshot_date")
    for row in result[["customer_id", "snapshot_date", *FEATURES]].to_dict("records"):
        CustomerSnapshot.model_validate(row)
    if labeled:
        if not result["churn"].isin([0, 1]).all():
            raise ValueError("churn must be 0 or 1")
        result["label_end_date"] = pd.to_datetime(result["label_end_date"], errors="raise")
        expected = result["snapshot_date"] + pd.Timedelta(days=HORIZON_DAYS)
        if not result["label_end_date"].eq(expected).all():
            raise ValueError("label_end_date must be snapshot_date + 30 days")
    if not np.isfinite(result[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Numeric features must be finite")
    return result
