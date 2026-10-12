import pandas as pd

from telecom_churn.schema import FEATURES, validate_features


def risk_signals(row: pd.Series) -> list[str]:
    signals = []
    if row["usage_trend"] < 0.6 and row["usage_mb_90d"] > 0:
        signals.append("近期流量使用下降")
    if row["complaints_30d"] >= 2:
        signals.append("近期多次服务投诉")
    if 0 < row["contract_days_remaining"] <= 30:
        signals.append("合约即将到期")
    if row["overdue_invoices"] > 0:
        signals.append("存在逾期账单")
    if row["active_days_30d"] < 15:
        signals.append("近期活跃度偏低")
    return signals


def score_customers(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    frame = validate_features(frame)
    scores = bundle["model"].predict_proba(frame[FEATURES])[:, 1]
    meta = bundle["metadata"]
    thresholds = meta["thresholds"]
    rows = []
    for (_, row), score in zip(frame.iterrows(), scores, strict=True):
        level = (
            "high"
            if score >= thresholds["high"]
            else "medium"
            if score >= thresholds["medium"]
            else "low"
        )
        signals = risk_signals(row)
        if level == "low":
            action = "常规服务维护"
        elif "近期多次服务投诉" in signals:
            action = "优先服务回访"
        elif "存在逾期账单" in signals:
            action = "账单核查与服务提醒"
        elif "合约即将到期" in signals:
            action = "合约续约沟通"
        else:
            action = "套餐使用情况回访"
        rows.append(
            {
                "customer_id": row["customer_id"],
                "snapshot_date": str(row["snapshot_date"].date()),
                "risk_score": float(score),
                "risk_level": level,
                "risk_signals": signals,
                "suggested_action": action,
                "model_version": meta["model_version"],
            }
        )
    return pd.DataFrame(rows)


def retention_list(
    scores: pd.DataFrame, eligibility: pd.DataFrame, capacity: int, cooldown_days: int = 14
) -> pd.DataFrame:
    if capacity < 1 or cooldown_days < 0:
        raise ValueError("capacity must be positive and cooldown_days nonnegative")
    if scores["snapshot_date"].nunique() != 1:
        raise ValueError("Retention export requires exactly one snapshot date")
    if scores["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer_id in scores")
    required = {"customer_id", "contact_allowed", "last_contact_at"}
    if not required.issubset(eligibility.columns):
        raise ValueError(f"Eligibility requires {sorted(required)}")
    if eligibility["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer_id in eligibility")
    if (
        eligibility["contact_allowed"].isna().any()
        or not eligibility["contact_allowed"].isin([0, 1]).all()
    ):
        raise ValueError("contact_allowed must be 0 or 1")
    joined = scores.merge(
        eligibility[list(required)], on="customer_id", how="inner", validate="one_to_one"
    )
    last = pd.to_datetime(joined["last_contact_at"], errors="raise")
    as_of = pd.to_datetime(joined["snapshot_date"])
    eligible = joined[
        (joined["contact_allowed"] == 1)
        & (last.isna() | ((as_of - last).dt.days >= cooldown_days))
        & (joined["risk_level"] == "high")
    ].copy()
    eligible = eligible.sort_values(["risk_score", "customer_id"], ascending=[False, True]).head(
        capacity
    )
    eligible.insert(0, "priority", range(1, len(eligible) + 1))
    return eligible.drop(columns=["contact_allowed", "last_contact_at"])
