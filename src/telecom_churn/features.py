from pathlib import Path

import numpy as np
import pandas as pd

from telecom_churn.schema import CHURN_TYPES, HORIZON_DAYS, validate_features

TABLE_COLUMNS = {
    "customers": ["customer_id", "joined_at", "contract_end", "monthly_fee", "plan_type", "region"],
    "usage": ["customer_id", "date", "data_mb", "call_minutes"],
    "invoices": ["invoice_id", "customer_id", "issued_at", "due_at", "paid_at", "amount"],
    "tickets": ["ticket_id", "customer_id", "opened_at"],
    "events": ["customer_id", "event_at", "event_type"],
}
DATE_COLUMNS = {
    "customers": ["joined_at", "contract_end"],
    "usage": ["date"],
    "invoices": ["issued_at", "due_at", "paid_at"],
    "tickets": ["opened_at"],
    "events": ["event_at"],
}


def load_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name, columns in TABLE_COLUMNS.items():
        frame = pd.read_csv(raw_dir / f"{name}.csv", dtype={"customer_id": str})
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        for col in DATE_COLUMNS[name]:
            frame[col] = pd.to_datetime(frame[col], errors="raise")
            if col != "paid_at" and frame[col].isna().any():
                raise ValueError(f"{name}.{col} cannot be null")
        if frame["customer_id"].isna().any():
            raise ValueError(f"{name}.customer_id cannot be null")
        tables[name] = frame
    customers = tables["customers"]
    for name, keys in {
        "customers": ["customer_id"],
        "usage": ["customer_id", "date"],
        "invoices": ["invoice_id"],
        "tickets": ["ticket_id"],
    }.items():
        if tables[name].duplicated(keys).any():
            raise ValueError(f"{name}: duplicate primary key {keys}")
    for name, frame in tables.items():
        if not frame["customer_id"].isin(customers["customer_id"]).all():
            raise ValueError(f"{name}: unknown customer_id")
    for name, cols in {
        "usage": ["data_mb", "call_minutes"],
        "invoices": ["amount"],
        "customers": ["monthly_fee"],
    }.items():
        values = tables[name][cols].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name}: numeric values must be finite and nonnegative")
    invoices = tables["invoices"]
    if (invoices["due_at"] < invoices["issued_at"]).any():
        raise ValueError("Invoice due date precedes issue date")
    if (invoices["paid_at"] < invoices["issued_at"]).any():
        raise ValueError("Invoice payment date precedes issue date")
    return tables


def _window(frame: pd.DataFrame, column: str, cutoff: pd.Timestamp, days: int):
    # Scoring occurs at 00:00. Same-day observations are not yet available.
    return frame.loc[(frame[column] >= cutoff - pd.Timedelta(days=days)) & (frame[column] < cutoff)]


def snapshot(tables: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff).normalize()
    customers, usage, invoices, tickets, events = (tables[n] for n in TABLE_COLUMNS)
    terminated = events.loc[events["event_at"] < cutoff, "customer_id"]
    out = (
        customers.loc[
            (customers["joined_at"] < cutoff) & ~customers["customer_id"].isin(terminated)
        ]
        .set_index("customer_id")
        .copy()
    )
    if out.empty:
        raise ValueError(f"No eligible customers at {cutoff.date()}")
    out["snapshot_date"] = cutoff
    out["tenure_days"] = (cutoff - out["joined_at"]).dt.days
    out["contract_days_remaining"] = (out["contract_end"] - cutoff).dt.days.clip(lower=0)
    for days in [30, 90]:
        u = _window(usage, "date", cutoff, days)
        out[f"usage_mb_{days}d"] = u.groupby("customer_id")["data_mb"].sum()
        out[f"complaints_{days}d"] = (
            _window(tickets, "opened_at", cutoff, days).groupby("customer_id").size()
        )
        if days == 30:
            out["call_minutes_30d"] = u.groupby("customer_id")["call_minutes"].sum()
            active = u[(u["data_mb"] > 0) | (u["call_minutes"] > 0)]
            out["active_days_30d"] = active.groupby("customer_id")["date"].nunique()
    known_invoices = invoices.loc[invoices["issued_at"] < cutoff]
    unpaid = known_invoices.loc[
        (known_invoices["due_at"] < cutoff)
        & (known_invoices["paid_at"].isna() | (known_invoices["paid_at"] >= cutoff))
    ]
    out["overdue_invoices"] = unpaid.groupby("customer_id").size()
    out["bill_amount_90d"] = (
        _window(known_invoices, "issued_at", cutoff, 90).groupby("customer_id")["amount"].sum()
    )
    aggregates = [
        "usage_mb_30d",
        "usage_mb_90d",
        "call_minutes_30d",
        "active_days_30d",
        "complaints_30d",
        "complaints_90d",
        "overdue_invoices",
        "bill_amount_90d",
    ]
    out[aggregates] = out[aggregates].fillna(0)
    out["usage_trend"] = (3 * out["usage_mb_30d"] / out["usage_mb_90d"].replace(0, np.nan)).fillna(
        0
    )
    return validate_features(out.drop(columns=["joined_at", "contract_end"]).reset_index())


def build_dataset(
    raw_dir: Path, start: str, end: str, data_end: str | None, labeled: bool = True
) -> pd.DataFrame:
    tables = load_tables(raw_dir)
    cutoffs = pd.date_range(start, end, freq="MS")
    if len(cutoffs) == 0:
        raise ValueError("No month-start snapshot dates in requested range")
    if labeled and data_end is None:
        raise ValueError("data_end is required to prove label maturity")
    frames = []
    events = tables["events"]
    for cutoff in cutoffs:
        label_end = cutoff + pd.Timedelta(days=HORIZON_DAYS)
        if labeled and label_end > pd.Timestamp(data_end):
            raise ValueError(f"Immature label window: {cutoff.date()} ends at {label_end.date()}")
        frame = snapshot(tables, cutoff)
        if labeled:
            churned = events.loc[
                (events["event_at"] >= cutoff)
                & (events["event_at"] < label_end)
                & events["event_type"].isin(CHURN_TYPES),
                "customer_id",
            ]
            frame["churn"] = frame["customer_id"].isin(churned).astype(int)
            frame["label_end_date"] = label_end
        frames.append(frame)
    return validate_features(pd.concat(frames, ignore_index=True), labeled=labeled)
