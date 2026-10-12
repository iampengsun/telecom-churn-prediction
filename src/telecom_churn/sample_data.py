"""Reproducible integration fixture generator; not a production data connector."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from telecom_churn.features import TABLE_COLUMNS


def generate(raw_dir: Path, customers: int = 800, seed: int = 42) -> dict:
    if customers < 80:
        raise ValueError("At least 80 customers are required for temporal evaluation")
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = {name: [] for name in TABLE_COLUMNS}
    start, end = pd.Timestamp("2024-10-01"), pd.Timestamp("2026-02-01")
    dates = pd.date_range(start, end - pd.Timedelta(days=1))
    for i in range(customers):
        cid = f"C{i:07d}"
        joined = start - pd.Timedelta(days=int(rng.integers(1, 1800)))
        plan = rng.choice(["basic", "standard", "premium"], p=[0.4, 0.4, 0.2])
        fee = {"basic": 39, "standard": 89, "premium": 169}[plan]
        contract_end = start + pd.Timedelta(days=int(rng.integers(-100, 700)))
        records["customers"].append(
            [cid, joined, contract_end, fee, plan, rng.choice(["east", "west", "north", "south"])]
        )
        churn_date = None
        if rng.random() < 0.45:
            churn_date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 395)))
            records["events"].append([cid, churn_date, rng.choice(["voluntary_close", "port_out"])])
        latent = rng.normal()
        base_usage = fee * rng.uniform(4, 12)
        for day in dates:
            if churn_date is not None and day >= churn_date:
                break
            days_to_churn = (churn_date - day).days if churn_date is not None else 10000
            stress = max(0.0, 1 - days_to_churn / 75) if days_to_churn < 75 else 0.0
            # Shared latent state creates noisy early warning signals across source tables.
            active = rng.random() > 0.08 + stress * 0.25
            data_mb = rng.gamma(2, base_usage / 2) * (1 - stress * 0.75) if active else 0.0
            calls = rng.gamma(2, 5) * (1 - stress * 0.6) if active else 0.0
            records["usage"].append([cid, day, round(data_mb, 3), round(calls, 3)])
            if rng.random() < 0.003 + stress * 0.05 + max(latent, 0) * 0.002:
                records["tickets"].append([f"T{len(records['tickets']):09d}", cid, day])
            if day.day == 1:
                due = day + pd.Timedelta(days=10)
                delay = int(rng.choice([0, 2, 15, 40], p=[0.65, 0.2, 0.1, 0.05]))
                paid = due + pd.Timedelta(days=delay)
                if paid >= end:
                    paid = pd.NaT
                records["invoices"].append(
                    [
                        f"I{len(records['invoices']):09d}",
                        cid,
                        day,
                        due,
                        paid,
                        round(fee * rng.uniform(0.9, 1.3), 2),
                    ]
                )
    for name, rows in records.items():
        pd.DataFrame(rows, columns=TABLE_COLUMNS[name]).to_csv(raw_dir / f"{name}.csv", index=False)
    metadata = {
        "source": "generated_fixture",
        "seed": seed,
        "customers": customers,
        "data_start": str(start.date()),
        "data_end_exclusive": str(end.date()),
    }
    (raw_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
