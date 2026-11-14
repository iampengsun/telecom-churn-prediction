import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from telecom_churn.features import build_dataset, load_tables, snapshot
from telecom_churn.schema import validate_features


def test_future_observations_do_not_change_features(raw_dir):
    tables = load_tables(raw_dir)
    cutoff = pd.Timestamp("2025-06-01")
    expected = snapshot(tables, cutoff)
    tables["usage"].loc[tables["usage"]["date"] >= cutoff, "data_mb"] = 1e9
    tables["tickets"].loc[tables["tickets"]["opened_at"] >= cutoff, "opened_at"] += pd.Timedelta(
        days=500
    )
    tables["invoices"].loc[tables["invoices"]["issued_at"] >= cutoff, "amount"] = 1e9
    # Payment after cutoff remains unpaid regardless of its actual future payment date.
    tables["invoices"].loc[tables["invoices"]["paid_at"] >= cutoff, "paid_at"] += pd.Timedelta(
        days=500
    )
    assert_frame_equal(expected, snapshot(tables, cutoff))


def test_window_boundaries_and_label_horizon(raw_dir):
    tables = load_tables(raw_dir)
    cutoff = pd.Timestamp("2025-06-01")
    cid = snapshot(tables, cutoff).iloc[0]["customer_id"]
    tables["usage"] = pd.DataFrame(
        {
            "customer_id": [cid] * 4,
            "date": [cutoff - pd.Timedelta(days=d) for d in [0, 1, 30, 31]],
            "data_mb": [10000, 10, 20, 40],
            "call_minutes": [10000, 1, 2, 4],
        }
    )
    row = snapshot(tables, cutoff).set_index("customer_id").loc[cid]
    assert row["usage_mb_30d"] == 30
    assert row["usage_mb_90d"] == 70
    assert row["active_days_30d"] == 2
    assert row["usage_trend"] == pytest.approx(90 / 70)


def test_customer_removed_after_churn(raw_dir):
    tables = load_tables(raw_dir)
    cutoff = pd.Timestamp("2025-07-01")
    result = snapshot(tables, cutoff)
    churned = tables["events"].loc[tables["events"]["event_at"] < cutoff, "customer_id"]
    assert not result["customer_id"].isin(churned).any()


def test_label_boundary_and_event_type(raw_dir, tmp_path):
    import shutil

    target = tmp_path / "raw"
    shutil.copytree(raw_dir, target)
    tables = load_tables(raw_dir)
    cutoff = pd.Timestamp("2025-06-01")
    ids = snapshot(tables, cutoff)["customer_id"].head(4).tolist()
    events = pd.DataFrame(
        {
            "customer_id": ids,
            "event_at": ["2025-06-01", "2025-06-30", "2025-07-01", "2025-06-15"],
            "event_type": ["port_out", "voluntary_close", "port_out", "involuntary_close"],
        }
    )
    events.to_csv(target / "events.csv", index=False)
    rows = build_dataset(target, "2025-06-01", "2025-06-01", "2025-07-01").set_index("customer_id")
    assert rows.loc[ids, "churn"].tolist() == [1, 1, 0, 0]


def test_immature_labels_rejected(raw_dir):
    with pytest.raises(ValueError, match="Immature"):
        build_dataset(raw_dir, "2025-12-01", "2025-12-01", "2025-12-15")


def test_inference_does_not_need_labels(raw_dir):
    frame = build_dataset(raw_dir, "2026-01-01", "2026-01-01", None, labeled=False)
    assert "churn" not in frame


@pytest.mark.parametrize(
    "column,value",
    [("usage_trend", -1), ("monthly_fee", float("inf")), ("active_days_30d", 31), ("region", None)],
)
def test_invalid_features_rejected(latest, column, value):
    latest[column] = value
    with pytest.raises(ValueError):
        validate_features(latest)


def test_duplicate_rows_rejected(latest):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_features(pd.concat([latest, latest.iloc[:1]]))


def test_missing_columns_rejected(latest):
    with pytest.raises(ValueError, match="Missing"):
        validate_features(latest.drop(columns=["monthly_fee"]))


def test_partitioned_compressed_inputs_match_single_directory(raw_dir, tmp_path):
    """按客户分片不能改变窗口聚合和标签；同时覆盖跨片重复快照拒绝。"""
    tables = load_tables(raw_dir)
    ids = tables["customers"]["customer_id"].tolist()
    root = tmp_path / "partitioned"
    for i, group in enumerate([ids[:80], ids[80:]]):
        part = root / f"part-{i:03d}"
        part.mkdir(parents=True)
        for name, table in tables.items():
            table[table["customer_id"].isin(group)].to_csv(part / f"{name}.csv.gz", index=False)
    expected = build_dataset(raw_dir, "2025-06-01", "2025-07-01", "2026-02-01")
    actual = build_dataset(root, "2025-06-01", "2025-07-01", "2026-02-01")
    keys = ["customer_id", "snapshot_date"]
    assert_frame_equal(
        expected.sort_values(keys).reset_index(drop=True),
        actual.sort_values(keys).reset_index(drop=True),
    )
    import shutil

    shutil.copytree(root / "part-000", root / "part-002")
    with pytest.raises(ValueError, match="Duplicate"):
        build_dataset(root, "2025-06-01", "2025-07-01", "2026-02-01")
