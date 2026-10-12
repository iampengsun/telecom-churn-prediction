import pandas as pd
import pytest

from telecom_churn.scoring import retention_list


def test_retention_permissions_cooldown_and_capacity():
    scores = pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d", "e", "f"],
            "snapshot_date": ["2025-12-01"] * 6,
            "risk_score": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
            "risk_level": ["high"] * 5 + ["low"],
        }
    )
    eligibility = pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d", "f"],
            "contact_allowed": [0, 1, 1, 1, 1],
            "last_contact_at": [None, "2025-11-30", "2025-11-17", None, None],
        }
    )
    result = retention_list(scores, eligibility, capacity=1)
    assert result["customer_id"].tolist() == ["c"]
    assert result["priority"].tolist() == [1]
    result = retention_list(scores, eligibility, capacity=10)
    assert result["customer_id"].tolist() == ["c", "d"]


def test_multiple_snapshot_dates_rejected():
    scores = pd.DataFrame({"snapshot_date": ["2025-11-01", "2025-12-01"]})
    with pytest.raises(ValueError, match="exactly one"):
        retention_list(scores, pd.DataFrame(), 10)
