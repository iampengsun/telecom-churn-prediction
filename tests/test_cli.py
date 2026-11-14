import json

import pandas as pd
from typer.testing import CliRunner

from telecom_churn.cli import app

runner = CliRunner()


def invoke(*args):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output + str(result.exception)
    return result


def test_cli_pipeline(raw_dir, tmp_path):
    feature_file = tmp_path / "features.csv"
    artifacts = tmp_path / "artifacts"
    invoke(
        "build-features",
        "--raw-dir",
        str(raw_dir),
        "--output",
        str(feature_file),
        "--data-end",
        "2026-02-01",
    )
    invoke(
        "train",
        "--features",
        str(feature_file),
        "--artifacts",
        str(artifacts),
        "--algorithm",
        "logistic",
    )
    version = json.loads((artifacts / "latest.json").read_text())["model_version"]
    metadata = json.loads((artifacts / version / "metadata.json").read_text())
    assert metadata["data_source"]["source"] == "generated_fixture"
    latest = pd.read_csv(feature_file)
    latest = latest[latest["snapshot_date"] == latest["snapshot_date"].max()]
    latest.to_csv(feature_file, index=False)
    scores = tmp_path / "scores.csv"
    invoke(
        "score", "--features", str(feature_file), "--model", str(artifacts), "--output", str(scores)
    )
    scored = pd.read_csv(scores)
    assert len(scored) == len(latest)
    assert scored["risk_score"].between(0, 1).all()
    for command in ["evaluate", "monitor"]:
        report = tmp_path / f"{command}.json"
        invoke(
            command,
            "--features",
            str(feature_file),
            "--model",
            str(artifacts),
            "--output",
            str(report),
        )
        assert json.loads(report.read_text())["model_version"] == version
    eligibility = latest[["customer_id"]].copy()
    eligibility["contact_allowed"] = 1
    eligibility["last_contact_at"] = None
    eligibility_file = tmp_path / "eligibility.csv"
    eligibility.to_csv(eligibility_file, index=False)
    output = tmp_path / "retention.csv"
    invoke(
        "retention-list",
        "--scores",
        str(scores),
        "--eligibility",
        str(eligibility_file),
        "--capacity",
        "5",
        "--output",
        str(output),
    )
    selected = pd.read_csv(output)
    assert 0 < len(selected) <= 5
    assert selected["risk_level"].eq("high").all()


def test_cli_requires_mature_labels(raw_dir, tmp_path):
    result = runner.invoke(
        app,
        ["build-features", "--raw-dir", str(raw_dir), "--output", str(tmp_path / "features.csv")],
    )
    assert result.exit_code != 0
    assert "data_end" in str(result.exception)
