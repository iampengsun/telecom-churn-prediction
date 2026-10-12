import json
import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from telecom_churn.features import build_dataset
from telecom_churn.metrics import classification_metrics
from telecom_churn.modeling import load_model, train, write_json
from telecom_churn.monitoring import drift_report
from telecom_churn.sample_data import generate
from telecom_churn.schema import validate_features
from telecom_churn.scoring import retention_list, score_customers

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Telecom churn training, scoring and monitoring.",
)


def read_features(path: Path, labeled: bool = False) -> pd.DataFrame:
    return validate_features(pd.read_csv(path, dtype={"customer_id": str}), labeled=labeled)


def save_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    if "risk_signals" in out:
        out["risk_signals"] = out["risk_signals"].map(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
    out.to_csv(path, index=False)


@app.command("generate-data")
def generate_data(raw_dir: Path = Path("data/raw"), customers: int = 800, seed: int = 42):
    """Generate reproducible integration data."""
    typer.echo(json.dumps(generate(raw_dir, customers, seed), ensure_ascii=False))


@app.command("build-features")
def build_features(
    raw_dir: Path = Path("data/raw"),
    output: Path = Path("data/features.csv"),
    start: str = "2025-01-01",
    end: str = "2025-12-01",
    data_end: str | None = None,
    labeled: bool = True,
):
    """Build point-in-time customer features."""
    frame = build_dataset(raw_dir, start, end, data_end, labeled)
    save_csv(frame, output)
    source = raw_dir / "metadata.json"
    metadata = (
        json.loads(source.read_text()) if source.exists() else {"source": "external_raw_tables"}
    )
    metadata.update(
        {
            "feature_start": start,
            "feature_end": end,
            "data_end_exclusive": data_end,
            "labeled": labeled,
            "rows": len(frame),
        }
    )
    write_json(output.with_suffix(".metadata.json"), metadata)
    typer.echo(f"{output}: {len(frame)} rows")


@app.command("train")
def train_model(
    features: Path = Path("data/features.csv"),
    artifacts: Path = Path("artifacts"),
    validation_start: str = "2025-08-01",
    test_start: str = "2025-10-01",
    algorithm: str = "lightgbm",
    seed: int = 42,
    top_fraction: float = 0.1,
):
    """Compare candidates on validation data and evaluate the selected model on test data."""
    metadata_path = features.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    output = train(
        read_features(features, True),
        artifacts,
        validation_start,
        test_start,
        algorithm,
        seed,
        top_fraction,
        metadata,
    )
    typer.echo(str(output))


@app.command("score")
def score(
    features: Annotated[Path, typer.Option()],
    output: Path = Path("reports/scores.csv"),
    model: Path = Path("artifacts"),
):
    """Score feature rows with a persisted model."""
    save_csv(score_customers(read_features(features), load_model(model)), output)
    typer.echo(str(output))


@app.command("evaluate")
def evaluate(
    features: Annotated[Path, typer.Option()],
    output: Path = Path("reports/evaluation.json"),
    model: Path = Path("artifacts"),
    top_fraction: float = 0.1,
):
    """Evaluate a persisted model on labeled feature rows."""
    frame, bundle = read_features(features, True), load_model(model)
    scored = score_customers(frame, bundle)
    report = classification_metrics(
        frame["churn"], scored["risk_score"], bundle["metadata"]["thresholds"]["high"], top_fraction
    )
    write_json(output, {"model_version": bundle["metadata"]["model_version"], "metrics": report})
    typer.echo(str(output))


@app.command("monitor")
def monitor(
    features: Annotated[Path, typer.Option()],
    output: Path = Path("reports/drift.json"),
    model: Path = Path("artifacts"),
):
    """Measure feature distribution drift against the training reference."""
    bundle = load_model(model)
    report = drift_report(read_features(features), bundle["reference"])
    report["model_version"] = bundle["metadata"]["model_version"]
    write_json(output, report)
    typer.echo(str(output))


@app.command("retention-list")
def export_retention(
    scores: Annotated[Path, typer.Option()],
    eligibility: Annotated[Path, typer.Option()],
    capacity: int = 100,
    output: Path = Path("reports/retention.csv"),
    cooldown_days: int = 14,
):
    """Export high-risk customers satisfying contact and cooldown constraints."""
    scored = pd.read_csv(scores, dtype={"customer_id": str})
    scored["risk_signals"] = scored["risk_signals"].map(json.loads)
    allowed = pd.read_csv(eligibility, dtype={"customer_id": str})
    save_csv(retention_list(scored, allowed, capacity, cooldown_days), output)
    typer.echo(str(output))


@app.command("demo")
def demo(customers: int = 800, seed: int = 42, algorithm: str = "lightgbm"):
    """Run the integrated data, training, scoring and monitoring workflow."""
    generate_data(customers=customers, seed=seed)
    build_features(data_end="2026-02-01")
    train_model(algorithm=algorithm, seed=seed)
    latest = read_features(Path("data/features.csv"))
    latest = latest[latest["snapshot_date"] == latest["snapshot_date"].max()]
    save_csv(latest, Path("data/latest.csv"))
    payload = latest.head(3).copy()
    payload["snapshot_date"] = payload["snapshot_date"].dt.strftime("%Y-%m-%d")
    payload = payload.drop(columns=["churn", "label_end_date"], errors="ignore")
    write_json(Path("data/request.json"), {"customers": payload.to_dict("records")})
    score(Path("data/latest.csv"))
    monitor(Path("data/latest.csv"))
    eligibility = latest[["customer_id"]].copy()
    eligibility["contact_allowed"] = 1
    eligibility["last_contact_at"] = None
    save_csv(eligibility, Path("data/eligibility.csv"))
    export_retention(Path("reports/scores.csv"), Path("data/eligibility.csv"))


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the scoring API and operations console."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run("telecom_churn.api:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    app()
