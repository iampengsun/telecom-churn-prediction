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
from telecom_churn.schema import validate_features
from telecom_churn.scoring import retention_list, score_customers

# 计算并保存 app；调用 typer.Typer 完成本步处理。
app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Telecom churn training, scoring and monitoring.",
)


# 读取客户编号为字符串的 CSV，然后执行统一特征校验。
def read_features(path: Path, labeled: bool = False) -> pd.DataFrame:
    # 返回经过统一字段和业务约束校验的特征表。
    return validate_features(pd.read_csv(path, dtype={"customer_id": str}), labeled=labeled)


# 把风险信号列表序列化为 JSON 字符串，再保存 CSV。
def save_csv(frame: pd.DataFrame, path: Path):
    # 确保输出目录存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 计算并保存 out；调用 frame.copy 完成本步处理。
    out = frame.copy()
    # 检查 'risk_signals' in out；满足条件才执行下面的分支。
    if "risk_signals" in out:
        # 把每个风险信号列表编码成 JSON 字符串，避免 CSV 丢失列表结构。
        out["risk_signals"] = out["risk_signals"].map(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
    # 将当前表格持久化为 CSV。
    out.to_csv(path, index=False)


# 命令行特征构建入口，同时保存数据来源及构建参数。
@app.command("build-features")
def build_features(
    output: Path = Path("data/features.csv"),
    start: str = "2025-01-01",
    end: str = "2025-12-01",
    data_end: str | None = None,
    labeled: bool = True,
):
    """Build point-in-time customer features."""
    # 计算并保存 frame；调用 build_dataset 完成本步处理。
    frame = build_dataset(Path("data/bundled"), start, end, data_end, labeled)
    # 把表格写入 CSV 文件。
    save_csv(frame, output)
    # 定位原始数据的来源元数据文件。
    source = Path("data/bundled/metadata.json")
    # metadata：数据来源、模型参数或评估结果的元数据。
    metadata = (
        json.loads(source.read_text()) if source.exists() else {"source": "external_raw_tables"}
    )
    # 合并本步骤补充的元数据字段。
    metadata.update(
        {
            "feature_start": start,
            "feature_end": end,
            "data_end_exclusive": data_end,
            "labeled": labeled,
            "rows": len(frame),
        }
    )
    # 把结果写入 JSON 文件。
    write_json(output.with_suffix(".metadata.json"), metadata)
    # 向终端显示执行结果或输出路径。
    typer.echo(f"{output}: {len(frame)} rows")


# 从特征文件及其元数据发起训练，并打印新模型版本目录。
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
    # 把特征 CSV 后缀改成 .metadata.json，找到配套来源记录。
    metadata_path = features.with_suffix(".metadata.json")
    # metadata：数据来源、模型参数或评估结果的元数据。
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    # 计算并保存 output；调用 train 完成本步处理。
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
    # 向终端显示执行结果或输出路径。
    typer.echo(str(output))


# 批量评分入口，将客户风险结果写入指定文件。
@app.command("score")
def score(
    features: Annotated[Path, typer.Option()],
    output: Path = Path("reports/scores.csv"),
    model: Path = Path("artifacts"),
):
    """Score feature rows with a persisted model."""
    # 把表格写入 CSV 文件。
    save_csv(score_customers(read_features(features), load_model(model)), output)
    # 向终端显示执行结果或输出路径。
    typer.echo(str(output))


# 用已有模型评估带成熟标签的数据，不重新训练模型。
@app.command("evaluate")
def evaluate(
    features: Annotated[Path, typer.Option()],
    output: Path = Path("reports/evaluation.json"),
    model: Path = Path("artifacts"),
    top_fraction: float = 0.1,
):
    """Evaluate a persisted model on labeled feature rows."""
    # 同时准备经过标签校验的特征和已持久化的模型。
    frame, bundle = read_features(features, True), load_model(model)
    # 计算并保存 scored；调用 score_customers 完成本步处理。
    scored = score_customers(frame, bundle)
    # 计算并保存 report；调用 classification_metrics 完成本步处理。
    report = classification_metrics(
        frame["churn"], scored["risk_score"], bundle["metadata"]["thresholds"]["high"], top_fraction
    )
    # 把结果写入 JSON 文件。
    write_json(output, {"model_version": bundle["metadata"]["model_version"], "metrics": report})
    # 向终端显示执行结果或输出路径。
    typer.echo(str(output))


# 用模型保存的训练参考分布生成当前批次的漂移报告。
@app.command("monitor")
def monitor(
    features: Annotated[Path, typer.Option()],
    output: Path = Path("reports/drift.json"),
    model: Path = Path("artifacts"),
):
    """Measure feature distribution drift against the training reference."""
    # bundle：模型管道、元数据和训练参考分布组成的工件包。
    bundle = load_model(model)
    # 计算并保存 report；调用 drift_report 完成本步处理。
    report = drift_report(read_features(features), bundle["reference"])
    # 在报告中附上实际使用的模型版本，方便后续追溯。
    report["model_version"] = bundle["metadata"]["model_version"]
    # 把结果写入 JSON 文件。
    write_json(output, report)
    # 向终端显示执行结果或输出路径。
    typer.echo(str(output))


# 读取评分及联系资格，导出受容量限制的挽留名单。
@app.command("retention-list")
def export_retention(
    scores: Annotated[Path, typer.Option()],
    eligibility: Annotated[Path, typer.Option()],
    capacity: int = 100,
    output: Path = Path("reports/retention.csv"),
    cooldown_days: int = 14,
):
    """Export high-risk customers satisfying contact and cooldown constraints."""
    # 计算并保存 scored；读取 CSV；由扩展名自动识别 gzip 压缩。
    scored = pd.read_csv(scores, dtype={"customer_id": str})
    # 把 CSV 中的 JSON 字符串恢复为风险信号列表。
    scored["risk_signals"] = scored["risk_signals"].map(json.loads)
    # 读取联系资格表，保留客户编号的字符串类型。
    allowed = pd.read_csv(eligibility, dtype={"customer_id": str})
    # 把表格写入 CSV 文件。
    save_csv(retention_list(scored, allowed, capacity, cooldown_days), output)
    # 向终端显示执行结果或输出路径。
    typer.echo(str(output))


# 串联固定数据的特征构建、训练、最新评分、监控和名单导出，不生成原始数据。
@app.command("demo")
def demo(seed: int = 42, algorithm: str = "lightgbm"):
    """Run the integrated data, training, scoring and monitoring workflow."""
    # 构建指定月份的特征并写入 CSV 与来源元数据。
    build_features(data_end="2026-02-01")
    # 训练模型并保存独立版本。
    train_model(algorithm=algorithm, seed=seed)
    # 构建指定月份的特征并写入 CSV 与来源元数据。
    build_features(
        output=Path("data/latest.csv"), start="2026-01-01", end="2026-01-01", labeled=False
    )
    # latest：最新评分批次的客户特征。
    latest = read_features(Path("data/latest.csv"))
    # 把表格写入 CSV 文件。
    save_csv(latest, Path("data/latest.csv"))
    # payload：准备提交给预测接口的数据结构。
    payload = latest.head(3).copy()
    # 把 Pandas 时间戳转成 YYYY-MM-DD，满足 JSON 日期传输格式。
    payload["snapshot_date"] = payload["snapshot_date"].dt.strftime("%Y-%m-%d")
    # payload：准备提交给预测接口的数据结构。
    payload = payload.drop(columns=["churn", "label_end_date"], errors="ignore")
    # 把结果写入 JSON 文件。
    write_json(Path("data/request.json"), {"customers": payload.to_dict("records")})
    # 对最新特征执行批量评分。
    score(Path("data/latest.csv"))
    # 输出最新批次的漂移报告。
    monitor(Path("data/latest.csv"))
    # eligibility：供名单筛选使用的联系资格表。
    eligibility = latest[["customer_id"]].copy()
    # 流程演练将客户设为允许触达；接入业务时必须替换为真实授权记录。
    eligibility["contact_allowed"] = 1
    # 流程演练留空最近触达时间；业务中应使用实际触达记录。
    eligibility["last_contact_at"] = None
    # 把表格写入 CSV 文件。
    save_csv(eligibility, Path("data/eligibility.csv"))
    # 输出满足联系条件的高风险名单。
    export_retention(Path("reports/scores.csv"), Path("data/eligibility.csv"))


# 启动 Uvicorn，由应用工厂创建 API 服务。
@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the scoring API and operations console."""
    import uvicorn

    # 调用 logging.basicConfig 完成本步处理。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # 调用 uvicorn.run 完成本步处理。
    uvicorn.run("telecom_churn.api:create_app", factory=True, host=host, port=port)


# 检查 __name__ == '__main__'；满足条件才执行下面的分支。
if __name__ == "__main__":
    # 调用 app 完成本步处理。
    app()
