import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telecom_churn.metrics import classification_metrics
from telecom_churn.monitoring import make_reference
from telecom_churn.schema import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    SCHEMA_VERSION,
    validate_features,
)


# 按时间切分，并剔除标签窗口跨越下一分区起点的记录。
def temporal_split(frame: pd.DataFrame, validation_start: str, test_start: str) -> dict:
    # 计算并保存 frame；执行统一特征校验，避免离线与在线口径不同。
    frame = validate_features(frame, labeled=True)
    # 把验证和测试起始日解析为可比较的时间戳。
    val_date, test_date = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    # 检查 val_date >= test_date；满足条件才执行下面的分支。
    if val_date >= test_date:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("validation_start must precede test_start")
    # Labels must have fully matured before the next partition's first scoring date.
    # 选择验证开始前的快照，且要求其未来标签在验证开始前已完全成熟。
    train = frame[(frame["snapshot_date"] < val_date) & (frame["label_end_date"] <= val_date)]
    # 选择验证区间快照，且要求标签在测试开始前已完全成熟。
    validation = frame[
        (frame["snapshot_date"] >= val_date)
        & (frame["snapshot_date"] < test_date)
        & (frame["label_end_date"] <= test_date)
    ]
    # 选取测试起始日及之后的快照，保留作最终独立评估。
    test = frame[frame["snapshot_date"] >= test_date]
    # parts：经过时间边界清除的训练、验证和测试分区。
    parts = {"train": train, "validation": validation, "test": test}
    # 逐个处理 parts.items() 中的元素，当前值保存为 (name, part)。
    for name, part in parts.items():
        # 检查 part.empty or part['churn'].nunique() != 2；满足条件才执行下面的分支。
        if part.empty or part["churn"].nunique() != 2:
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError(f"{name} partition must contain both target classes")
    # 返回已经清除跨界标签的三个时间分区。
    return parts


# 将数值标准化、类别编码和分类器封装为同一个可保存的管道。
def make_pipeline(algorithm: str, seed: int) -> Pipeline:
    # preprocess：按列分别执行标准化和类别编码的预处理器。
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    # 检查 algorithm == 'logistic'；满足条件才执行下面的分支。
    if algorithm == "logistic":
        # estimator：当前选择的分类器及其超参数。
        estimator = LogisticRegression(C=0.5, max_iter=2000, random_state=seed)
    # 检查 algorithm == 'hist_gradient_boosting'；满足条件才执行下面的分支。
    elif algorithm == "hist_gradient_boosting":
        # estimator：当前选择的分类器及其超参数。
        estimator = HistGradientBoostingClassifier(
            max_iter=180,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=2,
            early_stopping=False,
            random_state=seed,
        )
    # 检查 algorithm == 'lightgbm'；满足条件才执行下面的分支。
    elif algorithm == "lightgbm":
        # estimator：当前选择的分类器及其超参数。
        estimator = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=5,
            min_child_samples=30,
            reg_lambda=2,
            verbosity=-1,
            n_jobs=2,
            random_state=seed,
            deterministic=True,
            force_col_wise=True,
        )
    else:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    # 按先预处理后分类器的顺序构造 Pipeline，保证训练与推理一致。
    return Pipeline([("preprocess", preprocess), ("classifier", estimator)])


# 创建父目录并写入可读 JSON；拒绝 NaN 和 Infinity 等非标准 JSON 值。
def write_json(path: Path, payload: dict):
    # 确保输出目录存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 path.write_text 完成本步处理。
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


# 在训练集拟合候选模型，验证集选型定阈值，最后仅评估选定模型的测试表现。
def train(
    frame: pd.DataFrame,
    artifacts: Path,
    validation_start: str = "2025-08-01",
    test_start: str = "2025-10-01",
    algorithm: str = "lightgbm",
    seed: int = 42,
    top_fraction: float = 0.1,
    source_metadata: dict | None = None,
) -> Path:
    # 检查 not 0 < top_fraction < 0.5；满足条件才执行下面的分支。
    if not 0 < top_fraction < 0.5:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("top_fraction must be between 0 and 0.5")
    # parts：经过时间边界清除的训练、验证和测试分区。
    parts = temporal_split(frame, validation_start, test_start)
    # 解包三个时间分区，后续分别用于拟合、选型和最终评估。
    tr, val, test = (parts[n] for n in ["train", "validation", "test"])
    # 建立候选模型缓存和验证指标字典，避免重新拟合获胜模型。
    models, comparison = {}, {}
    # 逐个处理 dict.fromkeys(['logistic', algorithm]) 中的元素，当前值保存为 name。
    for name in dict.fromkeys(["logistic", algorithm]):
        # 按候选算法创建全新的预处理及分类器管道。
        pipeline = make_pipeline(name, seed)
        # 仅使用训练分区拟合预处理参数和分类器。
        pipeline.fit(tr[FEATURES], tr["churn"])
        # 计算并保存 scores；按右侧表达式完成取列、计算或结果构造。
        scores = pipeline.predict_proba(val[FEATURES])[:, 1]
        # 取验证分数的 1-top_fraction 分位数，作为候选高风险阈值。
        threshold = float(np.quantile(scores, 1 - top_fraction))
        # 保存当前候选的验证指标，用于按 AP 选择模型。
        comparison[name] = classification_metrics(val["churn"], scores, threshold, top_fraction)
        # 缓存已经训练好的完整管道，后续直接取获胜者。
        models[name] = pipeline
    # winner：验证集 Average Precision 最大的候选模型名。
    winner = max(comparison, key=lambda name: comparison[name]["average_precision"])
    # model：选中的完整预处理和分类器管道。
    model = models[winner]
    # high：验证集分数确定的高风险阈值。
    high = comparison[winner]["threshold"]
    # medium：验证集分数中位数确定的中风险阈值。
    medium = float(np.quantile(model.predict_proba(val[FEATURES])[:, 1], 0.5))
    # test_scores：最终模型对独立测试集的正类分数。
    test_scores = model.predict_proba(test[FEATURES])[:, 1]
    # test_metrics：独立测试集的评估指标。
    test_metrics = classification_metrics(test["churn"], test_scores, high, top_fraction)
    # importance：在验证集上打乱特征得到的置换重要性。
    importance = permutation_importance(
        model,
        val[FEATURES],
        val["churn"],
        scoring="average_precision",
        n_repeats=3,
        random_state=seed,
        n_jobs=1,
    )
    # 计算并保存 version；按右侧表达式完成取列、计算或结果构造。
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    # 计算并保存 output；按右侧表达式完成取列、计算或结果构造。
    output = artifacts / version
    # 确保输出目录存在。
    output.mkdir(parents=True, exist_ok=False)
    # fingerprint：排序后训练输入内容的 SHA-256 指纹。
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(
            frame.sort_values(["snapshot_date", "customer_id"])[
                ["customer_id", "snapshot_date", *FEATURES, "churn"]
            ],
            index=False,
        ).values.tobytes()
    ).hexdigest()
    # versions：训练时实际安装的关键依赖版本。
    versions = {
        name: importlib.metadata.version(name)
        for name in ["pandas", "numpy", "scikit-learn", "joblib"]
    }
    # 检查 'lightgbm' in models；满足条件才执行下面的分支。
    if "lightgbm" in models:
        # 仅在实际训练 LightGBM 时记录其安装版本。
        versions["lightgbm"] = importlib.metadata.version("lightgbm")
    # metadata：数据来源、模型参数或评估结果的元数据。
    metadata = {
        "model_version": version,
        "schema_version": SCHEMA_VERSION,
        "selected_algorithm": winner,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "features": FEATURES,
        "thresholds": {"high": high, "medium": medium},
        "selection_metric": "validation_average_precision",
        "validation_comparison": comparison,
        "test_metrics": test_metrics,
        "test_by_month": {
            str(date.date()): classification_metrics(
                group["churn"], model.predict_proba(group[FEATURES])[:, 1], high, top_fraction
            )
            for date, group in test.groupby("snapshot_date")
        },
        "partitions": {
            name: {
                "rows": len(p),
                "start": str(p["snapshot_date"].min().date()),
                "end": str(p["snapshot_date"].max().date()),
                "positives": int(p["churn"].sum()),
            }
            for name, p in parts.items()
        },
        "purged_rows": len(frame) - sum(len(p) for p in parts.values()),
        "data_fingerprint": fingerprint,
        "data_source": source_metadata or {"source": "external_feature_table"},
        "python_version": platform.python_version(),
        "dependencies": versions,
        "permutation_importance": sorted(
            [
                {"feature": col, "mean": float(mean), "std": float(std)}
                for col, mean, std in zip(
                    FEATURES, importance.importances_mean, importance.importances_std, strict=True
                )
            ],
            key=lambda row: row["mean"],
            reverse=True,
        ),
    }
    # bundle：模型管道、元数据和训练参考分布组成的工件包。
    bundle = {"model": model, "metadata": metadata, "reference": make_reference(tr)}
    # 计算并保存 model_path；按右侧表达式完成取列、计算或结果构造。
    model_path = output / "model.joblib"
    # 把模型管道及参考分布保存到版本目录。
    joblib.dump(bundle, model_path)
    # 对模型文件字节计算 SHA-256；加载时重算以检测损坏。
    metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    # 把结果写入 JSON 文件。
    write_json(output / "metadata.json", metadata)
    # scored_test：保留真实标签以便复核的测试预测明细。
    scored_test = test[["customer_id", "snapshot_date", "churn"]].copy()
    # 把测试集正类分数加入预测明细，便于逐条复核。
    scored_test["risk_score"] = test_scores
    # 将当前表格持久化为 CSV。
    scored_test.to_csv(output / "test_predictions.csv", index=False)
    # Promote only after all immutable artifacts have been written successfully.
    # pointer：原子更新当前版本时使用的临时指针文件。
    pointer = artifacts / f".latest-{uuid4().hex}.tmp"
    # 把结果写入 JSON 文件。
    write_json(pointer, {"model_version": version})
    # 在所有模型文件写入成功后，原子替换当前版本指针。
    os.replace(pointer, artifacts / "latest.json")
    # 返回本次新建的模型版本目录。
    return output


# 把模型根目录、版本目录或文件路径统一解析成 model.joblib。
def resolve_model(path: Path) -> Path:
    # 检查 path.is_file()；满足条件才执行下面的分支。
    if path.is_file():
        # 检查 path.suffix != '.joblib'；满足条件才执行下面的分支。
        if path.suffix != ".joblib":
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError("Expected a model.joblib file or artifact directory")
        # 返回本步骤的处理结果给调用方。
        return path
    # 检查 (path / 'model.joblib').is_file()；满足条件才执行下面的分支。
    if (path / "model.joblib").is_file():
        # 返回本步骤的处理结果给调用方。
        return path / "model.joblib"
    # pointer：原子更新当前版本时使用的临时指针文件。
    pointer = json.loads((path / "latest.json").read_text(encoding="utf-8"))
    # 计算并保存 version；按右侧表达式完成取列、计算或结果构造。
    version = pointer["model_version"]
    # 检查 not isinstance(version, str) or Path(version).name != version or version in {'.', '..'}；满足条件才执行下面的分支。
    if not isinstance(version, str) or Path(version).name != version or version in {".", ".."}:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Invalid model version pointer")
    # 返回本步骤的处理结果给调用方。
    return path / version / "model.joblib"


# 检查模型校验和与特征版本，再从可信目录载入模型。
def load_model(path: Path) -> dict:
    # 计算并保存 model_path；调用 resolve_model 完成本步处理。
    model_path = resolve_model(path)
    # metadata：数据来源、模型参数或评估结果的元数据。
    metadata = json.loads(model_path.with_name("metadata.json").read_text(encoding="utf-8"))
    # 检查 hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata['model_sha256']；满足条件才执行下面的分支。
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Model artifact checksum mismatch")
    # 检查 metadata['schema_version'] != SCHEMA_VERSION or metadata['features'] != FEATURES；满足条件才执行下面的分支。
    if metadata["schema_version"] != SCHEMA_VERSION or metadata["features"] != FEATURES:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Model feature schema does not match the application")
    # Joblib is an executable serialization format; artifact directories must be trusted.
    # bundle：模型管道、元数据和训练参考分布组成的工件包。
    bundle = joblib.load(model_path)
    # 检查 bundle['metadata']['model_version'] != metadata['model_version']；满足条件才执行下面的分支。
    if bundle["metadata"]["model_version"] != metadata["model_version"]:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Model version mismatch")
    # 使用外部 metadata.json 中包含校验和的完整元数据替换包内副本。
    bundle["metadata"] = metadata
    # 返回通过文件校验和版本校验的模型包。
    return bundle
