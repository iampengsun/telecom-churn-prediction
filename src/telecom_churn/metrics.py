# 导入 math，供本模块后续逻辑使用。
import math

# 导入 np，供本模块后续逻辑使用。
import numpy as np

# 导入 average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score, roc_auc_score，供本模块后续逻辑使用。
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


# 同时评估整体分类质量和有限触达容量下的名单排序质量。
def classification_metrics(y, score, threshold: float, top_fraction: float = 0.1) -> dict:
    # 把标签转为整数数组、风险分数转为浮点数组，统一后续计算类型。
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    # 检查 len(y) == 0 or len(y) != len(score)；满足条件才执行下面的分支。
    if len(y) == 0 or len(y) != len(score):
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Labels and scores must have equal nonzero lengths")
    # 检查 not 0 < top_fraction <= 1；满足条件才执行下面的分支。
    if not 0 < top_fraction <= 1:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("top_fraction must be in (0, 1]")
    # 检查 not np.isfinite(score).all() or ((score < 0) | (score > 1)).any()；满足条件才执行下面的分支。
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Scores must be finite values in [0, 1]")
    # 检查 not np.isin(y, [0, 1]).all()；满足条件才执行下面的分支。
    if not np.isin(y, [0, 1]).all():
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Labels must be binary")
    # k：按触达比例向上取整、且至少为 1 的名单人数。
    k = max(1, math.ceil(len(y) * top_fraction))
    # order：风险分数从高到低排列的前 K 个样本下标。
    order = np.argsort(-score, kind="stable")[:k]
    # 分别统计流失客户总数和整体流失率，作为召回率与 Lift 的分母。
    positives, base_rate = int(y.sum()), float(y.mean())
    # precision_k：前 K 位客户中实际流失的比例。
    precision_k = float(y[order].mean())
    # predicted：依据固定阈值得到的二分类预测。
    predicted = score >= threshold
    # two_classes：当前样本是否同时包含正负类。
    two_classes = len(np.unique(y)) == 2
    # 返回分类、校准误差和 Top-K 排序指标；单类时无法定义的指标返回 None。
    return {
        "rows": len(y),
        "positives": positives,
        "churn_rate": base_rate,
        "roc_auc": float(roc_auc_score(y, score)) if two_classes else None,
        "average_precision": float(average_precision_score(y, score)) if positives else None,
        "brier_score": float(brier_score_loss(y, score)),
        "threshold": threshold,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "top_k": k,
        "top_fraction": top_fraction,
        "precision_at_k": precision_k,
        "recall_at_k": float(y[order].sum() / positives) if positives else None,
        "lift_at_k": precision_k / base_rate if positives else None,
    }
