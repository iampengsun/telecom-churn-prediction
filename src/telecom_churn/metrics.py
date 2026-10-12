import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y, score, threshold: float, top_fraction: float = 0.1) -> dict:
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    if len(y) == 0 or len(y) != len(score):
        raise ValueError("Labels and scores must have equal nonzero lengths")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Scores must be finite values in [0, 1]")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")
    k = max(1, math.ceil(len(y) * top_fraction))
    order = np.argsort(-score, kind="stable")[:k]
    positives, base_rate = int(y.sum()), float(y.mean())
    precision_k = float(y[order].mean())
    predicted = score >= threshold
    two_classes = len(np.unique(y)) == 2
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
