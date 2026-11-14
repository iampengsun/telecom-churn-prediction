import pandas as pd

from telecom_churn.schema import FEATURES, validate_features


# 根据明确的业务规则生成可读信号；这些信号不是模型的因果解释。
def risk_signals(row: pd.Series) -> list[str]:
    # signals：当前客户命中的业务规则信号。
    signals = []
    # 检查 row['usage_trend'] < 0.6 and row['usage_mb_90d'] > 0；满足条件才执行下面的分支。
    if row["usage_trend"] < 0.6 and row["usage_mb_90d"] > 0:
        # 把当前结果追加到列表，供稍后统一汇总。
        signals.append("近期流量使用下降")
    # 检查 row['complaints_30d'] >= 2；满足条件才执行下面的分支。
    if row["complaints_30d"] >= 2:
        # 把当前结果追加到列表，供稍后统一汇总。
        signals.append("近期多次服务投诉")
    # 检查 0 < row['contract_days_remaining'] <= 30；满足条件才执行下面的分支。
    if 0 < row["contract_days_remaining"] <= 30:
        # 把当前结果追加到列表，供稍后统一汇总。
        signals.append("合约即将到期")
    # 检查 row['overdue_invoices'] > 0；满足条件才执行下面的分支。
    if row["overdue_invoices"] > 0:
        # 把当前结果追加到列表，供稍后统一汇总。
        signals.append("存在逾期账单")
    # 检查 row['active_days_30d'] < 15；满足条件才执行下面的分支。
    if row["active_days_30d"] < 15:
        # 把当前结果追加到列表，供稍后统一汇总。
        signals.append("近期活跃度偏低")
    # 返回该客户命中的规则文本列表，未命中时为空列表。
    return signals


# 用已保存的预处理与分类器评分，并复用验证集确定的固定阈值。
def score_customers(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    # 计算并保存 frame；执行统一特征校验，避免离线与在线口径不同。
    frame = validate_features(frame)
    # 计算并保存 scores；按右侧表达式完成取列、计算或结果构造。
    scores = bundle["model"].predict_proba(frame[FEATURES])[:, 1]
    # 从模型包读取固定阈值及版本信息，评分时不重新拟合。
    meta = bundle["metadata"]
    # thresholds：训练时已确定的高、中风险阈值。
    thresholds = meta["thresholds"]
    # 初始化逐客户结果列表，最后一次性转换成 DataFrame。
    rows = []
    # 逐个处理 zip(frame.iterrows(), scores, strict=True) 中的元素，当前值保存为 ((_, row), score)。
    for (_, row), score in zip(frame.iterrows(), scores, strict=True):
        # level：当前客户的高、中、低风险等级。
        level = (
            "high"
            if score >= thresholds["high"]
            else "medium"
            if score >= thresholds["medium"]
            else "low"
        )
        # signals：当前客户命中的业务规则信号。
        signals = risk_signals(row)
        # 检查 level == 'low'；满足条件才执行下面的分支。
        if level == "low":
            # action：依据风险和业务信号推荐的回访动作。
            action = "常规服务维护"
        # 检查 '近期多次服务投诉' in signals；满足条件才执行下面的分支。
        elif "近期多次服务投诉" in signals:
            # action：依据风险和业务信号推荐的回访动作。
            action = "优先服务回访"
        # 检查 '存在逾期账单' in signals；满足条件才执行下面的分支。
        elif "存在逾期账单" in signals:
            # action：依据风险和业务信号推荐的回访动作。
            action = "账单核查与服务提醒"
        # 检查 '合约即将到期' in signals；满足条件才执行下面的分支。
        elif "合约即将到期" in signals:
            # action：依据风险和业务信号推荐的回访动作。
            action = "合约续约沟通"
        else:
            # action：依据风险和业务信号推荐的回访动作。
            action = "套餐使用情况回访"
        # 把当前结果追加到列表，供稍后统一汇总。
        rows.append(
            {
                "customer_id": row["customer_id"],
                "snapshot_date": str(row["snapshot_date"].date()),
                "risk_score": float(score),
                "risk_level": level,
                "risk_signals": signals,
                "suggested_action": action,
                "model_version": meta["model_version"],
            }
        )
    # 把逐客户评分记录转换成 DataFrame，交给 CLI 或 API。
    return pd.DataFrame(rows)


# 按授权、冷却期和容量筛选高风险客户；缺少授权记录的客户不会入选。
def retention_list(
    scores: pd.DataFrame, eligibility: pd.DataFrame, capacity: int, cooldown_days: int = 14
) -> pd.DataFrame:
    # 检查 capacity < 1 or cooldown_days < 0；满足条件才执行下面的分支。
    if capacity < 1 or cooldown_days < 0:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("capacity must be positive and cooldown_days nonnegative")
    # 检查 scores['snapshot_date'].nunique() != 1；满足条件才执行下面的分支。
    if scores["snapshot_date"].nunique() != 1:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Retention export requires exactly one snapshot date")
    # 检查 scores['customer_id'].duplicated().any()；满足条件才执行下面的分支。
    if scores["customer_id"].duplicated().any():
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Duplicate customer_id in scores")
    # required：当前处理步骤必须具备的字段集合。
    required = {"customer_id", "contact_allowed", "last_contact_at"}
    # 检查 not required.issubset(eligibility.columns)；满足条件才执行下面的分支。
    if not required.issubset(eligibility.columns):
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError(f"Eligibility requires {sorted(required)}")
    # 检查 eligibility['customer_id'].duplicated().any()；满足条件才执行下面的分支。
    if eligibility["customer_id"].duplicated().any():
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Duplicate customer_id in eligibility")
    # 检查 eligibility['contact_allowed'].isna().any() or not eligibility['contact_allowed'].isin([0, 1]).all()；满足条件才执行下面的分支。
    if (
        eligibility["contact_allowed"].isna().any()
        or not eligibility["contact_allowed"].isin([0, 1]).all()
    ):
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("contact_allowed must be 0 or 1")
    # joined：评分名单和联系资格按客户编号一对一内连接的结果。
    joined = scores.merge(
        eligibility[list(required)], on="customer_id", how="inner", validate="one_to_one"
    )
    # last：解析为日期的上次实际触达时间。
    last = pd.to_datetime(joined["last_contact_at"], errors="raise")
    # as_of：本次评分日期。
    as_of = pd.to_datetime(joined["snapshot_date"])
    # eligible：满足授权、冷却期和高风险条件的候选名单。
    eligible = joined[
        (joined["contact_allowed"] == 1)
        & (last.isna() | ((as_of - last).dt.days >= cooldown_days))
        & (joined["risk_level"] == "high")
    ].copy()
    # eligible：满足授权、冷却期和高风险条件的候选名单。
    eligible = eligible.sort_values(["risk_score", "customer_id"], ascending=[False, True]).head(
        capacity
    )
    # 调用 eligible.insert 完成本步处理。
    eligible.insert(0, "priority", range(1, len(eligible) + 1))
    # 去掉内部授权筛选字段，只保留业务名单及优先级。
    return eligible.drop(columns=["contact_allowed", "last_contact_at"])
