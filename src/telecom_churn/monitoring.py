# 导入 np，供本模块后续逻辑使用。
import numpy as np

# 导入 pd，供本模块后续逻辑使用。
import pandas as pd

# 导入 CATEGORICAL_FEATURES, NUMERIC_FEATURES，供本模块后续逻辑使用。
from telecom_churn.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


# 计算加平滑的直方图概率，避免零概率导致对数无定义。
def _distribution(values, edges):
    # 按固定区间统计样本数；下划线表示不再使用直方图返回的边界。
    counts, _ = np.histogram(values, bins=edges)
    # 给每个箱的计数加 0.5，避免 PSI 计算出现零概率和无穷对数。
    smooth = counts.astype(float) + 0.5
    # 把平滑计数除以总和，返回总概率为 1 的分布。
    return smooth / smooth.sum()


# 仅从训练集记录数值分箱和类别比例，作为后续漂移比较基准。
def make_reference(frame: pd.DataFrame) -> dict:
    # reference：训练时的数值分箱和类别概率基准。
    reference = {"numeric": {}, "categorical": {}, "rows": len(frame)}
    # 逐个处理 NUMERIC_FEATURES 中的元素，当前值保存为 column。
    for column in NUMERIC_FEATURES:
        # 计算并保存 values；调用 frame[column].to_numpy 完成本步处理。
        values = frame[column].to_numpy(dtype=float)
        # interior：数值分箱的内部切分点。
        interior = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        # Include extrema so a formerly constant feature can still register drift.
        # interior：数值分箱的内部切分点。
        interior = np.unique(np.r_[interior, np.nextafter(interior[-1], np.inf)])
        # edges：加入正负无穷边界后的直方图区间。
        edges = np.r_[-np.inf, interior, np.inf]
        # 保存该特征的分箱点和训练样本占比；线上复用同一组箱。
        reference["numeric"][column] = {
            "cuts": interior.tolist(),
            "proportions": _distribution(values, edges).tolist(),
        }
    # 逐个处理 CATEGORICAL_FEATURES 中的元素，当前值保存为 column。
    for column in CATEGORICAL_FEATURES:
        # 把各类别出现频率归一化为概率，并保存为字典。
        reference["categorical"][column] = frame[column].value_counts(normalize=True).to_dict()
    # 返回数值和类别分布基准，随模型一起保存。
    return reference


# 比较当前数据与训练参考分布，给出逐特征漂移状态。
def drift_report(frame: pd.DataFrame, reference: dict) -> dict:
    # 检查 frame.empty；满足条件才执行下面的分支。
    if frame.empty:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Cannot monitor an empty dataset")
    # 初始化逐特征漂移结果列表。
    results = []
    # 逐个处理 reference['numeric'].items() 中的元素，当前值保存为 (column, baseline)。
    for column, baseline in reference["numeric"].items():
        # 计算并保存 actual；调用 _distribution 完成本步处理。
        actual = _distribution(
            frame[column].to_numpy(dtype=float), np.r_[-np.inf, baseline["cuts"], np.inf]
        )
        # expected：训练参考分布概率。
        expected = np.asarray(baseline["proportions"])
        # psi：当前数值分布相对训练分布的 PSI。
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        # 把当前结果追加到列表，供稍后统一汇总。
        results.append(
            {
                "feature": column,
                "metric": "psi",
                "value": psi,
                "status": "alert" if psi >= 0.25 else "watch" if psi >= 0.1 else "stable",
            }
        )
    # 逐个处理 reference['categorical'].items() 中的元素，当前值保存为 (column, baseline)。
    for column, baseline in reference["categorical"].items():
        # 计算并保存 actual；调用 frame[column].value_counts(normalize=True).to_dict 完成本步处理。
        actual = frame[column].value_counts(normalize=True).to_dict()
        # 合并当前和参考类别的集合，既比较消失类别，也比较新出现类别。
        keys = set(actual) | set(baseline)
        # distance：类别概率分布的总变差距离。
        distance = sum(abs(actual.get(k, 0) - baseline.get(k, 0)) for k in keys) / 2
        # 把当前结果追加到列表，供稍后统一汇总。
        results.append(
            {
                "feature": column,
                "metric": "total_variation",
                "value": distance,
                "unknown_rate": sum(v for k, v in actual.items() if k not in baseline),
                "status": "alert" if distance >= 0.2 else "watch" if distance >= 0.1 else "stable",
            }
        )
    # 返回样本量、小样本标记、告警数量和逐特征报告。
    return {
        "rows": len(frame),
        "small_sample": len(frame) < 200,
        "alert_count": sum(row["status"] == "alert" for row in results),
        "features": results,
    }
