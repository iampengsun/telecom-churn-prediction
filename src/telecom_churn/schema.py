# 导入 date，供本模块后续逻辑使用。
from datetime import date

# 导入 Literal，供本模块后续逻辑使用。
from typing import Literal

# 导入 np，供本模块后续逻辑使用。
import numpy as np

# 导入 pd，供本模块后续逻辑使用。
import pandas as pd

# 导入 BaseModel, ConfigDict, Field, model_validator，供本模块后续逻辑使用。
from pydantic import BaseModel, ConfigDict, Field, model_validator

# SCHEMA_VERSION：特征契约版本，加载旧模型时用于一致性检查。
SCHEMA_VERSION = "1.0"
# HORIZON_DAYS：未来流失观察窗口天数。
HORIZON_DAYS = 30
# NUMERIC_FEATURES：参与训练的数值特征名单。
NUMERIC_FEATURES = [
    "tenure_days",
    "monthly_fee",
    "contract_days_remaining",
    "usage_mb_30d",
    "usage_mb_90d",
    "usage_trend",
    "call_minutes_30d",
    "active_days_30d",
    "complaints_30d",
    "complaints_90d",
    "overdue_invoices",
    "bill_amount_90d",
]
# CATEGORICAL_FEATURES：需要独热编码的类别特征名单。
CATEGORICAL_FEATURES = ["plan_type", "region"]
# FEATURES：模型输入列的固定顺序。
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# CHURN_TYPES：算作主动流失的终止事件类型。
CHURN_TYPES = {"voluntary_close", "port_out"}


# 定义单客户特征及每个字段的业务范围；Field 的 ge/le 分别限定最小值和最大值。
class CustomerSnapshot(BaseModel):
    # model_config：Pydantic 的输入规则，禁止多余字段或非法浮点数。
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # 客户唯一标识，只允许字母、数字、下划线和短横线，长度 1–80。
    customer_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    # 评分时点日期；特征只能使用当日零点之前的信息。
    snapshot_date: date
    # 在网天数，限制在 0–36,500 天。
    tenure_days: int = Field(ge=0, le=36500)
    # 套餐月费，单位元，范围 0–10,000。
    monthly_fee: float = Field(ge=0, le=10000)
    # 合约剩余天数，已到期为 0，上限 36,500。
    contract_days_remaining: int = Field(ge=0, le=36500)
    # 近 30 天累计流量，单位 MB，不能为负。
    usage_mb_30d: float = Field(ge=0)
    # 近 90 天累计流量，单位 MB，不能为负。
    usage_mb_90d: float = Field(ge=0)
    # 近 30 天流量相对近 90 天月均量的比例，理论范围 0–3。
    usage_trend: float = Field(ge=0, le=3.000001)
    # 近 30 天累计通话分钟数，上限为 30×24×60。
    call_minutes_30d: float = Field(ge=0, le=43200)
    # 近 30 天有流量或通话的日期数，范围 0–30。
    active_days_30d: int = Field(ge=0, le=30)
    # 近 30 天投诉工单数量，非负整数。
    complaints_30d: int = Field(ge=0)
    # 近 90 天投诉工单数量，非负整数。
    complaints_90d: int = Field(ge=0)
    # 评分时已逾期且仍未支付的账单数。
    overdue_invoices: int = Field(ge=0)
    # 近 90 天出账金额之和，单位元，不能为负。
    bill_amount_90d: float = Field(ge=0)
    # 套餐类别，仅接受 basic、standard 或 premium。
    plan_type: Literal["basic", "standard", "premium"]
    # 客户所属业务区域，字符串长度 1–40。
    region: str = Field(min_length=1, max_length=40)

    # 校验窗口之间的业务约束，防止累计量和趋势字段互相矛盾。
    @model_validator(mode="after")
    def check_windows(self):
        # 检查 self.usage_mb_30d > self.usage_mb_90d + 1e-06；满足条件才执行下面的分支。
        if self.usage_mb_30d > self.usage_mb_90d + 1e-6:
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError("usage_mb_30d must not exceed usage_mb_90d")
        # expected：按窗口定义计算应有的趋势值或标签成熟日期。
        expected = self.usage_mb_30d * 3 / self.usage_mb_90d if self.usage_mb_90d else 0.0
        # 检查 abs(self.usage_trend - expected) > 1e-05；满足条件才执行下面的分支。
        if abs(self.usage_trend - expected) > 1e-5:
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError("usage_trend must equal 3 * usage_mb_30d / usage_mb_90d (0 if empty)")
        # 检查 self.complaints_30d > self.complaints_90d；满足条件才执行下面的分支。
        if self.complaints_30d > self.complaints_90d:
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError("complaints_30d must not exceed complaints_90d")
        # 返回校验通过的当前 Pydantic 对象。
        return self


# 定义一次批量预测请求：客户条数必须在 1 到 1,000 之间。
class PredictRequest(BaseModel):
    # model_config：Pydantic 的输入规则，禁止多余字段或非法浮点数。
    model_config = ConfigDict(extra="forbid")
    # customers：声明为 list[CustomerSnapshot]，并设置字段范围或默认配置。
    customers: list[CustomerSnapshot] = Field(min_length=1, max_length=1000)


# 定义接口返回字段，让输出结构保持稳定。
class Prediction(BaseModel):
    # 客户唯一标识，只允许字母、数字、下划线和短横线，长度 1–80。
    customer_id: str
    # 评分时点日期；特征只能使用当日零点之前的信息。
    snapshot_date: str
    # 模型输出的流失类别分数，用于排序和风险分层。
    risk_score: float
    # 保存阈值划分的高、中、低三种风险等级。
    risk_level: Literal["high", "medium", "low"]
    # 规则命中的业务信号列表，可能为空。
    risk_signals: list[str]
    # 根据风险等级与业务信号给出的服务动作建议。
    suggested_action: str
    # 生成本条结果时使用的持久化模型版本。
    model_version: str


# 统一校验离线表和接口输入；返回转换过日期类型的副本，不修改调用方数据。
def validate_features(frame: pd.DataFrame, labeled: bool = False) -> pd.DataFrame:
    # required：当前处理步骤必须具备的字段集合。
    required = {"customer_id", "snapshot_date", *FEATURES}
    # 检查 labeled；满足条件才执行下面的分支。
    if labeled:
        required |= {"churn", "label_end_date"}
    # missing：输入中缺少的字段。
    missing = required - set(frame.columns)
    # 检查 missing；满足条件才执行下面的分支。
    if missing:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError(f"Missing columns: {sorted(missing)}")
    # 检查 frame.empty；满足条件才执行下面的分支。
    if frame.empty:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Feature dataset is empty")
    # result：校验和转换后的结果副本。
    result = frame.copy()
    # 严格解析评分日期，非法日期立即报错。
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], errors="raise")
    # 检查 result['snapshot_date'].isna().any()；满足条件才执行下面的分支。
    if result["snapshot_date"].isna().any():
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("snapshot_date cannot be null")
    # 检查 result.duplicated(['customer_id', 'snapshot_date']).any()；满足条件才执行下面的分支。
    if result.duplicated(["customer_id", "snapshot_date"]).any():
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Duplicate customer_id / snapshot_date")
    # 逐个处理 result[['customer_id', 'snapshot_date', *FEATURES]].to_dict('records') 中的元素，当前值保存为 row。
    for row in result[["customer_id", "snapshot_date", *FEATURES]].to_dict("records"):
        # 调用 CustomerSnapshot.model_validate 完成本步处理。
        CustomerSnapshot.model_validate(row)
    # 检查 labeled；满足条件才执行下面的分支。
    if labeled:
        # 检查 not result['churn'].isin([0, 1]).all()；满足条件才执行下面的分支。
        if not result["churn"].isin([0, 1]).all():
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError("churn must be 0 or 1")
        # 严格解析标签成熟日期，随后校验是否等于评分日加 30 天。
        result["label_end_date"] = pd.to_datetime(result["label_end_date"], errors="raise")
        # expected：按窗口定义计算应有的趋势值或标签成熟日期。
        expected = result["snapshot_date"] + pd.Timedelta(days=HORIZON_DAYS)
        # 检查 not result['label_end_date'].eq(expected).all()；满足条件才执行下面的分支。
        if not result["label_end_date"].eq(expected).all():
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError("label_end_date must be snapshot_date + 30 days")
    # 检查 not np.isfinite(result[NUMERIC_FEATURES].to_numpy(dtype=float)).all()；满足条件才执行下面的分支。
    if not np.isfinite(result[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Numeric features must be finite")
    # 返回类型已整理、约束已校验的特征表副本。
    return result
