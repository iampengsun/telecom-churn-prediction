# 导入 Path，供本模块后续逻辑使用。
from pathlib import Path

# 导入 np，供本模块后续逻辑使用。
import numpy as np

# 导入 pd，供本模块后续逻辑使用。
import pandas as pd

# 导入 CHURN_TYPES, HORIZON_DAYS, validate_features，供本模块后续逻辑使用。
from telecom_churn.schema import CHURN_TYPES, HORIZON_DAYS, validate_features

# TABLE_COLUMNS：五张输入业务表分别需要的字段。
TABLE_COLUMNS = {
    "customers": ["customer_id", "joined_at", "contract_end", "monthly_fee", "plan_type", "region"],
    "usage": ["customer_id", "date", "data_mb", "call_minutes"],
    "invoices": ["invoice_id", "customer_id", "issued_at", "due_at", "paid_at", "amount"],
    "tickets": ["ticket_id", "customer_id", "opened_at"],
    "events": ["customer_id", "event_at", "event_type"],
}
# DATE_COLUMNS：每张表需要解析为日期的字段。
DATE_COLUMNS = {
    "customers": ["joined_at", "contract_end"],
    "usage": ["date"],
    "invoices": ["issued_at", "due_at", "paid_at"],
    "tickets": ["opened_at"],
    "events": ["event_at"],
}


# 读取五张业务表，检查必填字段、日期、主外键与金额范围。
def load_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    # tables：按表名索引的业务数据。
    tables = {}
    # 逐个处理 TABLE_COLUMNS.items() 中的元素，当前值保存为 (name, columns)。
    for name, columns in TABLE_COLUMNS.items():
        # 计算并保存 path；按右侧表达式完成取列、计算或结果构造。
        path = raw_dir / f"{name}.csv"
        # 检查 not path.exists()；满足条件才执行下面的分支。
        if not path.exists():
            # 计算并保存 path；按右侧表达式完成取列、计算或结果构造。
            path = raw_dir / f"{name}.csv.gz"
        # 计算并保存 frame；读取 CSV；由扩展名自动识别 gzip 压缩。
        frame = pd.read_csv(path, dtype={"customer_id": str})
        # missing：输入中缺少的字段。
        missing = set(columns) - set(frame.columns)
        # 检查 missing；满足条件才执行下面的分支。
        if missing:
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        # 逐个处理 DATE_COLUMNS[name] 中的元素，当前值保存为 col。
        for col in DATE_COLUMNS[name]:
            # 计算并保存 frame[col]；严格解析日期，非法日期直接报错。
            frame[col] = pd.to_datetime(frame[col], errors="raise")
            # 检查 col != 'paid_at' and frame[col].isna().any()；满足条件才执行下面的分支。
            if col != "paid_at" and frame[col].isna().any():
                # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
                raise ValueError(f"{name}.{col} cannot be null")
        # 检查 frame['customer_id'].isna().any()；满足条件才执行下面的分支。
        if frame["customer_id"].isna().any():
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError(f"{name}.customer_id cannot be null")
        # 把已解析并校验字段的表保存到字典，后续按表名取用。
        tables[name] = frame
    # 取出客户维表，用于外键校验和在网客户筛选。
    customers = tables["customers"]
    # 逐个处理 {'customers': ['customer_id'], 'usage': ['customer_id', 'date'], 'invoices': ['invoice_id'], 'tickets': ['ticket_id']}.items() 中的元素，当前值保存为 (name, keys)。
    for name, keys in {
        "customers": ["customer_id"],
        "usage": ["customer_id", "date"],
        "invoices": ["invoice_id"],
        "tickets": ["ticket_id"],
    }.items():
        # 检查 tables[name].duplicated(keys).any()；满足条件才执行下面的分支。
        if tables[name].duplicated(keys).any():
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError(f"{name}: duplicate primary key {keys}")
    # 逐个处理 tables.items() 中的元素，当前值保存为 (name, frame)。
    for name, frame in tables.items():
        # 检查 not frame['customer_id'].isin(customers['customer_id']).all()；满足条件才执行下面的分支。
        if not frame["customer_id"].isin(customers["customer_id"]).all():
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError(f"{name}: unknown customer_id")
    # 逐个处理 {'usage': ['data_mb', 'call_minutes'], 'invoices': ['amount'], 'customers': ['monthly_fee']}.items() 中的元素，当前值保存为 (name, cols)。
    for name, cols in {
        "usage": ["data_mb", "call_minutes"],
        "invoices": ["amount"],
        "customers": ["monthly_fee"],
    }.items():
        # 计算并保存 values；调用 tables[name][cols].to_numpy 完成本步处理。
        values = tables[name][cols].to_numpy(dtype=float)
        # 检查 not np.isfinite(values).all() or (values < 0).any()；满足条件才执行下面的分支。
        if not np.isfinite(values).all() or (values < 0).any():
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError(f"{name}: numeric values must be finite and nonnegative")
    # 取出账单表，继续检查出账、应付和实际支付日期关系。
    invoices = tables["invoices"]
    # 检查 (invoices['due_at'] < invoices['issued_at']).any()；满足条件才执行下面的分支。
    if (invoices["due_at"] < invoices["issued_at"]).any():
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Invoice due date precedes issue date")
    # 检查 (invoices['paid_at'] < invoices['issued_at']).any()；满足条件才执行下面的分支。
    if (invoices["paid_at"] < invoices["issued_at"]).any():
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("Invoice payment date precedes issue date")
    # 返回通过校验的五张业务表。
    return tables


# 只保留评分日前的左闭右开时间窗口，阻止未来信息泄漏。
def _window(frame: pd.DataFrame, column: str, cutoff: pd.Timestamp, days: int):
    # Scoring occurs at 00:00. Same-day observations are not yet available.
    # 返回左边界包含、评分日不包含的历史记录子集。
    return frame.loc[(frame[column] >= cutoff - pd.Timedelta(days=days)) & (frame[column] < cutoff)]


# 对一个评分日聚合在网客户的行为、账单和合约特征。
def snapshot(tables: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> pd.DataFrame:
    # 把输入转换为日期并归一到当日零点，统一时间边界。
    cutoff = pd.Timestamp(cutoff).normalize()
    # 按 TABLE_COLUMNS 的固定顺序解包五张业务表。
    customers, usage, invoices, tickets, events = (tables[n] for n in TABLE_COLUMNS)
    # terminated：评分日前已经终止服务的客户编号。
    terminated = events.loc[events["event_at"] < cutoff, "customer_id"]
    # 计算并保存 out；调用 customers.loc[(customers['joined_at'] < cutoff) & ~customers['customer_id'].isin(terminated)].set_index('customer_id').copy 完成本步处理。
    out = (
        customers.loc[
            (customers["joined_at"] < cutoff) & ~customers["customer_id"].isin(terminated)
        ]
        .set_index("customer_id")
        .copy()
    )
    # 检查 out.empty；满足条件才执行下面的分支。
    if out.empty:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError(f"No eligible customers at {cutoff.date()}")
    # 为本批所有在网客户写入相同的评分日期。
    out["snapshot_date"] = cutoff
    # 用评分日减入网日，得到在网天数。
    out["tenure_days"] = (cutoff - out["joined_at"]).dt.days
    # 用合约到期日减评分日得到剩余天数，已到期的负值截为 0。
    out["contract_days_remaining"] = (out["contract_end"] - cutoff).dt.days.clip(lower=0)
    # 逐个处理 [30, 90] 中的元素，当前值保存为 days。
    for days in [30, 90]:
        # 筛出当前 30 天或 90 天窗口内的使用记录。
        u = _window(usage, "date", cutoff, days)
        # 按客户汇总当前窗口流量；Pandas 按客户索引对齐回填。
        out[f"usage_mb_{days}d"] = u.groupby("customer_id")["data_mb"].sum()
        # 筛选当前窗口投诉，按客户统计工单条数。
        out[f"complaints_{days}d"] = (
            _window(tickets, "opened_at", cutoff, days).groupby("customer_id").size()
        )
        # 检查 days == 30；满足条件才执行下面的分支。
        if days == 30:
            # 按客户累计最近 30 天通话分钟数。
            out["call_minutes_30d"] = u.groupby("customer_id")["call_minutes"].sum()
            # active：当天流量或通话大于零的使用记录。
            active = u[(u["data_mb"] > 0) | (u["call_minutes"] > 0)]
            # 按客户统计有实际使用的不同日期数，而不是简单计行数。
            out["active_days_30d"] = active.groupby("customer_id")["date"].nunique()
    # known_invoices：评分日前已出账、当时能够知道的账单。
    known_invoices = invoices.loc[invoices["issued_at"] < cutoff]
    # unpaid：评分时已到期且当时仍未支付的账单。
    unpaid = known_invoices.loc[
        (known_invoices["due_at"] < cutoff)
        & (known_invoices["paid_at"].isna() | (known_invoices["paid_at"] >= cutoff))
    ]
    # 按客户统计评分时仍未支付的逾期账单数量。
    out["overdue_invoices"] = unpaid.groupby("customer_id").size()
    # 累计评分日前 90 天已出账金额，排除评分日及以后账单。
    out["bill_amount_90d"] = (
        _window(known_invoices, "issued_at", cutoff, 90).groupby("customer_id")["amount"].sum()
    )
    # aggregates：无业务记录时应补零的聚合特征列。
    aggregates = [
        "usage_mb_30d",
        "usage_mb_90d",
        "call_minutes_30d",
        "active_days_30d",
        "complaints_30d",
        "complaints_90d",
        "overdue_invoices",
        "bill_amount_90d",
    ]
    # 把没有匹配业务记录产生的空聚合值补为 0，原始缺失分区仍须上游保证完整。
    out[aggregates] = out[aggregates].fillna(0)
    # 用近 30 天流量除以近 90 天月均流量；先替换零分母，再将无流量结果补零。
    out["usage_trend"] = (3 * out["usage_mb_30d"] / out["usage_mb_90d"].replace(0, np.nan)).fillna(
        0
    )
    # 去掉原始日期字段、恢复客户编号列，并对最终特征执行统一校验。
    return validate_features(out.drop(columns=["joined_at", "contract_end"]).reset_index())


# 按月构建特征和未来 30 天标签；分片数据逐片处理后合并。
def build_dataset(
    raw_dir: Path, start: str, end: str, data_end: str | None, labeled: bool = True
) -> pd.DataFrame:
    # 同一客户的五张表放在同一分片：每次仅载入一万客户的明细，限制峰值内存。
    # partitions：按名称排序的客户数据分片目录。
    partitions = sorted(raw_dir.glob("part-*"))
    # 检查 partitions；满足条件才执行下面的分支。
    if partitions:
        # 计算并保存 frames；按右侧表达式完成取列、计算或结果构造。
        frames = [build_dataset(part, start, end, data_end, labeled) for part in partitions]
        # 合并后重新校验主键，防止同一客户快照被多个分片重复提供。
        # 按行拼接各月份或分片结果，再验证特征与快照主键。
        return validate_features(pd.concat(frames, ignore_index=True), labeled=labeled)
    # tables：按表名索引的业务数据。
    tables = load_tables(raw_dir)
    # cutoffs：请求日期范围内的月初评分时点。
    cutoffs = pd.date_range(start, end, freq="MS")
    # 检查 len(cutoffs) == 0；满足条件才执行下面的分支。
    if len(cutoffs) == 0:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("No month-start snapshot dates in requested range")
    # 检查 labeled and data_end is None；满足条件才执行下面的分支。
    if labeled and data_end is None:
        # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
        raise ValueError("data_end is required to prove label maturity")
    # 计算并保存 frames；按右侧表达式完成取列、计算或结果构造。
    frames = []
    # 取出终止事件表，仅用于在网状态判断与未来标签构建。
    events = tables["events"]
    # 逐个处理 cutoffs 中的元素，当前值保存为 cutoff。
    for cutoff in cutoffs:
        # label_end：未来 30 天标签观察窗口的右边界。
        label_end = cutoff + pd.Timedelta(days=HORIZON_DAYS)
        # 检查 labeled and label_end > pd.Timestamp(data_end)；满足条件才执行下面的分支。
        if labeled and label_end > pd.Timestamp(data_end):
            # 发现不符合契约或无法继续的情况，抛出带具体原因的异常。
            raise ValueError(f"Immature label window: {cutoff.date()} ends at {label_end.date()}")
        # 计算并保存 frame；调用 snapshot 完成本步处理。
        frame = snapshot(tables, cutoff)
        # 检查 labeled；满足条件才执行下面的分支。
        if labeled:
            # churned：标签观察窗口内发生主动流失的客户编号。
            churned = events.loc[
                (events["event_at"] >= cutoff)
                & (events["event_at"] < label_end)
                & events["event_type"].isin(CHURN_TYPES),
                "customer_id",
            ]
            # 客户出现在未来观察窗的主动流失集合时标记 1，否则标记 0。
            frame["churn"] = frame["customer_id"].isin(churned).astype(int)
            # 记录该行标签何时完全成熟，供时间分区防泄漏使用。
            frame["label_end_date"] = label_end
        # 把当前结果追加到列表，供稍后统一汇总。
        frames.append(frame)
    # 按行拼接各月份或分片结果，再验证特征与快照主键。
    return validate_features(pd.concat(frames, ignore_index=True), labeled=labeled)
