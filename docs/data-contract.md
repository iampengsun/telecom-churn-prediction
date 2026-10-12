# 数据契约

## 时间语义

所有日期使用同一业务时区，格式为 `YYYY-MM-DD`。评分日 `t` 表示当日 00:00。输入为截至该时点已可用的业务数据；数据仓库应在导出前完成迟到数据、回补和历史版本处理。

- 行为窗口：`[t-30天, t)`、`[t-90天, t)`。
- 流失标签窗口：`[t, t+30天)`。
- `data_end`：源数据已完整入库的右开边界，要求 `t+30天 <= data_end`。
- 在网客户：`joined_at < t` 且评分日之前没有终止事件。
- 客户套餐、区域和合约属性应为评分时点可用的历史值。本地批量构建使用不可变客户维表；属性发生变化时，上游必须提供历史快照并逐评分日调用 `snapshot`，不能使用当前维表回填历史训练数据。
- 同一客户允许出现在不同时间分区，适用目标为已有客户的未来风险。客户编号和评分日期不进入模型。

## 原始表

### customers.csv

| 字段 | 类型 | 约束 |
|---|---|---|
| customer_id | string | 主键，字母、数字、下划线或短横线，最长 80 字符 |
| joined_at | date | 入网日期 |
| contract_end | date | 当前有效合约到期日期，无有效合约使用已过去的日期 |
| monthly_fee | float | 非负套餐月费，单位元 |
| plan_type | enum | basic / standard / premium |
| region | string | 业务区域，最长 40 字符 |

### usage.csv

| 字段 | 类型 | 约束 |
|---|---|---|
| customer_id | string | 客户外键 |
| date | date | 业务日期，与客户编号组成唯一键 |
| data_mb | float | 当日数据流量，非负，单位 MB |
| call_minutes | float | 当日通话时长，非负，单位分钟 |

无使用记录聚合为 0；数据完整性水位由上游保证，缺失分区不能作为零使用量导入。

### invoices.csv

| 字段 | 类型 | 约束 |
|---|---|---|
| invoice_id | string | 唯一账单编号 |
| customer_id | string | 客户外键 |
| issued_at | date | 账单出账日期 |
| due_at | date | 应付日期，不早于出账日期 |
| paid_at | nullable date | 实际付款日期；未付为空 |
| amount | float | 非负金额，单位元 |

逾期账单要求 `issued_at < t`、`due_at < t`，且 `paid_at` 为空或不早于 `t`。评分之后的付款不能改变历史逾期状态。

### tickets.csv

`ticket_id` 为唯一编号，`customer_id` 为客户外键，`opened_at` 为投诉工单创建日期。只导入投诉类工单。

### events.csv

`customer_id` 为客户外键，`event_at` 为业务生效日期，`event_type` 取值如下：

| 类型 | 作为流失正例 | 生效后移出在网客户 |
|---|---|---|
| voluntary_close | 是 | 是 |
| port_out | 是 | 是 |
| involuntary_close | 否 | 是 |

此表仅包含终止事件，不接收普通客服、营销或套餐变更事件。客户重新入网必须使用独立的订阅实体编号，避免同一编号跨多个生命周期。

## 特征表

`customer_id` 与 `snapshot_date` 组成唯一键。

| 特征 | 定义 |
|---|---|
| tenure_days | 评分日与入网日的天数差 |
| monthly_fee | 评分时点的套餐月费 |
| contract_days_remaining | 合约剩余天数，已到期为 0 |
| usage_mb_30d / usage_mb_90d | 对应窗口累计流量 |
| usage_trend | `3 × usage_mb_30d / usage_mb_90d`，90 天流量为 0 时取 0 |
| call_minutes_30d | 近 30 天累计通话分钟数 |
| active_days_30d | 近 30 天流量或通话大于 0 的不同日期数 |
| complaints_30d / complaints_90d | 对应窗口投诉工单数 |
| overdue_invoices | 评分时点已出账但逾期未付的账单数 |
| bill_amount_90d | 近 90 天出账金额 |
| plan_type / region | 套餐及区域属性 |

标签列为 `churn`（0 / 1）和 `label_end_date`（`snapshot_date + 30天`）。模型输入不包含标签、客户编号或日期。特征不存在时拒绝输入，不对缺失业务数据做隐式推断。未知区域允许进入模型，其类别编码为全零并由监控记录。

## 联系资格表

`eligibility.csv`：

| 字段 | 定义 |
|---|---|
| customer_id | 唯一客户编号 |
| contact_allowed | 1 表示允许触达，0 表示禁止 |
| last_contact_at | 最近一次实际触达日期，无历史触达为空 |

名单按客户内连接；缺少资格记录的客户不导出。仅导出高风险、可触达且冷却期已满的客户，再按风险降序和客户编号排序，最多导出 `capacity` 条。名单不执行短信、电话或CRM写入。
