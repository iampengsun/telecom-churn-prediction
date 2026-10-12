# Telecom Customer Churn Prediction

电信客户流失预警与挽留决策支持系统。

系统以客户历史通信、账单、合约和服务工单为输入，预测未来 30 天主动销户或携号转出风险，提供客户风险分层、批量评分、挽留名单、模型评估和数据漂移监控。

## 功能

- 五类业务表接入、主键和数值校验、客户级时间窗口聚合。
- 30 / 90 天行为特征、合约特征、账单逾期特征。
- 按时间划分训练、验证与测试集，清除跨分区标签窗口。
- Logistic Regression 基线与 LightGBM 对比，支持 HistGradientBoosting。
- 验证集选择模型及风险分层阈值，测试集独立评估。
- ROC-AUC、Average Precision、Brier Score、Top-K Precision / Recall / Lift。
- 模型版本、数据指纹、依赖版本、校验和及原子版本指针。
- 批量评分、FastAPI 接口、客户风险工作台。
- 联系授权、触达冷却期和容量约束下的挽留名单。
- 数值特征 PSI、类别分布变化和未知类别比例监控。

## 工程结构

```text
src/telecom_churn/
├── schema.py          输入契约与特征版本
├── features.py        业务表读取与时间窗口计算
├── modeling.py        时间验证、训练、模型持久化
├── metrics.py         模型与名单排序指标
├── scoring.py         客户评分、风险信号、名单筛选
├── monitoring.py      训练参考分布与漂移检测
├── sample_data.py     集成测试数据生成器
├── cli.py             数据、训练、评分和监控命令
├── api.py             预测服务与访问控制
└── static/            客户风险工作台
 tests/                数据口径、模型、接口、名单与命令行测试
 docs/                 数据契约、架构和部署说明
```

## 环境

Python 3.11–3.14，依赖由 `uv.lock` 锁定。推荐 Python 3.12。

```bash
uv sync --frozen --extra lightgbm
```

macOS 的 LightGBM 依赖 OpenMP：

```bash
brew install libomp
```

不使用 LightGBM 的环境可执行 `uv sync --frozen`，训练参数指定 `--algorithm hist_gradient_boosting` 或 `--algorithm logistic`。

## 全流程

```bash
churn demo --customers 800 --seed 42
churn serve
```

命令使用项目虚拟环境；也可通过 `uv run --frozen --extra lightgbm churn ...` 执行。工作台地址为 `http://127.0.0.1:8000`，OpenAPI 页面为 `/docs`。

全流程生成 `data/raw/`、`data/features.csv`、`data/latest.csv`、`data/request.json`、版本化模型、测试集预测明细、批量评分、漂移报告与挽留名单。`demo` 和 `generate-data` 是集成验证入口；业务数据由独立的数据接入流程提供，来源信息随特征和模型版本保存。

## 业务数据流水线

```bash
churn build-features \
  --raw-dir data/raw \
  --start 2025-01-01 --end 2025-12-01 \
  --data-end 2026-02-01 \
  --output data/features.csv

churn train \
  --features data/features.csv \
  --validation-start 2025-08-01 \
  --test-start 2025-10-01 \
  --algorithm lightgbm \
  --artifacts artifacts

churn build-features \
  --raw-dir data/raw \
  --start 2026-01-01 --end 2026-01-01 \
  --no-labeled --output data/current.csv

churn score --features data/current.csv --output reports/scores.csv
churn monitor --features data/current.csv --output reports/drift.json

churn retention-list \
  --scores reports/scores.csv \
  --eligibility data/eligibility.csv \
  --capacity 100 --cooldown-days 14 \
  --output reports/retention.csv
```

`--data-end` 为数据完整性水位的右开边界。只有完整观察到评分日之后 30 天的样本才可构建标签。特征构建按月初生成样本；线上任意日期可调用 `features.snapshot`。

## 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 进程存活检查 |
| GET | `/ready` | 模型可用性检查 |
| GET | `/api/v1/model` | 模型版本与评估结果 |
| POST | `/api/v1/predict` | 1–1,000 条客户特征评分 |

```bash
curl http://127.0.0.1:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${CHURN_API_KEY}" \
  --data-binary @data/request.json
```

API 和批量评分共用校验与模型管道。额外字段、重复客户快照、非法数值、错误窗口关系返回 422；模型未加载返回 503。

## 工程检查

```bash
make lint
make test
make build
```

CI 在 Python 3.11、3.12、3.14 上执行静态检查、测试和构建。

## 相关文档

- [数据契约](docs/data-contract.md)
- [模型与系统设计](docs/architecture.md)
- [部署与运维](docs/operations.md)

技术参考：[scikit-learn 数据泄漏与 Pipeline](https://scikit-learn.org/stable/common_pitfalls.html)、[LightGBM 安装说明](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)、[FastAPI Lifespan](https://fastapi.tiangolo.com/advanced/events/)。
