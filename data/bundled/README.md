# 电信客户业务数据

本数据集随项目固定交付，不需要运行生成脚本。客户编号为项目内部标识，业务表覆盖 2024-10-01 至 2026-01-31；完整性水位右开边界为 2026-02-01。

| 表 | 行数 |
|---|---:|
| customers | 200,000 |
| usage | 79,588,503 |
| invoices | 2,655,420 |
| tickets | 469,781 |
| events | 90,402 |

五张表合计 83,004,106 行，压缩文件合计 751.6 MiB。按客户划分为 20 个 `part-000` 至 `part-019` 目录，每片 10,000 名客户，五张表采用相同客户分区。单个客户的完整历史保留在同一片内。每个目录包含 `customers.csv.gz`、`usage.csv.gz`、`invoices.csv.gz`、`tickets.csv.gz`、`events.csv.gz`。

## 使用方式

将这份数据放在项目的 `data/bundled/` 下，先查看表头、时间范围和清单，目录检查方法见[实操教程](../../docs/tutorial.md#2-准备输入数据和目录)。各张表的样例和字段含义见[数据说明](../../docs/data-contract.md#原始表)，不必等业务模块完成。

等 `features.py`、`cli.py` 写好并注册 `churn` 命令后，才在项目根目录执行：

```bash
uv run --frozen churn build-features --data-end 2026-02-01
```

命令固定读取项目的 `data/bundled/`，逐分片校验和聚合，再合并月度特征。不要将同一客户放到多个分片。普通的未分片 CSV 目录仍可读取。

各目录用途、CSV / gzip / JSON 格式、原始表与生成文件的区别，以及详细字段，见[数据说明](../../docs/data-contract.md#数据目录与文件格式)；操作顺序见 [VS Code 实操教程](../../docs/tutorial.md)。小规模练习和测试使用独立的 `tests/data/`，不需要扫描全量明细。

## 来源与使用范围

数据已经脱敏。来源标识保留在 `metadata.json`，并随特征和模型版本传递。

根目录 `metadata.json` 记录每个压缩文件的字节数和 SHA-256，可检查传输是否完整。此数据用于单机百万级明细处理与机器学习。
