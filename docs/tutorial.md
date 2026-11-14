# VS Code 从零实操：电信客户流失预测

本教程按环境准备、模块开发、数据处理、模型训练和服务运行的顺序组织。每个步骤说明文件位置、函数编写顺序、模块职责和验收方式。逐语句说明见源文件注释。

首次使用 uv、pytest、Ruff、Make 或 GitHub Actions 时，先阅读 [开发工具入门](tooling.md)。

## 1. 准备项目与开发工具

安装 Python 3.14 和 VS Code。在 VS Code 扩展市场安装 Microsoft 的 **Python** 与 **Pylance** 扩展。需要断点调试时也安装 Microsoft 的 **Python Debugger** 扩展。可选安装 Ruff，以便编辑时发现格式和静态检查问题。

使用「文件 → 打开文件夹」打开 `telecom-churn-prediction` 的根目录，确认资源管理器能看到 `pyproject.toml`、`uv.lock`、`src`、`tests` 和 `data/bundled`。不要只打开 `src`，后续所有命令都在项目根目录运行。

通过「终端 → 新建终端」打开集成终端，检查 Python 并安装环境管理工具：

```bash
python --version
python -m pip install uv==0.9.18
uv --version
uv python install 3.14
uv sync --frozen --extra lightgbm
```

如果系统命令叫 `python3`，前两行改用 `python3`；Windows 也可以用 `py -3.14`。已有 uv 时跳过安装。`uv sync` 创建 `.venv`，按锁文件安装项目、LightGBM 和开发依赖；不需要单独手动安装 numpy/pandas/sklearn。`--frozen` 表示不修改锁定结果。

在命令面板选择 **Python: Select Interpreter**，选择项目 `.venv` 中的解释器。若没有出现，选择手动输入路径：类 Unix 系统为 `.venv/bin/python`，Windows 为 `.venv\Scripts\python.exe`。

运行下面的检查，确认执行的是项目解释器，并且 LightGBM 可以导入：

```bash
uv run --frozen --extra lightgbm python -c "import sys, lightgbm; print(sys.executable); print(lightgbm.__version__)"
```

后文统一使用 `uv run`，无需手动激活环境。不要用 VS Code 的「运行 Python 文件」去直接运行带包内导入的 `cli.py`，使用后面的 `churn` 命令或模块调试。

## 2. 理解目录与输入数据

| 目录或文件 | 用途 | 需要手写吗 |
|---|---|---|
| `pyproject.toml` | 包名、依赖、CLI 入口、测试和静态检查配置 | 阅读并理解，依赖已配置 |
| `uv.lock` | 精确锁定依赖版本 | 不手写 |
| `src/telecom_churn/` | 业务代码 | 按下面顺序编写或阅读 |
| `tests/` | 业务边界和接口验证 | 跟随对应模块编写 |
| `tests/data/` | 160 名客户的固定测试子集 | 不修改，不生成 |
| `data/bundled/` | 随项目提供的 200,000 名客户原始数据 | 直接读取，不生成 |
| `data/features.csv` | 全年带标签特征 | 命令自动产出 |
| `data/latest.csv` | 最新月份无标签特征 | 命令自动产出 |
| `artifacts/` | 训练版本、模型、指标和测试预测 | 命令自动产出 |
| `reports/` | 批量评分、漂移报告、挽留名单 | 命令自动产出 |

原始数据按客户编号分成 20 个目录，每个目录有 10,000 名客户的完整五张表。单个客户的所有记录放在同一分片，不会跨片计算窗口。CSV 使用 gzip 压缩，程序直接读取，不需要解压，也不要用 VS Code 打开数千万行明细。

先打开 `data/bundled/README.md` 和 `metadata.json` 查看实际行数、时间范围和来源；再打开 `docs/data-contract.md` 理解五张表的字段。数据是固定规则合成的教学数据，其离线指标不能代表运营商真实上线效果。来源记录保存在数据及模型元数据中。

全量运行是单机 Pandas 工程练习，建议使用至少 16 GB 内存、预留 15 GB 磁盘空间；具体峰值取决于运行时和表格复制。特征构建逐分片读取明细，训练仍会把聚合后的特征表放入内存。全量训练使用单机内存，不提供分布式调度。

## 3. 第一个模块：`schema.py`

先创建 `src/telecom_churn/__init__.py`，再编写 `schema.py`。顺序如下：

1. 定义 `SCHEMA_VERSION`、`HORIZON_DAYS`、数值和类别特征列表，以及主动流失事件集合。
2. 编写 `CustomerSnapshot`，声明客户编号、评分日期和各特征的类型及范围。
3. 编写 `check_windows`，校验 30 天量不超过 90 天量、趋势字段与累计量一致。
4. 编写 `PredictRequest` 和 `Prediction`，规定请求与响应结构。
5. 编写 `validate_features`，处理缺列、空表、重复快照、非法数值和标签窗口。

作用：所有训练、评分和 API 都引用同一份契约，避免各自理解字段。特征中不放客户编号、评分日和未来标签；这些列只用于关联或评估。

完成后检查语法：

```bash
uv run --frozen --extra lightgbm python -m compileall -q src/telecom_churn/schema.py
```

## 4. 第二个模块：`features.py`

按 `TABLE_COLUMNS` → `DATE_COLUMNS` → `load_tables` → `_window` → `snapshot` → `build_dataset` 编写。

- `load_tables`：读取普通或压缩 CSV，检查主外键、日期和非负数值。
- `_window`：选取 `[评分日-窗口天数, 评分日)`，评分当天的数据不能进入特征。
- `snapshot`：排除已流失客户，按客户聚合 30/90 天流量、投诉、活跃天数及账单信息。
- `build_dataset`：逐月生成快照，用未来 30 天终止事件产生标签；有分片时逐片构建再合并。

特别注意：2025-06-01 的特征只能使用当日零点前已知的信息。不能因为某客户后来还清账单，就把过去评分时点的逾期状态改成已支付。

这一步再编写 `tests/conftest.py` 中的固定数据读取 fixture，以及 `tests/test_features.py`。在尚未写完 modeling 模块时，conftest 顶部的训练导入尚不可用，先做语法检查；完成第 7 步后统一运行测试。

## 5. 第三个模块：`metrics.py`

编写 `classification_metrics`。先校验标签与分数，再计算阈值分类结果、ROC-AUC、Average Precision、Brier Score 和前 K 位客户的 Precision / Recall / Lift。

作用：流失通常属于少数类，只看准确率会掩盖漏检。Top-K 指标回答「只能联系风险最高的 10% 客户时，能找到多少实际流失客户」。同分时使用稳定排序，保证相同输入得到相同名单顺序。

## 6. 第四个模块：`monitoring.py`

按 `_distribution` → `make_reference` → `drift_report` 编写。

作用：保存训练集的分布作为基准，在新月份对数值特征计算 PSI，对类别特征比较分布与未知类别比例。分布变化需要排查业务原因，不等同于模型已经失效，也不会自动触发重训。

训练模块需要调用 `make_reference`，所以监控模块先于训练模块编写。

## 7. 第五个模块：`modeling.py`

按下面顺序完成：

1. `temporal_split`：按日期分训练、验证和测试集，并清除跨越分区的标签窗口。
2. `make_pipeline`：组合标准化、One-Hot 编码和 Logistic Regression / LightGBM 分类器。
3. `write_json`：统一写 JSON。
4. `train`：拟合候选模型，在验证集按 Average Precision 选型和确定阈值，评估测试集，保存工件。
5. `resolve_model`：解析模型路径。
6. `load_model`：检查 SHA-256 和特征版本，再加载完整管道。

默认训练期从 2025-01 起，验证期从 2025-08 起，测试期从 2025-10 起。标签观察窗必须在下一分区开始前结束。预处理只在训练分区 `fit`，验证和测试只做转换、预测。

候选指定 LightGBM 不代表它必然获胜；若 Logistic Regression 验证分数更高，保存的就是基线模型。不要反复查看测试集后修改超参数，否则测试集也被用于选型。

此时补完 `conftest.py` 的特征、模型和请求 fixture，编写 `test_modeling.py`，运行：

```bash
uv run --frozen --extra lightgbm pytest tests/test_features.py tests/test_modeling.py
```

验收：测试通过；未来观测不会改变过去特征；破坏模型校验和后会拒绝加载。

## 8. 第六个模块：`scoring.py`

按 `risk_signals` → `score_customers` → `retention_list` 编写，然后编写 `tests/test_retention.py`。

`score_customers` 调用模型管道的 `predict_proba`，取流失类别分数，并按保存的阈值分层。规则信号帮助解释回访原因，不是 SHAP 值。`retention_list` 只导出高风险、允许联系、冷却期已满的客户，最后施加人数上限。

```bash
uv run --frozen --extra lightgbm pytest tests/test_retention.py
```

## 9. 第七个模块：`cli.py`

先定义 Typer 应用，再编写 `read_features`、`save_csv`，随后依次实现 `build_features`、`train_model`、`score`、`evaluate`、`monitor`、`export_retention`、`demo` 和 `serve`。

作用：让同一套 Python 函数既能被其他代码调用，也能从终端按步骤执行。`demo` 只是把已有步骤串起来，不生成任何原始数据。`pyproject.toml` 中 `churn = "telecom_churn.cli:app"` 将命令映射到 Typer 应用。

```bash
uv run --frozen --extra lightgbm churn --help
uv run --frozen --extra lightgbm pytest tests/test_cli.py
```

验收：帮助列出特征、训练、评分、评估、监控、名单、演示和服务命令，没有 `generate-data`。

## 10. 第八个模块：`api.py` 与前端

在 `api.py` 先写 `create_app`，再在其中依次注册生命周期 `lifespan`、鉴权和模型依赖函数、请求日志中间件、健康检查、模型信息、预测及首页路由，最后挂载静态目录。

作用：启动时载入模型一次；每个请求复用模型。数据结构不合法返回 422，模型未就绪返回 503，需要密钥但验证失败返回 401。

前端按 `static/index.html` → `static/style.css` → `static/app.js` 编写。HTML 定义页面区域与元素 ID；CSS 决定布局；JS 按这些 ID 获取元素，调用模型和预测接口，把结果填到表格，并提供 CSV 下载。此页面无需 Node.js 或前端构建工具。

编写 `tests/test_api.py`，随后运行全部检查：

```bash
uv run --frozen --extra lightgbm ruff check src tests
uv run --frozen --extra lightgbm ruff format --check src tests
uv run --frozen --extra lightgbm pytest
```

## 11. 先用固定小样本演练

第一次建议只跑固定的 160 客户子集，先检查环境和命令。输出放到独立目录，避免覆盖全量模型：

```bash
uv run --frozen --extra lightgbm churn build-features --raw-dir tests/data --data-end 2026-02-01 --output data/practice/features.csv
uv run --frozen --extra lightgbm churn train --features data/practice/features.csv --artifacts artifacts/practice
```

验收：出现特征行数、模型版本目录和 `artifacts/practice/latest.json`。原始压缩数据不应被修改。

## 12. 构建全量训练特征

```bash
uv run --frozen --extra lightgbm churn build-features --raw-dir data/bundled --start 2025-01-01 --end 2025-12-01 --data-end 2026-02-01 --output data/features.csv
```

作用：读取 20 个客户分片，构建 12 个月的历史快照。结果是客户×月份级记录，不是每天明细；已流失客户会退出后续月份，行数会少于 200,000×12。

验收：终端打印实际行数，生成 `data/features.csv` 和 `data/features.metadata.json`。水位 `2026-02-01` 是右开边界，表示此日期前的数据已完整观察。不要在真实数据中凭空填写这个日期。

## 13. 训练与查看结果

```bash
uv run --frozen --extra lightgbm churn train --features data/features.csv --validation-start 2025-08-01 --test-start 2025-10-01 --algorithm lightgbm --artifacts artifacts
```

训练会比较基线与 LightGBM，并写入：

```text
artifacts/latest.json
artifacts/<版本号>/model.joblib
artifacts/<版本号>/metadata.json
artifacts/<版本号>/test_predictions.csv
```

打开 `latest.json` 获取当前版本，再打开对应 `metadata.json`，依次检查 `partitions`、`purged_rows`、`validation_comparison`、`selected_algorithm`、`test_metrics` 和 `permutation_importance`。来源元数据保存在 `data_source`。

全量训练和置换重要性计算可能耗时较长。终端未报错且进程仍在运行时不要重复启动多个训练任务。重复训练会产生新版本，不覆盖旧版本。

## 14. 构建最新特征并批量评分

```bash
uv run --frozen --extra lightgbm churn build-features --raw-dir data/bundled --start 2026-01-01 --end 2026-01-01 --no-labeled --output data/latest.csv
uv run --frozen --extra lightgbm churn score --features data/latest.csv --output reports/scores.csv
uv run --frozen --extra lightgbm churn monitor --features data/latest.csv --output reports/drift.json
```

作用：2026-01-01 作为新评分批次，不需要未来标签。批量评分应输出客户、日期、分数、等级、信号、建议和模型版本；漂移报告记录当前样本量及各特征状态。

## 15. 准备 API 请求与联系资格

下面是输入文件整理，不是原始业务数据生成。新建 `data/prepare_inputs.py`，写入以下代码，并执行一次。`data/` 下派生文件不进入版本控制：

```python
import json  # 将少量客户特征转换为 API 的 JSON 请求。
from pathlib import Path  # 指定输出位置。
import pandas as pd  # 读取已构建的特征表。

latest = pd.read_csv("data/latest.csv")  # 最新特征不包含未来标签。
request = {"customers": latest.head(3).to_dict("records")}  # API 单次最多 1,000 人，演练仅取 3 人。
Path("data/request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存请求。
eligibility = latest[["customer_id"]].copy()  # 只保留客户关联键。
eligibility["contact_allowed"] = 1  # 仅用于流程演练；业务必须使用实际授权记录。
eligibility["last_contact_at"] = None  # 演练假定无历史触达；业务需填实际触达日期。
eligibility.to_csv("data/eligibility.csv", index=False)  # 保存名单筛选的资格输入。
```

```bash
uv run --frozen --extra lightgbm python data/prepare_inputs.py
uv run --frozen --extra lightgbm churn retention-list --scores reports/scores.csv --eligibility data/eligibility.csv --capacity 100 --cooldown-days 14 --output reports/retention.csv
```

验收：名单不超过 100 人，只有高风险客户；`priority` 从 1 开始。该操作只导出文件，不会发短信或拨打电话。

## 16. 启动服务并在 VS Code 调试

```bash
uv run --frozen --extra lightgbm churn serve
```

浏览器依次访问：

- `http://127.0.0.1:8000/health`：进程存活。
- `http://127.0.0.1:8000/ready`：模型就绪及版本号。
- `http://127.0.0.1:8000`：客户风险工作台，上传 `data/request.json` 并预测。
- `http://127.0.0.1:8000/docs`：展开 POST 预测接口，点击 Try it out，粘贴请求 JSON 后执行。

验收：工作台显示模型和评估结果，3 名客户返回评分。JSON 顶层必须有 `customers`，不能把整个 CSV 直接粘贴到预测输入框。按终端 `Ctrl+C` 停止服务。

需要断点调试时，先停掉旧服务，调试配置位于 `.vscode/launch.json`，在「运行和调试」选择「客户流失 API」；配置使用 `module: "telecom_churn.cli"`、`args: ["serve"]` 和 `cwd: "${workspaceFolder}"`。在 `api.py` 的 `predict` 或 `scoring.py` 的 `score_customers` 内设置断点，按 F5 启动，然后从页面提交请求。检查 `frame`、`scores` 和 `thresholds`。

本地默认不配置密钥；部署时配置 `CHURN_API_KEY` 和 `CHURN_ENV=production`，细节见 `operations.md`。

## 17. 第二次运行与交付检查

完整流水线入口：

```bash
uv run --frozen --extra lightgbm churn demo
```

它从固定数据重新构建全年和最新特征、训练、评分、监控、准备接口请求与演练资格并输出名单。它会覆盖同名派生 CSV/报告并新增模型版本，所以不应在需要保留的业务输出目录直接运行。

最后执行：

```bash
uv run --frozen --extra lightgbm pytest
uv run --frozen --extra lightgbm ruff check src tests
uv run --frozen --extra lightgbm ruff format --check src tests
uv build
```

交付时包含源码、锁文件、文档、固定测试数据和完整 `data/bundled`。`data/bundled/metadata.json` 提供每个压缩文件的 SHA-256 和大小。构建的 Python wheel 只包含代码和静态页面，原始数据通过项目目录一起交付，不会被塞进 wheel。

## 常见问题

| 现象 | 检查与处理 |
|---|---|
| 找不到 `churn` 或导入包失败 | 确认在根目录运行过 `uv sync`，使用完整的 `uv run ... churn` 命令 |
| VS Code 提示导入无法解析 | 重新选择项目 `.venv` 解释器 |
| 构建标签提示需要 `data_end` | 为带标签构建传入完整性水位；评分使用 `--no-labeled` |
| 模型没有选 LightGBM | 查看验证集 AP，基线获胜是正常行为 |
| 页面就绪检查返回 503 | 先训练模型，检查 `artifacts/latest.json` 和服务启动日志 |
| 预测返回 422 | 按数据契约检查字段、条数、数值范围和窗口关系 |
| 内存不足 | 先跑 `tests/data`；关闭重复进程，或在内存更大的机器执行全量训练 |
| 模型更新后页面仍显示旧版本 | 重启服务，进程仅在启动时加载模型 |
