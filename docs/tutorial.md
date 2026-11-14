# 搭建电信客户流失预测项目

## 1. 准备项目与开发工具

```bash
mkdir telecom-churn-prediction
cd telecom-churn-prediction
mkdir -p src/telecom_churn tests docs data
```

用 VS Code 打开这个目录，此时没有业务代码和训练模型，这是正常的。新建空文件 `src/telecom_churn/__init__.py`，让该目录成为 Python 包；新建 `README.md`，先写一行项目名称；新建 `.python-version`，内容为 `3.14`。

新建 `pyproject.toml`，写入：

```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "telecom-churn-prediction"
version = "0.1.0"
description = "Point-in-time telecom churn prediction and retention decision support"
readme = "README.md"
requires-python = ">=3.14,<3.15"
dependencies = [
  "numpy>=2.1,<2.4", "pandas>=2.2,<3", "scikit-learn>=1.6,<1.9",
  "joblib>=1.4,<2", "fastapi>=0.115,<0.130", "uvicorn>=0.34,<1",
  "pydantic>=2.10,<3", "typer>=0.15,<1", "anyio>=4.8,<4.12",
  "lightgbm>=4.5,<5"
]

[dependency-groups]
dev = ["pytest>=8.3,<10", "pytest-cov>=6,<8", "httpx>=0.28,<1", "ruff>=0.11,<1"]

[tool.hatch.build.targets.wheel]
packages = ["src/telecom_churn"]

[tool.hatch.build.targets.sdist]
exclude = ["/data", "/artifacts", "/reports"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py314"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
ignore = ["E501", "B008"]

```

`[project]` 声明项目及运行依赖，`[dependency-groups]` 声明开发工具，`[build-system]` 选择负责打包的后端。`packages` 指定要安装的源码目录，`exclude` 将数据和运行结果排除在源码分发包外。测试和 Ruff 配置先写好，等对应文件创建后再使用。

```bash
uv sync
uv run python -c "import sys; print(sys.executable)"
```

首次 `uv sync` 生成 `uv.lock` 和 `.venv`；后面出现的 `--frozen` 都以锁文件已经生成为前提。修改依赖或命令入口后，先用普通 `uv sync` 更新环境和锁文件。

## 2. 准备输入数据和目录

| 阶段 | 创建或准备的内容 | 来源 |
|---|---|---|
| 项目初始化 | `pyproject.toml`、`.python-version`、`README.md`、空包目录 | 自己创建 |
| 首次安装 | `.venv/`、`uv.lock` | uv 生成 |
| 输入数据 | `data/bundled/`、`tests/data/` | 原始业务表和独立的测试样本 |
| 模块开发 | `schema.py` 等业务模块、测试文件 | 按第 3–10 节逐个编写 |
| 特征构建 | `data/features.csv`、配套元数据 | 第 11–12 节执行命令生成 |
| 模型训练 | `artifacts/` 下的版本目录和指针 | 第 11、13 节生成 |
| 评分与名单 | `data/latest.csv`、请求、资格表和 `reports/` | 第 14–15 节生成 |
| 工程配置 | Makefile、GitHub 工作流、Dockerfile、Compose 配置 | 最后添加并验证 |

把提供的小样本放到 `tests/data/`，全量数据放到 `data/bundled/`。已有这些目录时直接检查文件，不必再复制。小样本目录应直接包含五张 `.csv.gz` 表和 `metadata.json`；全量目录应直接包含 `part-000/` 至 `part-019/`、说明文件和元数据，避免多套一层同名目录。

`tests/data/` 是 160 客户的小样本，先用它开发和测试。`data/bundled/` 是全量数据，按客户分成 20 片，每片包含五张压缩业务表，同一客户的完整记录留在同一片。CSV 使用 gzip 压缩，读取时不必先解压。

编写读取代码前，先按[数据说明](data-contract.md#原始表)检查表头、类型和时间字段，再查看数据清单中的范围和行数。

全量特征构建逐片加载，但训练仍会把聚合特征放入内存。先跑小样本，再根据机器内存和剩余磁盘决定是否做全量计算。

## 3. 第一个模块：`schema.py`

在已创建的包目录中新建 `schema.py`，按以下顺序实现：

1. 定义 `SCHEMA_VERSION`、`HORIZON_DAYS`、数值和类别特征列表，以及主动流失事件集合。
2. 编写 `CustomerSnapshot`，声明客户编号、评分日期和各特征的类型及范围。
3. 编写 `check_windows`，校验 30 天量不超过 90 天量、趋势字段与累计量一致。
4. 编写 `PredictRequest` 和 `Prediction`，规定请求与响应结构。
5. 编写 `validate_features`，处理缺列、空表、重复快照、非法数值和标签窗口。

所有训练、评分和 API 都引用同一份契约，避免各自理解字段。特征中不放客户编号、评分日和未来标签；这些列只用于关联或评估。

## 4. 第二个模块：`features.py`

按 `TABLE_COLUMNS` -> `DATE_COLUMNS` -> `load_tables` -> `_window` -> `snapshot` -> `build_dataset` 编写。

- `load_tables`：读取普通或压缩 CSV，检查主外键、日期和非负数值。
- `_window`：选取 `[评分日-窗口天数, 评分日)`，评分当天的数据不能进入特征。
- `snapshot`：排除已流失客户，按客户聚合 30/90 天流量、投诉、活跃天数及账单信息。
- `build_dataset`：逐月生成快照，用未来 30 天终止事件产生标签；有分片时逐片构建再合并。

2025-06-01 的特征只能使用当日零点前已知的信息。不能因为某客户后来还清账单，就把过去评分时点的逾期状态改成已支付。

现在新建 `tests/conftest.py`，只准备本阶段用到的数据目录，不要提前导入尚未编写的训练模块：

```python
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    target = tmp_path_factory.mktemp("raw")
    shutil.copytree(Path(__file__).parent / "data", target, dirs_exist_ok=True)
    return target
```

fixture 用来复用测试准备工作，临时目录用来隔离测试输入。编写 `tests/test_features.py` 中的未来观测隔离和窗口边界测试，准备记录、调用 `snapshot`，再断言聚合结果。先运行这两项：

```bash
uv run --frozen pytest tests/test_features.py -k "future_observations or window_boundaries"
```

这时不需要训练模型。如果后续测试需要 `features` 或 `latest` fixture，等相应准备函数添加后再运行。

## 5. 第三个模块：`metrics.py`

编写 `classification_metrics`。先校验标签与分数，再计算阈值分类结果、ROC-AUC、Average Precision、Brier Score 和前 K 位客户的 Precision / Recall / Lift。

流失通常属于少数类，只看准确率会掩盖漏检。Top-K 指标回答「只能联系风险最高的 10% 客户时，能找到多少实际流失客户」。同分时使用稳定排序，保证相同输入得到相同名单顺序。

## 6. 第四个模块：`monitoring.py`

按 `_distribution` → `make_reference` → `drift_report` 编写。

保存训练集的分布作为基准，在新月份对数值特征计算 PSI，对类别特征比较分布与未知类别比例。分布变化需要排查业务原因，不等同于模型已经失效，也不会自动触发重训。

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

最终模型由验证集表现决定；若基线分数更高，保存的就是基线模型。不要反复查看测试集后修改超参数，否则测试集也被用于选型。

此时在 `conftest.py` 中增加 `features(raw_dir)`、`model_dir(tmp_path_factory, features)`、`bundle(model_dir)` 和 `latest(features)`，分别准备特征、临时训练结果、已加载模型和最新快照。现在才能导入训练模块。请求用的 `payload` fixture 留到接口阶段添加。

补齐特征测试，再编写 `test_modeling.py`，运行：

```bash
uv run --frozen pytest tests/test_features.py tests/test_modeling.py
```

测试通过；未来观测不会改变过去特征；破坏模型校验和后会拒绝加载。

## 8. 第六个模块：`scoring.py`

先写 `risk_signals`，再写 `score_customers` 和 `retention_list`，然后编写 `tests/test_retention.py`。

`score_customers` 调用模型管道的 `predict_proba`，取流失类别分数，并按保存的阈值分层。规则信号帮助解释回访原因，不是 SHAP 值。`retention_list` 只导出高风险、允许联系、冷却期已满的客户，最后施加人数上限。

```bash
uv run --frozen pytest tests/test_retention.py
```

## 9. 第七个模块：`cli.py`

新建 `cli.py`，先定义 Typer 应用，再编写 `read_features`、`save_csv`，随后依次实现 `build_features`、`train_model`、`score`、`evaluate`、`monitor`、`export_retention`、`demo` 和 `serve`。

这些命令负责读取参数和文件，再调用已经写好的业务函数。`demo` 将各步骤串起来，不生成原始数据。`serve` 到被调用时再导入并启动接口，接口模块在下一节实现。

在文件末尾添加 `if __name__ == "__main__": app()`，使模块也可作为程序启动。然后在 `pyproject.toml` 新增命令入口：

```toml
[project.scripts]
churn = "telecom_churn.cli:app"
```

冒号前是模块路径，后面是其中的 Typer 对象。执行一次 `uv sync`，更新环境及锁文件并生成终端入口；从这一步开始才可以使用 `churn`。

编写 `tests/test_cli.py`，先将固定样本复制到临时目录的 `data/bundled/`，用 `monkeypatch.chdir` 将测试工作目录切到那里，再用 Typer 的 `CliRunner` 调用命令并检查退出码和输出文件。这样既验证固定输入目录，又不会扫描全量数据：

```bash
uv run --frozen churn --help
uv run --frozen pytest tests/test_cli.py
```

完成后，帮助应列出已经注册的子命令，CLI 测试应能串联特征、训练和评分。

## 10. 第八个模块：`api.py` 与前端

在 `api.py` 先写 `create_app`，再在其中依次注册生命周期 `lifespan`、鉴权和模型依赖函数、请求日志中间件、健康检查、模型信息、预测及首页路由，最后挂载静态目录。

启动时载入模型一次；每个请求复用模型。数据结构不合法返回 422，模型未就绪返回 503，需要密钥但验证失败返回 401。

前端按 `static/index.html` → `static/style.css` → `static/app.js` 编写。HTML 定义页面区域与元素 ID；CSS 决定布局；JS 按这些 ID 获取元素，调用模型和预测接口，把结果填到表格，并提供 CSV 下载。此页面无需 Node.js 或前端构建工具。

在 `conftest.py` 中添加 `payload(latest)`，将几条客户特征转换成请求字典；编写 `tests/test_api.py`，用 FastAPI 的 `TestClient` 发送测试请求。随后运行全部检查：

```bash
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
uv run --frozen pytest
```

## 11. 先用固定小样本演练

先运行固定 160 客户样本的命令行集成测试，检查特征构建、训练、评分和名单导出能否接起来：

```bash
uv run --frozen pytest tests/test_cli.py -v
```

测试把样本复制到临时项目的 `data/bundled/`，切换到临时工作目录，再执行命令。特征、模型和报告都写入临时目录，不会覆盖项目中的数据和训练结果。命令行的特征构建固定读取当前工作目录下的 `data/bundled/`。

## 12. 构建全量训练特征

```bash
uv run --frozen churn build-features --start 2025-01-01 --end 2025-12-01 --data-end 2026-02-01 --output data/features.csv
```

读取 20 个客户分片，构建 12 个月的历史快照。结果是客户×月份级记录，不是每天明细；已流失客户会退出后续月份，行数会少于 200,000×12。

终端打印实际行数，生成 `data/features.csv` 和 `data/features.metadata.json`。水位 `2026-02-01` 是右开边界，表示此日期前的数据已完整观察。这个日期应按数据完整情况填写。

## 13. 训练与查看结果

```bash
uv run --frozen churn train --features data/features.csv --validation-start 2025-08-01 --test-start 2025-10-01 --artifacts artifacts
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
uv run --frozen churn build-features --start 2026-01-01 --end 2026-01-01 --no-labeled --output data/latest.csv
uv run --frozen churn score --features data/latest.csv --output reports/scores.csv
uv run --frozen churn monitor --features data/latest.csv --output reports/drift.json
```

2026-01-01 作为新评分批次，不需要未来标签。批量评分应输出客户、日期、分数、等级、信号、建议和模型版本；漂移报告记录当前样本量及各特征状态。

## 15. 准备 API 请求与联系资格

已有最新特征后，为网页预测取出三名客户，并另建联系资格表。新建 `data/prepare_inputs.py`，写入以下代码，并执行一次。`data/` 下派生文件不进入版本控制：

```python
import json
from pathlib import Path
import pandas as pd

latest = pd.read_csv("data/latest.csv")  # 最新特征不包含未来标签。
request = {"customers": latest.head(3).to_dict("records")}  # API 单次最多 1,000 人，演练仅取 3 人。
Path("data/request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存请求。
eligibility = latest[["customer_id"]].copy()  # 只保留客户关联键。
eligibility["contact_allowed"] = 1  # 仅用于流程演练；业务必须使用实际授权记录。
eligibility["last_contact_at"] = None  # 演练假定无历史触达；业务需填实际触达日期。
eligibility.to_csv("data/eligibility.csv", index=False)  # 保存名单筛选的资格输入。
```

```bash
uv run --frozen python data/prepare_inputs.py
uv run --frozen churn retention-list --scores reports/scores.csv --eligibility data/eligibility.csv --capacity 100 --cooldown-days 14 --output reports/retention.csv
```

名单不超过 100 人，只有高风险客户；`priority` 从 1 开始。该操作只导出文件，不会发短信或拨打电话。

## 16. 启动服务并在 VS Code 调试

```bash
uv run --frozen churn serve
```

浏览器依次访问：

- `http://127.0.0.1:8000/health`：进程存活。
- `http://127.0.0.1:8000/ready`：模型就绪及版本号。
- `http://127.0.0.1:8000`：客户风险工作台，上传 `data/request.json` 并预测。
- `http://127.0.0.1:8000/docs`：展开 POST 预测接口，点击 Try it out，粘贴请求 JSON 后执行。

工作台显示模型和评估结果，3 名客户返回评分。JSON 顶层必须有 `customers`，不能把整个 CSV 直接粘贴到预测输入框。按终端 `Ctrl+C` 停止服务。

需要断点调试时，先停掉旧服务。创建 `.vscode/launch.json`，写入：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "客户流失 API",
      "type": "debugpy",
      "request": "launch",
      "module": "telecom_churn.cli",
      "args": ["serve"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    }
  ]
}
```

`module` 指定以模块方式启动，`args` 是子命令参数，`cwd` 是运行目录。在 `predict` 或 `score_customers` 中设置断点，选择“客户流失 API”并按 F5，再从页面提交请求，检查特征、分数和阈值。

本地默认不配置密钥；部署时配置 `CHURN_API_KEY` 和 `CHURN_ENV=production`。

## 17. 第二次运行与交付检查

完整流水线入口：

```bash
uv run --frozen churn demo
```

它从固定数据重新构建全年和最新特征、训练、评分、监控、准备接口请求与演练资格并输出名单。它会覆盖同名派生 CSV/报告并新增模型版本，所以不应在需要保留的业务输出目录直接运行。

最后执行：

```bash
uv run --frozen pytest
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
uv build
```

交付时包含源码、锁文件、文档、固定测试数据和完整 `data/bundled`。`data/bundled/metadata.json` 提供每个压缩文件的 SHA-256 和大小。构建的 Python wheel 只包含代码和静态页面，原始数据通过项目目录一起交付，不会被塞进 wheel。

## 18. 添加日常检查和自动化配置

前面的命令能单独运行后，再把它们整理成固定入口。先创建 `.gitignore`，排除环境、缓存、模型和运行结果；保留原始数据：

```gitignore
.venv/
.uv-cache/
__pycache__/
.pytest_cache/
.ruff_cache/
.coverage
*.egg-info/
.env
/data/*
!/data/bundled/
artifacts/
reports/
dist/
.DS_Store
```

创建 `Makefile`，内容如下。目标是操作名称，缩进的命令行以 Tab 开头：

```makefile
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR
.DEFAULT_GOAL := help
.PHONY: help install features demo train serve test lint format build
help:
	@echo "Targets: install features train demo serve test lint format build"
features:
	uv run --frozen churn build-features --data-end 2026-02-01
format:
	uv run --frozen ruff format src tests
install:
	uv sync --frozen
demo:
	uv run --frozen churn demo
train:
	uv run --frozen churn train
serve:
	uv run --frozen churn serve
test:
	uv run --frozen pytest --cov=telecom_churn --cov-report=term-missing
lint:
	uv run --frozen ruff check src tests
	uv run --frozen ruff format --check src tests
build:
	uv build

```

先运行 `make lint`、`make test`、`make build`，确认这些入口和手动命令结果一致。`features` 和 `train` 没有声明依赖关系，应按顺序执行。

然后创建 `.github/workflows/ci.yml`：

```yaml
name: quality # 工作流名称，在 GitHub 的 Actions 页面中显示为 quality。
on: # 声明哪些 GitHub 事件会触发这套自动检查。
  push: # 向仓库推送提交或标签时触发；未限定分支。
  pull_request: # PR（请求合并分支修改）创建、更新等默认事件发生时触发。
jobs: # 定义工作流中的作业；每个作业是一组在指定环境执行的步骤。
  test: # 当前作业的标识，名称为 test，负责安装、检查、测试和构建。
    runs-on: ubuntu-24.04 # 使用 GitHub 提供的 Ubuntu 24.04 Linux 执行机器（runner）。
    steps: # 列出同一作业依次执行的步骤；普通步骤失败后，后续普通步骤会跳过。
      - uses: actions/checkout@v5 # 调用检出代码的 Action，将当前运行对应的仓库代码放入工作目录；v5 是其版本。
      - uses: actions/setup-python@v6 # 调用准备 Python 解释器的 Action；v6 是 Action 版本。
        with: # 为上面的 setup-python Action 提供输入参数。
          python-version: "3.14" # 使用 Python 3.14 系列，与项目声明的解释器版本范围一致。
      - uses: astral-sh/setup-uv@v7 # 调用安装 uv 的 Action；uv 用来管理项目环境、依赖和构建。
        with: # 为上面的 setup-uv Action 提供输入参数。
          version: "0.9.18" # 固定 uv 自身的版本，减少开发环境与 CI 使用不同工具版本的差异。
      - name: Install dependencies # 步骤显示名称：安装项目及所需的软件包。
        run: uv sync --frozen # 使用现有 uv.lock 同步环境，不更新锁文件；安装项目运行依赖，默认也安装 dev 组。
      - name: Lint # 步骤显示名称：静态检查，即读取源码检查规则而不运行训练业务。
        run: uv run --frozen ruff check src tests # 在项目环境中检查源码和测试的 Ruff 规则；发现违规时返回非零退出码，不自动修复。
      - name: Check formatting # 步骤显示名称：检查代码排版是否符合统一格式。
        run: uv run --frozen ruff format --check src tests # 只检查格式，不改文件；需要重新格式化时此步骤失败。
      - name: Unit and integration tests # 步骤显示名称：验证局部逻辑以及多个组件之间的衔接。
        run: uv run --frozen pytest --cov=telecom_churn --cov-report=xml # 执行测试并统计项目包的覆盖率，生成 coverage.xml；失败会阻止后续普通步骤。
      - name: Build distributions # 步骤显示名称：把项目整理为可安装的 Python 发行包。
        run: uv build # 调用配置的 Hatchling 构建后端，在 dist/ 生成 wheel 和源码分发包；不部署服务。

```

工作流中的安装步骤使用已生成的锁文件，测试使用固定小样本，不运行全量训练。先用 Git 初始化项目目录，检查暂存内容并提交，再在自己的 GitHub 仓库中设置远程地址并推送。Actions 的基本概念和操作见[工具入门](tooling.md#9-github-actions把本地检查搬到另一台机器)。

## 19. 添加容器配置

本地流程通过后，再准备容器部署。镜像保存程序环境，容器是运行实例；先构建镜像，再挂载训练出的模型。新建 `Dockerfile`：

```dockerfile
FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.9.18
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev \
    && rm -rf /tmp/uv-cache \
    && useradd --uid 10001 --create-home churn \
    && mkdir -p /app/artifacts /app/data /app/reports \
    && chown -R churn:churn /app
USER churn
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"
CMD ["churn", "serve", "--host", "0.0.0.0", "--port", "8000"]

```

新建 `.dockerignore`，避免把本机环境、全量数据和训练结果发送到构建目录：

```text
.venv
.uv-cache
.git
__pycache__
data
artifacts
reports
dist
.env
```

新建 `compose.yaml`：

```yaml
services:
  api:
    build: .
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      CHURN_ENV: production
      CHURN_MODEL_PATH: /app/artifacts
      CHURN_API_KEY: ${CHURN_API_KEY:?CHURN_API_KEY is required}
    volumes:
      - ./artifacts:/app/artifacts:ro
    restart: unless-stopped
    read_only: true
    tmpfs:
      - /tmp
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL

```

新建 `.env.example`，只保存配置项和空密钥，不填真实秘密：

```dotenv
CHURN_API_KEY=
```

实际使用时另建被 Git 忽略的 `.env`，填入自己的密钥。
