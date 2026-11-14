# 开发工具入门

本项目使用 Python 3.14。开始编写业务代码前，需要区分环境管理、业务命令、自动化测试、静态检查和持续集成的职责。所有命令默认在项目根目录执行。

## 1. 组件与配置文件

| 组件 | 作用 | 配置或入口 |
|---|---|---|
| Python | 执行 Python 程序 | `.python-version` 指定 3.14 |
| VS Code | 编辑代码、导航、调试和查看测试 | `.vscode/` |
| uv | 管理解释器、虚拟环境、依赖和发行包 | `pyproject.toml`、`uv.lock` |
| Typer | 根据 Python 函数生成命令及参数帮助 | `src/telecom_churn/cli.py` |
| churn | 本项目的业务命令行程序 | `[project.scripts]` |
| pytest | 执行测试并检查程序行为 | `tests/`、`[tool.pytest.ini_options]` |
| pytest-cov | 统计测试覆盖的代码行 | pytest 的 `--cov` 参数 |
| Ruff | 检查代码规则、统一代码格式 | `[tool.ruff]` |
| Make | 用目标名称组织常用命令 | `Makefile` |
| Git | 保存本地文件版本及提交历史 | `.git/`、`.gitignore` |
| GitHub Actions | 在独立环境自动执行持续集成 | `.github/workflows/ci.yml` |
| Hatchling | 构建 Python 发行包 | `[build-system]` |
| FastAPI / Uvicorn | 定义 HTTP 接口 / 运行接口服务 | `api.py` / `churn serve` |

`pyproject.toml` 是项目声明，`uv.lock` 是依赖解析结果，`Makefile` 是操作入口。它们各自承担不同职责，不能相互替代。

## 2. uv：环境和依赖管理

### 2.1 首次安装

在已有 Python 的终端安装与 CI 一致的 uv 版本：

```bash
python -m pip install uv==0.9.18
uv --version
uv python install 3.14
```

系统使用 `python3` 时，第一行改为 `python3 -m pip ...`；Windows 可使用 `py -m pip ...`。随后获取项目并安装依赖：

```bash
git clone https://github.com/iampengsun/telecom-churn-prediction.git
cd telecom-churn-prediction
uv sync --frozen --extra lightgbm
uv run --frozen --extra lightgbm python --version
```

已有本地项目时跳过克隆。仓库包含约 752 MiB 的压缩业务数据，首次获取需要下载这些文件。

`.python-version` 选择 Python 3.14；`pyproject.toml` 中的 `requires-python` 声明允许的解释器范围。`uv sync` 创建 `.venv`，安装项目及依赖，默认也安装 `dev` 组中的 pytest、Ruff 等工具。

### 2.2 常用命令和参数

| 命令或参数 | 含义 |
|---|---|
| `uv sync` | 根据项目声明同步环境，必要时更新锁文件 |
| `--frozen` | 使用现有锁文件，不更新它，也不检查其与项目声明是否完全同步 |
| `--locked` | 要求锁文件与项目声明一致，需要更新时直接失败 |
| `--extra lightgbm` | 安装项目声明的 LightGBM 可选依赖 |
| `--no-dev` | 不安装开发依赖，适用于服务运行环境 |
| `uv run ...` | 在项目虚拟环境中执行命令，无需手动激活环境 |
| `uv lock` | 重新解析依赖并更新 `uv.lock` |
| `uv build` | 在 `dist/` 中生成 wheel 和源码分发包 |

`.venv/` 是当前机器的环境，不提交到 Git；`uv.lock` 记录具体依赖版本和下载校验值，应提交到 Git。新增依赖时使用 uv 同时维护配置与锁文件：

```bash
uv add 包名
uv add --group dev 开发工具包名
uv remove 包名
```

`包名` 是占位符，使用时替换成实际依赖名称。依赖变更通常需要同时提交 `pyproject.toml` 和 `uv.lock`；不要手动修改下载文件的哈希。

### 2.3 VS Code 解释器

安装 Microsoft 的 Python、Pylance 和 Python Debugger 扩展。在命令面板执行 `Python: Select Interpreter`，选择项目 `.venv`。类 Unix 系统解释器位于 `.venv/bin/python`，Windows 位于 `.venv\Scripts\python.exe`。

编辑器的解释器用于补全、调试和测试发现；`uv run` 则使用项目运行环境。终端运行正常而编辑器提示找不到模块时，应先检查解释器选择。

## 3. churn：项目业务命令

### 3.1 churn 是什么

`churn` 是本项目安装后提供的命令行程序，不是 Python 关键字，也不是需要单独下载的工具。这里的英文 churn 指客户流失。

注册配置位于 `pyproject.toml`：

```toml
[project.scripts]
churn = "telecom_churn.cli:app"
```

左侧是命令名；右侧冒号前是 Python 模块路径，冒号后是模块中的 Typer 应用对象。`uv sync` 安装项目时，会在虚拟环境中生成对应的终端入口。

执行顺序是：终端运行 `churn` → 进入 `telecom_churn.cli:app` → Typer 查找子命令并解析参数 → 调用相应 Python 函数 → 函数调用业务模块。

```bash
uv run --frozen --extra lightgbm churn --help
uv run --frozen --extra lightgbm churn train --help
```

第一条查看所有子命令，第二条查看训练参数和默认值。同一个应用也可以通过模块方式调用：

```bash
uv run --frozen --extra lightgbm python -m telecom_churn.cli --help
```

直接输入 `churn` 需要先激活环境或配置 PATH。文档统一使用 `uv run`，避免依赖终端的激活状态。

### 3.2 子命令、参数与布尔开关

```bash
uv run --frozen --extra lightgbm churn train --features data/features.csv --algorithm lightgbm
```

其中 `uv run` 选择环境，`churn` 是程序名，`train` 是子命令，`--features` 和 `--algorithm` 是命名选项，后面分别跟选项值。

| 子命令 | 输入 | 输出或行为 |
|---|---|---|
| `build-features` | 原始表、日期范围和完整性水位 | 特征 CSV 与来源元数据 |
| `train` | 带标签特征 | 模型版本、验证和测试指标 |
| `score` | 无标签特征及模型 | 客户风险评分 CSV |
| `evaluate` | 成熟标签特征及模型 | 评估 JSON，不重新训练 |
| `monitor` | 当前特征及训练参考分布 | 漂移报告 JSON |
| `retention-list` | 评分、联系资格、容量、冷却期 | 挽留名单 CSV |
| `demo` | 固定原始数据 | 串联特征、训练、评分、监控和名单导出 |
| `serve` | 已保存模型及服务配置 | 持续运行 API 与网页工作台 |

`build-features --no-labeled` 是布尔开关，表示不构建标签，不需要在后面写 `false`。`--data-end` 则需要实际日期值。

### 3.3 Typer 与 Python 函数

`cli.py` 使用 `@app.command("build-features")` 注册子命令，并由 `build_features` 函数实现。命令名可包含短横线，Python 函数名使用下划线。

Typer 根据函数签名、类型注解和默认值生成参数解析及帮助。`Annotated[Path, typer.Option()]` 等声明用于显式指定选项。学习或增加命令时，先阅读同文件中的 `build_features`，再用 `--help` 对照函数参数与终端选项。

## 4. pytest：自动化测试

### 4.1 测试发现与断言

pytest 默认发现 `test_*.py` 中的 `test_*` 函数。测试使用 Python `assert` 检查预期结果；断言失败或出现未处理异常时，测试失败。

本项目配置如下：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

`testpaths` 限定默认搜索目录，`-q` 减少终端日志。可先阅读 `tests/test_modeling.py` 的 `test_top_k_metrics_and_single_class`：它用已知标签和分数检查指标公式，不依赖模型恰好产生某个分数。

### 4.2 基础使用

```bash
uv run --frozen --extra lightgbm pytest
uv run --frozen --extra lightgbm pytest tests/test_features.py
uv run --frozen --extra lightgbm pytest tests/test_modeling.py::test_top_k_metrics_and_single_class
uv run --frozen --extra lightgbm pytest -k window -v
uv run --frozen --extra lightgbm pytest -x
```

依次表示全部测试、单个文件、单个函数、名称含 `window` 的测试，以及遇到首个失败就停止。成功退出码为 0，失败为非零；CI 使用退出码判断是否继续。

### 4.3 fixture 与参数化

`tests/conftest.py` 定义共享 fixture。测试函数声明 `raw_dir`、`features`、`bundle` 参数时，pytest 按名称注入相应资源，无需手动调用准备函数。

`@pytest.fixture(scope="session")` 表示一次运行只准备一次资源。`tmp_path` 为单个测试提供临时目录，`tmp_path_factory` 创建会话级临时目录，避免覆盖正式数据和模型。

`@pytest.mark.parametrize` 用多组输入重复执行同一个测试；`pytest.raises` 检查非法输入是否产生预期异常。对应例子是 `test_invalid_features_rejected` 和 `test_immature_labels_rejected`。

测试使用 `tests/data/` 中的固定 160 客户子集，不扫描全量业务明细，也不生成原始数据。多数测试使用快速基线，另有 LightGBM 算法检查。

### 4.4 覆盖率和测试职责

```bash
uv run --frozen --extra lightgbm pytest --cov=telecom_churn --cov-report=term-missing
```

pytest-cov 显示执行过的代码行比例和未覆盖行。覆盖率不能证明业务正确性，关键边界仍需要明确断言。

| 文件 | 检查内容 |
|---|---|
| `test_features.py` | 窗口边界、未来数据隔离、标签成熟、分片一致性、重复键 |
| `test_modeling.py` | 时间切分、预处理隔离、模型工件、校验和、指标、漂移 |
| `test_retention.py` | 授权、冷却期、名单排序与容量 |
| `test_cli.py` | 特征、训练、评分、评估、监控及名单的命令衔接 |
| `test_api.py` | HTTP 参数验证、鉴权、模型就绪与预测响应 |

## 5. Ruff：静态检查和格式化

`ruff check` 检查代码规则，`ruff format` 统一排版。Ruff 不执行模型训练，也不判断业务指标是否合理。

```bash
uv run --frozen --extra lightgbm ruff check src tests
uv run --frozen --extra lightgbm ruff format --check src tests
```

上述命令只检查、不改文件，适用于提交前和 CI。主动修复或格式化使用：

```bash
uv run --frozen --extra lightgbm ruff check --fix src tests
uv run --frozen --extra lightgbm ruff format src tests
```

`--fix` 只能处理支持自动修复的问题。其他问题需按文件、行号和规则编号修改；修改后用 `git diff` 查看变化。

配置中的 `E` 是部分代码风格规则，`F` 检查未定义或未使用名称等问题，`I` 检查导入顺序，`B` 检查常见错误模式，`UP` 提供较新语法建议。格式化行宽为 100，目标版本为 Python 3.14，具体忽略项见 `pyproject.toml`。

## 6. Makefile：常用命令入口

Make 按 `Makefile` 中的目标执行命令。目标后有冒号，命令行必须以 Tab 缩进，而不是若干空格：

```makefile
train:
	uv run --frozen --extra lightgbm churn train
```

`make train` 等价于执行下面的 uv 命令。`.PHONY` 声明目标不对应磁盘上的同名文件。`UV_CACHE_DIR ?= ...` 表示仅在外部未设置时使用默认值，`export` 把变量传给子命令。

| 命令 | 用途 |
|---|---|
| `make` / `make help` | 显示目标列表 |
| `make install` | 同步依赖 |
| `make features` | 构建全量历史特征 |
| `make train` | 读取已有特征并训练 |
| `make demo` | 完整业务流水线 |
| `make serve` | 读取已有模型并启动服务 |
| `make test` | 测试与覆盖率 |
| `make lint` | 规则和格式检查，不改文件 |
| `make format` | 格式化源码与测试 |
| `make build` | 构建发行包 |

目标之间没有自动声明业务依赖，需先 `make features` 再 `make train`；或者使用 `make demo` 串联流程。

`make --version` 可检查系统是否安装 Make。Windows 未安装 Make 时可使用 WSL，或直接执行 Makefile 内的 uv 命令。安装 Python 项目依赖不会自动安装 Make。

## 7. Git 与 GitHub CI

### 7.1 提交和推送

Git 在本地记录版本，GitHub 托管远程仓库。常用流程为：

```bash
git status
git diff
git add README.md docs pyproject.toml uv.lock
git diff --cached
git commit -m "docs: explain development tools"
git push origin main
```

`add` 把指定变更放入暂存区；`commit` 保存暂存区快照；`push` 上传本地提交。提交前检查暂存区，避免误提交密钥和运行产物。

`.gitignore` 排除虚拟环境、模型、报告和派生数据，固定原始数据 `data/bundled/` 属于交付内容，保留在版本控制中。`.env.example` 用于说明配置项，实际密钥保存在被忽略的 `.env` 或部署环境中。

### 7.2 持续集成是什么

CI 是 Continuous Integration，即持续集成。代码推送或提出 Pull Request 后，系统在独立环境自动执行检查，发现环境差异或代码缺陷。

本项目使用 GitHub Actions，配置文件为 `.github/workflows/ci.yml`，工作流名为 `quality`，触发事件为 `push` 和 `pull_request`。不在本地手动运行测试，并不意味着推送后不会执行 CI。

| YAML 字段 | 含义 |
|---|---|
| `name` | 工作流名称 |
| `on` | 触发事件 |
| `jobs` | 可调度的作业集合 |
| `runs-on` | 作业运行环境，本项目为 Ubuntu |
| `steps` | 同一作业内依次执行的步骤 |
| `uses` | 调用已有 Action |
| `with` | 传给 Action 的参数 |
| `run` | 执行终端命令 |

### 7.3 本项目的 CI 流程

1. `actions/checkout` 检出对应提交。
2. `actions/setup-python` 配置 Python 3.14。
3. `astral-sh/setup-uv` 安装 uv 0.9.18。
4. 同步锁定依赖并安装 LightGBM。
5. Ruff 检查代码规则和格式，不自动改写文件。
6. pytest 执行固定小样本测试，生成覆盖率 XML。
7. `uv build` 构建 Python 发行包。

某一步失败，后续普通步骤通常跳过。CI 不执行全量训练，不长期启动服务，也不自动部署网站或模型。检出仓库仍包含原始数据，下载时间取决于网络。

### 7.4 查看结果

打开 GitHub 仓库的 **Actions** 标签，选择 `quality`，进入对应提交的运行记录，展开失败步骤。Pull Request 页面也显示关联检查。排队或运行中的状态不能视为通过。

| 失败步骤 | 排查方向 |
|---|---|
| Install dependencies | Python 版本、锁文件、包下载 |
| Lint | 根据文件位置和规则编号修改代码 |
| Check formatting | 本地 `make format` 后提交变化 |
| Unit and integration tests | 查看断言和 traceback，定位对应测试 |
| Build distributions | 检查项目元数据、源码路径和打包配置 |

代码修复后正常提交并推送，会触发新运行。仅在确认是临时网络或 runner 故障时考虑重跑同一次 CI。

## 8. 构建与服务

`uv build` 调用 Hatchling，在 `dist/` 生成 wheel 和源码分发包。两者包含代码及相关发布文件，不包含全量数据、模型和运行报告；原始数据随项目仓库交付。

`churn serve` 调用 Uvicorn，由 FastAPI 的应用工厂 `create_app` 创建接口。服务启动时载入模型，修改模型版本后需要重启。`/docs` 提供接口参数与交互式调用页面。

理解上述组件后，按 [VS Code 实操教程](tutorial.md) 完成模块编写和全流程运行。
