# Python 开发工具入门


## 1. 先认识工具的分工

| 工具 | 是什么 | 用来做什么 |
|---|---|---|
| uv | Python 项目和依赖管理工具 | 准备环境、安装软件包、运行命令 |
| pytest | 自动化测试框架 | 执行测试，检查程序结果是否符合预期 |
| Ruff | 静态检查和格式化工具 | 找出部分代码问题，统一排版 |
| Typer | 命令行开发库 | 把 Python 函数做成可以从终端调用的命令 |
| Make | 按规则执行命令的工具 | 给一组常用命令起简短名称 |
| Git | 版本管理工具 | 保存文件修改历史，查看和恢复版本 |
| GitHub Actions | GitHub 上的自动化平台 | 推送代码后，在另一台机器上自动执行检查 |

## 2. uv：从空目录开始

### 2.1 依赖和虚拟环境

虚拟环境是为当前项目单独准备的 Python 包安装目录。两个项目即使需要不同版本的软件包，也可以各用各的环境，减少冲突。uv 默认使用项目中的 `.venv/`。

先安装独立的 uv 工具，再由它安装 Python。电脑上不需要预先准备 Python。打开终端执行：

```bash
curl -LsSf https://astral.sh/uv/0.9.18/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv python install 3.14
```

安装器通常把命令放到 `~/.local/bin/`。

`uv python install 3.14` 安装解释器，不会替换电脑上已有的其他 Python 版本。uv 本身独立于这些解释器；它管理的 Python 通常位于项目外，项目依赖则安装到 `.venv/`。可以用 `command -v uv`、`uv python dir`、`uv cache dir` 分别查看 uv 命令、Python 安装目录和下载缓存目录。

然后创建练习目录：

```bash
mkdir tool-practice
cd tool-practice
uv init --python 3.14
uv sync
```

`uv init --python 3.14` 按指定的 Python 版本初始化项目配置，`uv sync` 安装依赖并准备环境。


### 2.2 配置和锁文件

配置文件保存项目设置。`pyproject.toml` 使用 TOML 格式，以 `[分组]` 和 `名称 = 值` 表达内容；例如 `[project]` 保存名称、版本、Python 版本范围和依赖声明。

锁文件 `uv.lock` 记录解析后的具体依赖版本。依赖解析是在各软件包的版本要求之间找到兼容组合的过程；把结果锁下来，可以减少换台机器后安装到不同版本的情况。文件中的哈希是根据下载内容计算的摘要，用来检查下载文件是否符合记录。

| 文件或目录 | 如何产生 | 该怎么维护 |
|---|---|---|
| `pyproject.toml` | 初始化项目或手动创建 | 随代码保存，声明项目需要什么 |
| `.python-version` | 初始化或选择 Python 版本时创建 | 保存希望使用的解释器版本 |
| `uv.lock` | uv 解析依赖后生成 | 随代码保存，不手写版本和下载哈希 |
| `.venv/` | 同步环境时创建 | 每台机器各自生成，不提交到 Git |

### 2.3 添加依赖和执行程序

运行依赖用于执行程序；开发依赖用于测试和检查。现在安装后面练习会用到的工具：

```bash
uv add typer
uv add --group dev pytest pytest-cov ruff
```

`uv add` 更新配置和锁文件，并同步环境；`--group dev` 把工具放进开发依赖组。软件包不再需要时，用 `uv remove 包名` 删除；“包名”要替换成实际名称。

新建 `hello.py`，写入：

```python
print("Hello, Python")
```

执行：

```bash
uv run python hello.py
uv run python --version
```

`uv run` 在项目环境中执行后面的命令，不需要先激活虚拟环境。激活环境是调整当前终端的命令搜索路径，让它优先使用某个环境中的程序；由 uv 选择环境，可以省掉这一步。

| 命令或选项 | 什么时候用 |
|---|---|
| `uv sync` | 首次准备环境，或修改依赖后同步 |
| `uv lock` | 重新解析并更新锁文件 |
| `uv sync --locked` | 安装前检查声明与锁文件一致，需要更新锁文件时失败 |
| `uv sync --frozen` | 已有锁文件时按它安装，不更新，也不检查声明是否需要重新锁定 |
| `uv sync --no-dev` | 只准备程序运行环境，不安装开发工具 |
| `uv run ...` | 在项目环境中运行 Python 或工具 |

从零开始时先用 `uv sync` 生成锁文件；还没有锁文件时，不能照搬 `--frozen` 命令。

## 3. VS Code：编辑、运行与调试

这个项目强烈建议大家安装 VS Code。 另外也需要安装Microsoft 的 Python 扩展，配合 Pylance 提供代码补全和分析；需要逐步执行代码时，再安装 Python Debugger。

用“文件 → 打开文件夹”打开整个 `tool-practice/`。在“查看 → 命令面板”中搜索 `Python: Select Interpreter`，选择项目中的 `.venv/bin/python`。

选择解释器是为了让编辑器使用当前项目的依赖。终端能运行、编辑器却提示找不到包时，先检查这里。

调试是暂停程序并检查执行过程。在 `hello.py` 的行号左侧点击设置断点，再选择“运行和调试”启动 Python 文件。程序暂停后可以查看变量、逐行执行。检查简单脚本时，这通常比到处添加 `print` 更方便。

## 4. pytest：先写一个能检查结果的测试

### 4.1 测试、断言和回归

自动化测试把“输入是什么，结果应该是什么”写成代码。以后修改函数，再运行同一组测试，就能发现原本正常的行为是否被改坏，这叫回归测试。

新建 `calculator.py`：

```python
def add(a, b):
    return a + b
```

创建 `tests/` 目录，再新建 `tests/test_calculator.py`：

```python
from calculator import add


def test_add():
    assert add(2, 3) == 5
```

`assert` 是断言，表示后面的条件应该成立。条件为假时，pytest 报告失败。测试发现是框架查找测试的过程；这里采用 `test_*.py` 文件和 `test_*` 函数，`*` 可以匹配其他文字。

执行：

```bash
uv run python -m pytest
```

`python -m pytest` 用当前解释器启动 pytest，并让当前练习目录参与模块查找。看到 `1 passed` 表示这个测试通过。可以暂时把函数改成 `return a - b`，运行后观察失败信息，再改回去。

单元测试检查较小的逻辑，例如一个计算函数；集成测试检查多个组件的配合，例如读取文件、计算结果再保存。先写小测试，等模块接起来后再添加集成测试。

### 4.2 选择要运行的测试

```bash
uv run python -m pytest tests/test_calculator.py
uv run python -m pytest tests/test_calculator.py::test_add
uv run python -m pytest -k add -v
uv run python -m pytest -x
```

依次运行一个文件、一个函数、名称匹配 `add` 的测试，以及遇到首次失败就停止。`::` 指定文件内的测试，`-k` 按名称筛选，`-v` 显示详细结果。

退出码是程序结束时返回的数字。pytest 全部通过通常返回 0，测试失败等情况返回非零；后面的自动化流程会据此判断成功或失败。`skipped` 表示跳过，不能当作已经验证通过。

### 4.3 fixture：复用测试准备工作

多个测试可能需要相同的数据、临时文件或模型。fixture 是由 pytest 管理的准备函数，用来集中提供这些资源，省去每个测试重复准备的代码。

新建 `tests/conftest.py`：

```python
import pytest


@pytest.fixture
def fees():
    return [39, 89, 169]
```

`@pytest.fixture` 是装饰器，即写在函数上方、用于附加行为的声明；这里用来把 `fees` 注册为 fixture。`conftest.py` 是 pytest 自动加载的配置文件，其中的 fixture 可以供同目录及子目录的测试使用。

在 `tests/test_calculator.py` 中追加：

```python
def test_total_fee(fees):
    assert sum(fees) == 297
```

测试函数写出参数名 `fees`，pytest 就查找同名 fixture，把返回的列表传入。这个过程叫依赖注入，即框架准备好测试需要的对象，再交给测试使用。手动调用 `test_total_fee()` 不会自动获得列表，需要让 pytest 执行。

默认每个使用 fixture 的测试都会得到一次新的准备结果。加上 `scope="session"` 可以在一次 pytest 运行中复用同一个结果，适合准备成本高的资源。共享可修改对象时要小心，一个测试改过的数据可能影响另一个测试。

### 4.4 临时目录：别让测试覆盖正式文件

临时目录是测试专用的工作空间，适合写输入副本和输出结果。pytest 内置 `tmp_path`，为每个测试提供独立目录，无需自己定义。

在测试文件中追加：

```python
def test_write_file(tmp_path):
    output = tmp_path / "result.txt"
    output.write_text("5", encoding="utf-8")
    assert output.read_text(encoding="utf-8") == "5"
```

`tmp_path` 是 `Path` 路径对象，可以用 `/` 拼接文件名。测试只会写入分配的目录。pytest 可能保留近期临时目录供排错，不保证每次结束立即清空。

另一个内置 fixture 是 `tmp_path_factory`，它在一次运行期间可用，可调用 `.mktemp("名称")` 创建目录，适合会话级 fixture 使用。

### 4.5 参数化与异常测试

参数化是用多组输入执行同一个测试，每组分别报告结果。将测试文件顶部的导入和原 `test_add` 改成下面这样，其余测试函数保留：

```python
import pytest

from calculator import add


@pytest.mark.parametrize("a,b,expected", [(2, 3, 5), (-1, 1, 0), (0, 0, 0)])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

此时 pytest 会分别检查三组加法输入，不用复制三个测试函数。

异常是程序无法正常继续时抛出的错误。某些输入就应该被拒绝，测试要确认它确实被拒绝，而不是把任何报错都当成程序有问题：

```python
def test_invalid_integer():
    with pytest.raises(ValueError, match="invalid literal"):
        int("abc")
```

`with` 圈定要检查的代码，`ValueError` 表示值不符合要求，`match` 检查错误信息是否匹配文字模式。如果没有报错、异常类型错误或信息不匹配，这个测试都会失败。

### 4.6 覆盖率：哪些代码执行过

覆盖率是测试执行到的代码行占比。pytest-cov 是增加这项统计的插件：

```bash
uv run python -m pytest --cov=calculator --cov-report=term-missing
```

`--cov` 指定统计模块，`term-missing` 在终端列出未覆盖的行。覆盖率能帮助找遗漏，但执行到一行不代表已经验证它在所有输入下都正确。

## 5. Ruff：规则检查和代码排版

静态检查是读取源代码、按规则找问题，不执行程序业务。例如，导入了没使用的模块，或引用了没定义的变量。格式化则统一空格、缩进和换行，减少代码评审中无意义的排版差异。

先执行只检查、不改文件的命令：

```bash
uv run ruff check .
uv run ruff format --check .
```

`.` 表示当前目录。第一条检查规则，第二条检查排版；即使函数算错结果，只要没有违反这些规则，也可能通过，因此还需要测试。

需要修复时执行：

```bash
uv run ruff check --fix .
uv run ruff format .
```

`--fix` 只修复工具支持的问题，其余按文件、行号和规则编号处理；例如 `F401` 表示未使用的导入。修改后查看文件差异，再重新运行检查。

可以在 `pyproject.toml` 末尾追加配置：

```toml
[tool.ruff]
line-length = 100
target-version = "py314"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

`line-length` 是格式化目标行宽；`target-version` 表示允许使用的 Python 语法版本；`select` 选择规则组。`E` 检查部分风格问题，`F` 检查名称等问题，`I` 检查导入顺序，`B` 检查常见错误模式，`UP` 提供较新语法建议。

## 6. Typer：给函数加一个终端入口

命令行程序通过终端参数接收输入。Typer 根据 Python 函数的参数名、类型注解和默认值生成参数解析与帮助，适合把脚本整理成可重复调用的工具。

新建 `cli.py`：

```python
import typer

app = typer.Typer()


@app.command()
def greet(name: str = "Peter"):
    print(f"Hello, {name}")


if __name__ == "__main__":
    app()
```

执行：

```bash
uv run python cli.py --help
uv run python cli.py --name Alice
```

`name: str` 是类型注解，说明接收字符串；`"Peter"` 是没传参数时使用的默认值。这个例子只有一个命令，直接使用 `--name`；注册多个命令后，Typer 会以子命令区分操作。

以后将业务代码组织成 Python 包时，可以在 `[project.scripts]` 中注册短命令。Python 包是可导入的一组模块，模块通常对应一个 `.py` 文件。电信项目的 `churn` 入口要等到编写命令行模块时再注册，具体见[实操教程](tutorial.md#9-第七个模块clipy)。

## 7. Make

Make 按 Makefile 中的规则执行命令。规则中的目标是操作名称，下面的缩进行是要执行的命令。它也支持声明依赖，先处理依赖再处理目标；没有写出的依赖不会自动推断。

新建 `Makefile`，命令行开头必须是 Tab：

```makefile
.PHONY: test lint format

test:
	uv run python -m pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
```

`.PHONY` 声明这些名称代表操作，不是要生成的同名文件。之后可执行 `make test`、`make lint`、`make format`，少记几条长命令。

Make 是系统工具，不会随着 Python 依赖自动安装。`make --version` 可以检查是否可用；没安装时，直接执行规则中的 uv 命令也可以。

## 8. Git：保存一次可以回头查看的修改

Git 把项目文件及修改历史组织成仓库。工作区是你正在编辑的文件，暂存区保存准备放进下一次提交的变更，提交历史保存已经确认的版本。

先创建 `.gitignore`，告诉 Git 忽略可重新生成的环境、缓存和测试产物：

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.coverage
.env
```

它只影响未跟踪文件，不会自动删除已经纳入版本管理的内容。`.env` 常保存实际环境配置和密钥，也应忽略。

然后执行：

```bash
git init
git status
git add .
git diff --cached
git commit -m "Add calculator and tests"
```

`init` 初始化仓库；`status` 查看状态；`add` 选择本次提交内容；`diff --cached` 检查暂存区具体变化；`commit` 保存一个带说明的版本。提交前确认没有误选环境文件或密钥。若首次提交提示身份未配置，先按提示设置提交者名称和邮箱。

GitHub 是托管远程仓库的平台。远程仓库创建好后，按页面给出的地址配置 `origin`，再推送本地分支。`origin` 是远程地址简称；分支是一条可独立推进的提交历史，常用 `main` 保存主线。不要把示例地址当作自己的地址直接使用。

## 9. GitHub Actions：把本地检查搬到另一台机器

### 9.1 先理解工作流

持续集成（CI）是在推送或请求合并代码时自动运行检查，让问题尽早暴露。GitHub Actions 是执行这种自动化的平台，CI 是它的一种用途。

| 概念 | 含义 |
|---|---|
| workflow | 工作流，一整套自动化流程 |
| event | 触发事件，例如推送代码 `push` |
| job | 作业，包含一组步骤 |
| runner | 执行作业的机器 |
| step | 步骤，作业内的一次操作 |
| action | 可复用的步骤组件，例如安装 Python |
| Pull Request（PR） | 请求把某个分支的修改合入另一个分支，供审查和检查 |

runner 上没有你本机的 `.venv` 或未提交文件，必须先取代码、准备解释器、安装依赖，再执行检查。同一作业的步骤通常顺序执行并共享工作目录；不同作业不能假设会自动共享文件。

### 9.2 写第一个工作流

先在本机让测试和 Ruff 检查通过，再创建 `.github/workflows/check.yml`，写入：

```yaml
name: checks # Actions 页面显示的工作流名称。
on: # 定义触发事件。
  push: # 推送代码时执行。
  pull_request: # PR 创建、更新等默认事件发生时执行。
jobs: # 定义作业。
  test: # 作业标识。
    runs-on: ubuntu-24.04 # 指定 runner 的操作系统。
    steps: # 按顺序执行的步骤。
      - uses: actions/checkout@v5 # 将仓库代码取到 runner。
      - uses: actions/setup-python@v6 # 配置 Python 解释器。
        with: # 传给上一个 Action 的输入参数。
          python-version: "3.14" # Python 的版本。
      - uses: astral-sh/setup-uv@v7 # 安装 uv。
        with: # 指定 uv 的安装参数。
          version: "0.9.18" # uv 的版本。
      - run: uv sync --locked # 检查锁文件一致后安装依赖。
      - run: uv run ruff check . # 检查代码规则。
      - run: uv run ruff format --check . # 检查排版，不修改文件。
      - run: uv run python -m pytest # 执行测试。
```

YAML 是通过键值和缩进表达层级的配置格式，`-` 表示列表项。`uses` 调用现成组件，`run` 执行终端命令，`with` 向 Action 传参数。`@v5`、`@v6` 是各组件自己的版本，和 `python-version` 无关。

把工作流、测试、配置和 `uv.lock` 一起提交并推送。在 GitHub 仓库的 Actions 页面查看运行，展开失败步骤看日志。普通步骤失败后，后续普通步骤通常会跳过。

遇到 Python 异常时，日志中的 traceback 是函数调用记录：先读最后的异常类型和消息，再往上找自己的文件和行号。修复后重新提交会触发新检查；仅在确定是临时下载或运行机器问题时才考虑重跑。

## 10. 构建与服务：后面会用到什么

构建是将代码整理成可安装的发行包。wheel（`.whl`）保存整理好的安装内容；源码分发包（通常为 `.tar.gz`）保存源码和构建配置。构建后端是实际打包的工具，例如 Hatchling；`uv build` 调用项目配置中的后端生成文件。上面的简单脚本练习尚未配置安装包，先不用执行构建。
