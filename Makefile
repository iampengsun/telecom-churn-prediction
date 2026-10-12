UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR
.PHONY: install demo train serve test lint build
install:
	uv sync --frozen --extra lightgbm
demo:
	uv run --frozen --extra lightgbm churn demo
train:
	uv run --frozen --extra lightgbm churn train
serve:
	uv run --frozen --extra lightgbm churn serve
test:
	uv run --frozen --extra lightgbm pytest --cov=telecom_churn --cov-report=term-missing
lint:
	uv run --frozen --extra lightgbm ruff check src tests
	uv run --frozen --extra lightgbm ruff format --check src tests
build:
	uv build
