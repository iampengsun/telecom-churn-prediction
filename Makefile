UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR
.DEFAULT_GOAL := help
.PHONY: help install features demo train serve test lint format build
help:
	@echo "Targets: install features train demo serve test lint format build"
features:
	uv run --frozen --extra lightgbm churn build-features --data-end 2026-02-01
format:
	uv run --frozen --extra lightgbm ruff format src tests
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
