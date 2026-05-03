.PHONY: help install install-dev format lint typecheck security test test-cov check all clean

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package dependencies
	pip install -e .

install-dev:  ## Install package with dev dependencies
	pip install -e ".[dev,data,huggingface]"
	pre-commit install

format:  ## Format code with ruff
	ruff format .
	ruff check --fix .

lint:  ## Lint code with ruff (without fixing)
	ruff check .

typecheck:  ## Run mypy type checking
	mypy janogpt/

security:  ## Run security checks with bandit
	bandit -r janogpt/ -c pyproject.toml

test:  ## Run tests with pytest
	pytest tests/

test-cov:  ## Run tests with coverage report
	pytest --cov=janogpt --cov-report=term-missing --cov-report=html

check: lint typecheck security  ## Run all static checks (lint, typecheck, security)

all: format check test  ## Format, check, and test everything

clean:  ## Clean up generated files
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
