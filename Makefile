.PHONY: help install test lint format type-check clean docs

help:
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install package and dependencies
	pip install -e .
	pip install -r requirements-test.txt

test: ## Run tests
	pytest tests/ -v --cov=vision_detect_segment --cov-report=term-missing

lint: ## Run linters
	ruff check vision_detect_segment/
	black --check vision_detect_segment/

format: ## Format code
	ruff check --fix vision_detect_segment/
	black vision_detect_segment/

type-check: ## Run type checking
	mypy vision_detect_segment/ --config-file=pyproject.toml

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov coverage.xml

docs: ## Build documentation with MkDocs
	mkdocs build
