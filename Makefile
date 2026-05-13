.PHONY: test test-fast lint install install-dev run-api dry-run clean help

## ── Development ──────────────────────────────────────────────────────────────

install:        ## Install production dependencies
	pip install -r requirements.txt

install-dev:    ## Install dev + test dependencies
	pip install -r requirements.txt
	pip install pytest pytest-anyio httpx

test:           ## Run full test suite
	python -m pytest tests/ -v --tb=short

test-fast:      ## Run tests skipping slow integration tests
	python -m pytest tests/ -v --tb=short -m "not integration"

test-cov:       ## Run tests with coverage report
	python -m pytest tests/ --tb=short --cov=src --cov-report=term-missing

lint:           ## Run basic syntax check on all source files
	python -m py_compile src/**/*.py src/*.py && echo "Syntax OK"

## ── Running ──────────────────────────────────────────────────────────────────

run-api:        ## Start the FastAPI server (n8n integration)
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

dry-run:        ## Validate full pipeline without rendering or uploading
	python main.py pipeline --topic "test run" --dry-run --no-download

## ── Housekeeping ─────────────────────────────────────────────────────────────

clean:          ## Remove __pycache__, .pyc, temp render files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf output/_tmp_* output/*/audio output/*/_tmp_*

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
