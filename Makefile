# Task runner for make-capable shells (Linux/macOS/CI).
# On Windows PowerShell, run the equivalent `uv ...` commands from README.md.

.PHONY: all setup lint format test check-creds base-rate lock run-all report

# Full chain on the sample window: setup -> tests -> end-to-end -> report.
# Stage A needs a cached sample or JSOC_EMAIL (see docs/reproducibility.md).
all: setup test run-all report

run-all:
	uv run solarflare run-all -w ar11158_feb2011

report:
	uv run python scripts/build_report.py

setup:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

check-creds:
	uv run solarflare check-credentials

base-rate:
	uv run solarflare base-rate

lock:
	uv lock
	uv export --format requirements-txt -o requirements-lock.txt
