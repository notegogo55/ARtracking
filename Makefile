# Task runner for make-capable shells (Linux/macOS/CI).
# On Windows PowerShell, run the equivalent `uv ...` commands from README.md.

.PHONY: setup lint format test check-creds base-rate lock

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
