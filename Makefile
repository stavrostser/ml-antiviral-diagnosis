.PHONY: test lint format lock

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run black --check .

format:
	uv run ruff check . --fix
	uv run black .

lock:
	uv lock
