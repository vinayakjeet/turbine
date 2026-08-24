.PHONY: bench test lint conventions status clean

bench:
	uv run python -m turbine sweep --label mock
	uv run python scripts/status.py

test:
	uv run pytest

lint:
	uv run ruff check .

conventions:
	uv run python scripts/check_conventions.py

status:
	uv run python scripts/status.py

clean:
	rm -rf results .pytest_cache .ruff_cache
