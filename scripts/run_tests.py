"""Compatibility test entry point; prefer ``uv run --locked pytest``."""

import sys

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
