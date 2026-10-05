"""Keep automated tests offline unless a test explicitly injects a language client."""

import os

os.environ["LLM_ENABLED"] = "false"
os.environ["CONVERSATION_STORAGE_ENABLED"] = "false"
os.environ["RATE_LIMIT_BACKEND"] = "memory"

import pytest


@pytest.fixture(autouse=True)
def isolated_budgets():
    from app.monitoring.budgets import budgets

    with budgets.lock:
        budgets.items.clear()
    yield
