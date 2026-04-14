import pytest
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

# Global accumulator used by the track_cost decorator in harness.py
pytest.cost = 0.0
pytest.messages = None

@pytest.fixture
def default_agent_options():
    tools = ["Skill", "Read", "Grep", "Glob", "Write"]
    options = ClaudeAgentOptions(
        tools=tools,
        allowed_tools=tools,
        permission_mode="dontAsk",
        plugins=[{'type': 'local', 'path': str(Path(__file__).parent.parent)}],
        max_turns=10,
        max_budget_usd=1.00,
        setting_sources=["local", "project"],
    )
    return options

@pytest.fixture(autouse=True)
def track_test_cost(record_property):
    """Reset cost before each test and record it in JUnit XML after."""
    pytest.cost = 0.0
    yield
    record_property("cost", pytest.cost)
