import pytest
from pathlib import Path

# Global accumulator used by the track_cost decorator in harness.py
pytest.cost = 0.0
pytest.messages = None

@pytest.fixture
def default_agent_options():
    from claude_agent_sdk import ClaudeAgentOptions

    options = ClaudeAgentOptions(
        allowed_tools=["Skill", "Read", "Bash(oc:*)", "Bash(curl:*)", "Grep", "Glob"],
        permission_mode="dontAsk",
        plugins=[{'type': 'local', 'path': str(Path(__file__).parent.parent)}]
    )
    return options

@pytest.fixture(autouse=True)
def track_test_cost(record_property):
    """Reset cost before each test and record it in JUnit XML after."""
    pytest.cost = 0.0
    yield
    record_property("cost", pytest.cost)
