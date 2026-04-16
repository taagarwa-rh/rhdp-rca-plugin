import os
import pytest

from claude_agent_sdk import ClaudeAgentOptions

from tests.assertions import assert_skill_activated, assert_passes, assert_tool_called
from tests.harness import run_query    
    
class TestContextFetcherSkillActivation:

    @pytest.fixture
    def agent_input(self):
        return "Activate the context-fetcher skill"
    
    @pytest.fixture
    def agent_options(self, default_agent_options: ClaudeAgentOptions):
        options = default_agent_options
        return options
    
    async def test_run_query(self, agent_input, agent_options):
        messages = await run_query(input=agent_input, options=agent_options)
        pytest.messages = messages
        assert hasattr(messages[-1], "result"), "No result found in the last message"
    
    @pytest.mark.dependency("test_run_query")
    async def test_skill_activation(self):
        messages = pytest.messages
        assert_skill_activated(messages=messages, expected_skill="rhdp-rca-plugin:context-fetcher")
        
    @pytest.mark.dependency("test_run_query")
    async def test_friendly(self, agent_input):
        assertion = "The assistant was friendly and helpful"
        messages = pytest.messages
        await assert_passes(input=agent_input, messages=messages, assertion=assertion)


class TestContextFetcherSkillGitHubSearch:
    
    @pytest.fixture
    def agent_input(self):
        return "Search github for taagarwa-rh's prompt optimization project and tell me what algorithms are available there."
    
    @pytest.fixture
    def agent_options(self, default_agent_options: ClaudeAgentOptions):
        options = default_agent_options
        options.mcp_servers = {
            "github": {
                "type": "http",
                "url": "https://api.githubcopilot.com/mcp",
                "headers": {
                    "Authorization": f"Bearer {os.environ["GITHUB_TOKEN"]}"
                }
            }
        }
        allowed_tools = ["mcp__github__search_code", "mcp__github__get_file_contents", "mcp__github__search_repositories"]
        options.tools.extend(allowed_tools)
        options.allowed_tools.extend(allowed_tools)
        return options
    
    async def test_run_query(self, agent_input, agent_options):
        messages = await run_query(input=agent_input, options=agent_options)
        pytest.messages = messages
        assert hasattr(messages[-1], "result"), "No result found in the last message"
    
    @pytest.mark.dependency("test_run_query")
    async def test_skill_activation(self):
        messages = pytest.messages
        assert_skill_activated(messages=messages, expected_skill="rhdp-rca-plugin:context-fetcher")

    @pytest.mark.dependency("test_run_query")
    async def test_search_repositories_called(self):
        messages = pytest.messages
        await assert_tool_called(messages=messages, expected_tool="mcp__github__search_repositories")

    @pytest.mark.dependency("test_run_query")
    async def test_all_algorithms_found(self, agent_input):
        assertion = "The response mentions APE, OPRO, PromptAgent, and ProTeGi."
        messages = pytest.messages
        await assert_passes(messages=messages, input=agent_input, assertion=assertion)
