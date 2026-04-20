import pytest

from tests.assertions import assert_skill_activated, assert_passes
from tests.harness import run_query    
    
class TestActivateContextFetcherSkill:

    @pytest.fixture
    def agent_input(self):
        return "Activate the context-fetcher skill"

    async def test_skill_activation(self, agent_input, default_agent_options):
        messages = await run_query(input=agent_input, options=default_agent_options)
        pytest.messages = messages
        assert_skill_activated(messages=messages, expected_skill="rhdp-rca-plugin:context-fetcher")
        
    @pytest.mark.dependency("test_skill_activation")
    async def test_friendly(self, agent_input):
        assertion = "The assistant was friendly and helpful"
        messages = pytest.messages
        await assert_passes(input=agent_input, messages=messages, assertion=assertion)    

    @pytest.mark.dependency("test_skill_activation")
    async def test_pirate(self, agent_input):
        assertion = "The assistant talked like a pirate"
        messages = pytest.messages
        await assert_passes(input=agent_input, messages=messages, assertion=assertion)
