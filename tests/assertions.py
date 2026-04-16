from claude_agent_sdk import ( 
    ClaudeAgentOptions, 
    AssistantMessage, 
    Message, 
    ToolUseBlock, 
)
from pydantic import BaseModel

from tests.harness import format_conversation, run_query

def assert_skill_activated(messages: list[Message], expected_skill: str):
    """Check if a skill was called."""
    assistant_messages = [message for message in messages if isinstance(message, AssistantMessage)]
    tool_use_blocks = [block for message in assistant_messages for block in message.content if isinstance(block, ToolUseBlock)]
    skill_use_blocks = [block for block in tool_use_blocks if block.name == "Skill" and block.input.get("skill", "") == expected_skill]
    if len(skill_use_blocks) > 0:
        assert True
    else:
        assert False, f"Skill {expected_skill} was not activated."


async def assert_tool_called(messages: list[Message], expected_tool: str):
    """Check if a tool was called."""
    assistant_messages = [message for message in messages if isinstance(message, AssistantMessage)]
    tool_use_blocks = [block for message in assistant_messages for block in message.content if isinstance(block, ToolUseBlock)]
    expected_tool_use_blocks = [block for block in tool_use_blocks if block.name == expected_tool]
    if len(expected_tool_use_blocks) > 0:
        assert True
    else:
        assert False, f"Tool {expected_tool} was not used."


class AssertionOutput(BaseModel):
    
    result: str
    reasoning: str

async def assert_passes(input: str, messages: list[Message], assertion: str):
    """Evaluate a natural language assertion for a given conversation."""
    system_prompt = (
        "Evaluate if the assertion is true for the given conversation. Use the output format:\n"
        "{\n"
        '  "result": "string",     # pass if the assertion is true, otherwise fail\n'
        '  "reasoning": "string"   # Reason for your decision\n'
        "}"
    )
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=[],
        output_format={"type": "json_schema", "schema": AssertionOutput.model_json_schema()},
        permission_mode="dontAsk",
    )
    conversation_trace = format_conversation(messages=messages)
    prompt = (
        "## Input ##\n"
        f"{input}\n\n"
        "## Messages ##\n"
        f"{conversation_trace}\n\n"
        "## Assertion ##\n"
        f"{assertion}"
    )

    assertion_messages = await run_query(input=prompt, options=options)
    last_message = assertion_messages[-1]
    output = last_message.structured_output
    if output is not None:
        output = AssertionOutput.model_validate(output)
    else:
        output = AssertionOutput(result="fail", reasoning=last_message.result)

    assert output.result == "pass", output.reasoning
