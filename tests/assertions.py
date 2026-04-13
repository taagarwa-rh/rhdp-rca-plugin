from claude_agent_sdk import Message, AssistantMessage, ToolUseBlock


def skill_activated(input: str, messages: list[Message], config: dict):
    """Check if a skill was called."""
    expected_skill = config["expected_skill"]
    
    assistant_messages = [message for message in messages if isinstance(message, AssistantMessage)]
    tool_use_blocks = [block for message in assistant_messages for block in message.content if isinstance(block, ToolUseBlock)]
    skill_use_blocks = [block for block in tool_use_blocks if block.name == "Skill" and block.input.get("skill", "") == expected_skill]
    if len(skill_use_blocks) > 0:
        return {
            "result": "pass",
            "reasoning": f"Skill {expected_skill} was activated."
        }
    return {
        "result": "fail",
        "reasoning": f"Skill {expected_skill} was not activated."
    }
