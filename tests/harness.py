from claude_agent_sdk import (
    query, 
    ClaudeAgentOptions, 
    AssistantMessage, 
    ResultMessage, 
    SystemMessage,
    Message, 
    UserMessage, 
    ToolUseBlock, 
    ToolResultBlock, 
    TextBlock, 
    TaskProgressMessage, 
    TaskNotificationMessage,
)

async def extract_cost(messages: list[Message]):
    """Extract total cost in USD from a list of messages."""
    results = []
    for message in messages:
        if isinstance(message, ResultMessage):
            results.append(message)
    
    model_usages = [m.model_usage for m in results]
    cost_usd = sum([usage["costUSD"] for m in model_usages for usage in m.values()])
    return cost_usd
    
def track_cost(func):
    """Wrap the run_query to track the cost."""
    async def wrapper(*args, **kwargs):
        import pytest
        messages = await func(*args, **kwargs)
        cost_usd = await extract_cost(messages)
        if hasattr(pytest, "cost"):
            pytest.cost += cost_usd
        return messages
    return wrapper

@track_cost
async def run_query(input: str, options: ClaudeAgentOptions) -> list[Message]:
    """Run a query against the Claude Code CLI harness."""    
    # Run query and collect the conversation trace
    messages = []
    async for message in query(prompt=input, options=options):
        messages.append(message)
        
    return messages

def format_conversation(messages: list[Message]) -> str:
    """Format the conversation into a readable string."""
    parts = []
    for message in messages:
        if isinstance(message, AssistantMessage) and message.content:
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    parts.append(f"Tool Use: {block.name} Args: {block.input}")
                elif isinstance(block, TextBlock):
                    parts.append(f"Assistant: {block.text}")
                elif hasattr(block, "content"):
                    parts.append(f"Assistant: {block.content}")
        elif isinstance(message, SystemMessage):
            if tools := message.data.get("tools", None):
                parts.append(f"Available tools: {tools}")
            if skills := message.data.get("skills", None):
                parts.append(f"Available skills: {skills}")
            if isinstance(message, TaskProgressMessage):
                parts.append(f"System message: {message.data.get("description", None)}")
            if isinstance(message, TaskNotificationMessage):
                parts.append(f"System message: {message.data.get('summary', None)}")
        elif isinstance(message, UserMessage):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    parts.append(f"Tool Result: {block.content}")
                elif isinstance(block, TextBlock):
                    parts.append(f"User: {block.text}")
                elif hasattr(block, "content"):
                    parts.append(f"User: {block.content}")
        elif isinstance(message, ResultMessage):
            continue
    return "\n".join(parts)


