import argparse
import asyncio
from pathlib import Path
from typing import Any, Iterable, Callable
import importlib
import inspect
from pydantic import BaseModel
import warnings
import json

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


class TestCase(BaseModel):
    
    input: str
    assertions: list[str]
    config: dict[str, Any]
    tags: list[str]
    

class AssertionOutput(BaseModel):
    
    result: str
    reasoning: str

class AssertionResult(BaseModel):
    
    assertion: str
    output: AssertionOutput
    messages: list[Message] = []
    
    
class TestResult(BaseModel):
    
    input: str
    config: dict[str, Any]
    tags: list[str]
    conversation_trace: str
    assertion_results: list[AssertionResult]
    messages: list[Message]
    
    
class TestReport(BaseModel):
    
    score: float
    costs: dict
    results: list[TestResult]
    
    def to_json_report(self) -> str:
        """Convert to a JSON report."""
        report = self.model_dump()
        failures = [
            {"input": result.input, "assertion": assertion.assertion, "result": assertion.output.result, "reasoning": assertion.output.reasoning}
            for result in self.results for assertion in result.assertion_results 
            if assertion.output.result == "fail"
        ]
        report["failures"] = failures
        return json.dumps(report, indent=2)


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


def load_function(path: str, func_name: str) -> Callable:
    try:
        spec = importlib.util.spec_from_file_location("module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name)
    except Exception as e:
        return None


async def evaluate_assertion(input: str, conversation_trace: str, assertion: str):
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
    prompt = (
        "## Input ##\n"
        f"{input}\n\n"
        "## Messages ##\n"
        f"{conversation_trace}\n\n"
        "## Assertion ##\n"
        f"{assertion}"
    )
    _messages = []
    async for message in query(prompt=prompt, options=options):
        _messages.append(message)
        if hasattr(message, "result"):
            output = message.structured_output
            if output is not None:
                output = AssertionOutput.model_validate(output)
            else:
                output = AssertionOutput(result="fail", reasoning=message.result)
    assertion_result = AssertionResult(assertion=assertion, output=output, messages=_messages)
    return assertion_result
        
async def execute_assertion(input: str, messages: list[Message], assertion_func: Callable, assertion: str, config: dict[str, Any]):
    """Execute an assertion function."""
    required_params = {"input", "messages", "config"}
    actual_params = set(inspect.signature(assertion_func).parameters.keys())
    if not required_params.issubset(actual_params):
        missing = required_params - actual_params
        raise Exception(f"Assertion function must contain required parameters {missing}")
    
    if inspect.iscoroutinefunction(assertion_func):
        output = await assertion_func(input=input, messages=messages, config=config)
    else:
        output = assertion_func(input=input, messages=messages, config=config)
    if isinstance(output, dict) and not all(key in output.keys() for key in ["result", "reasoning"]):
        raise ValueError("Assertion functions returning a dictionary must have 'result' and 'reasoning' keys.")
    output = output if isinstance(output, AssertionOutput) else AssertionOutput(result=output["result"], reasoning=output["reasoning"])
    assertion_result = AssertionResult(assertion=assertion, output=output)
    return assertion_result

async def run_query(input: str) -> list[Message]:
    """Run a query against the Claude Code CLI harness."""
    # Uses the Claude Code CLI by default
    options = ClaudeAgentOptions(
        allowed_tools=["Skill", "Read", "Bash(oc:*)", "Bash(curl:*)", "Grep", "Glob"],
        permission_mode="dontAsk",
        plugins=[{'type': 'local', 'path': str(Path(__file__).parent.parent)}]
    )
    
    # Run query and collect the conversation trace
    messages = []
    async for message in query(prompt=input, options=options):
        messages.append(message)
        
    return messages

async def test_case(case: TestCase) -> dict[str, Any]:
    """Evaluate one case."""
    # Run query and collect the conversation trace
    messages = await run_query(input=case.input)
    conversation_trace = format_conversation(messages)

    # Evaluate assertions
    assertion_results = []
    for assertion in case.assertions:
        parts = assertion.split(":", maxsplit=1)
        if len(parts) > 1 and Path(parts[0]).exists() and (assertion_func := load_function(parts[0], parts[1])):
            assertion_result = await execute_assertion(input=case.input, messages=messages, assertion=assertion, assertion_func=assertion_func, config=case.config)
        else:
            assertion_result = await evaluate_assertion(input=case.input, conversation_trace=conversation_trace, assertion=assertion)
        assertion_results.append(assertion_result)
    
    # Return assertion_results, assertion_messages, and messages
    test_result = TestResult(
        input=case.input,
        config=case.config,
        tags=case.tags,
        conversation_trace=conversation_trace, 
        assertion_results=assertion_results,
        messages=messages,
    )
    return test_result 


async def calculate_costs(test_results: Iterable[TestResult]) -> dict[str, Any]:
    """Calculate costs for the given return object."""
    all_result_messages: list[ResultMessage] = []
    for test_result in test_results:
        for message in test_result.messages:
            if isinstance(message, ResultMessage):
                all_result_messages.append(message)
        for assertion_result in test_result.assertion_results:
            for message in assertion_result.messages:
                if isinstance(message, ResultMessage):
                    all_result_messages.append(message)
    
    # Calculate total cost
    model_usages = [message.model_usage for message in all_result_messages]
    cost_usd = sum([usage["costUSD"] for m in model_usages for usage in m.values()])
    return {"costUSD": cost_usd}


async def generate_test_report(test_results: Iterable[TestResult]):
    """Generate test report."""
    costs = await calculate_costs(test_results=test_results)

    failures = 0
    successes = 0
    for test_result in test_results:
        for assertion_result in test_result.assertion_results:
            if assertion_result.output.result.lower() == "pass":
                successes += 1
            else:
                failures += 1

    score = successes / (successes + failures)
    report = TestReport(results=test_results, score=score, costs=costs)
    return report


def parse_args() -> argparse.Namespace:
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=str, help="List of tags to run as a comma separated list")
    parser.add_argument("-o", "--output", type=str, help="Output file path", default=None)
    args = parser.parse_args()
    return args
    

async def main():
    """Run evals in parallel."""
    # Parse arguments from the command line
    args = parse_args()
    tags = args.tags.split(",") if args.tags else []
    
    # Load eval cases from jsonl files
    path = Path(__file__).parent.parent / "skills"
    test_files = list(path.rglob("tests.jsonl"))
    test_cases = []
    for file in test_files:
        with open(file, "r") as f:
            cases = [TestCase.model_validate_json(line) for line in f.readlines()]
            test_cases.extend(cases)
    
    # Run eval cases in parallel
    test_results = await asyncio.gather(*[
        test_case(case=case) 
        for case in test_cases 
        if ((tags == []) or any(tag in tags for tag in case.tags))
    ])

    # Parse and save results
    test_report = await generate_test_report(test_results)
    if args.output is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(args.output, "w") as f:
                f.write(test_report.to_json_report())
        
    # Print the score to stdout
    score = test_report.score
    print(f"{score:.2f}")
    

if __name__ == "__main__":
    asyncio.run(main())