import asyncio
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination


async def setup_code_generation_and_execution():
    # 1. Set up the model client
    model_client = OpenAIChatCompletionClient(
        model="qwq:latest",  # Replace with your model
        base_url="http://127.0.0.1:11434/v1",  # Replace with your endpoint
        api_key="NULL",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
        },
    )

    # 2. Create the code generator agent (AssistantAgent)
    code_generator = AssistantAgent(
        name="code_generator",
        model_client=model_client,
        system_message="You are a Python programmer. When asked to solve a problem, think step by step and always provide your solution as a Python script within ```python code blocks. Add a blank line at the start of the block",
    )

    # 3. Set up the code executor with Docker
    code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
    await code_executor.start()

    # 4. Create the code executor agent
    code_runner = CodeExecutorAgent(
        name="code_runner",
        code_executor=code_executor,
        sources=["code_generator"],  # Only execute code from the code generator
    )

    # 5. Create a group chat for the agents to collaborate
    termination = TextMentionTermination("TERMINATE")  # Terminate the conversation
    team = RoundRobinGroupChat(
        participants=[code_generator, code_runner],
        max_turns=3,
        termination_condition=termination,  # Limit the conversation rounds
    )

    return team, code_executor


async def run_code_generation_and_execution(task: str):
    # Set up the agents
    team, code_executor = await setup_code_generation_and_execution()

    try:
        # Run the team chat with the given task
        print(f"Starting task: {task}")
        await Console(
            team.run_stream(task=task, cancellation_token=CancellationToken())
        )
    finally:
        # Clean up: Stop the code executor
        await code_executor.stop()


# Example usage
async def main():
    # Example task that requires generating and executing code
    # task = "Write a Python script that creates a list of first 5 Fibonacci numbers and prints them"
    tasks = [
        "Create a Python script that calculates the prime numbers up to 20",
        "Write code to read a CSV file and calculate the average of a column",
        "Generate a simple plot using matplotlib",
        "Write a Python script that creates a list of first 5 Fibonacci numbers and prints them",
    ]
    for task in tasks:
        await run_code_generation_and_execution(task)


if __name__ == "__main__":
    asyncio.run(main())
