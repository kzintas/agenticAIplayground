import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os

# Create an OpenAI model client.
model_client = OpenAIChatCompletionClient(
    model="qwq:latest",
    base_url="http://127.0.0.1:11434/v1",
    api_key="NULL",
    # parallel_tool_calls=False,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
    },
)

model_client_gemini = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# Create the primary agent.
primary_agent = AssistantAgent(
    "primary",
    model_client=model_client_gemini,
    system_message="You are a helpful AI assistant.",
)

# Create the critic agent.
critic_agent = AssistantAgent(
    "critic",
    model_client=model_client_gemini,
    system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat(
    [primary_agent, critic_agent], termination_condition=text_termination
)


async def result() -> None:
    result = await team.run(task="Write a short poem about the fall season.")
    print(result)


async def stream_result() -> None:
    # When running inside a script, use a async main function and call it from `asyncio.run(...)`.
    # await team.reset()  # Reset the team for a new task.
    async for message in team.run_stream(task="Write a short poem about the fall season."):  # type: ignore
        if isinstance(message, TaskResult):
            print("\n=== Task Completed ===")
            print(f"Stop Reason: {message.stop_reason}")
            print("==================\n")
        else:
            print(f"\n[{message.source}]:")
            print(f"{message.content}")
            print("-" * 50)


asyncio.run(stream_result())
