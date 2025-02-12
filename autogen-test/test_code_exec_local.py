import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
import os

model_client_gemini = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key=os.environ.get("GEMINI_API_KEY"),
    model_info={
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-2.0-flash",
    },
)


async def main() -> None:
    tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))
    agent = AssistantAgent(
        "assistant",
        model_client_gemini,
        tools=[tool],
        reflect_on_tool_use=True,
    )
    await Console(
        agent.run_stream(
            task="Create a plot of MSFT stock prices in 2024 and save it to a file. Use yfinance and matplotlib. Generate and run code to solve the task."
        )
    )


asyncio.run(main())
