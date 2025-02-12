import asyncio
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
import os


async def setup_code_generation_and_execution():
    # 1. Set up the model client
    model_client = OpenAIChatCompletionClient(
        model="qwen2.5-coder:32b",  # Replace with your model
        base_url="http://127.0.0.1:11434/v1",  # Replace with your endpoint
        api_key="NULL",
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
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "gemini-2.0-flash",
        },
    )

    # 2. Create the code generator agent (AssistantAgent)
    code_execution = PythonCodeExecutionTool(
        LocalCommandLineCodeExecutor(work_dir="coding")
    )
    code_generator = AssistantAgent(
        name="coder",
        model_client=model_client_gemini,
        tools=[code_execution],
        reflect_on_tool_use=True,
        system_message="""You are a data analysis expert and Python programmer. Generate and run code to solve the task. You have tools available to execute Python code.""",
    )

    return code_generator


async def run_code_generation_and_execution(task: str):
    # Set up the agents
    code_executor = await setup_code_generation_and_execution()

    # Run the team chat with the given task
    print(f"Starting task: {task}")
    await Console(
        code_executor.run_stream(task=task, cancellation_token=CancellationToken())
    )


# Example usage
async def main():
    # Example data analysis task
    task = """Analyze the 'titanic.csv' dataset:
    1. Calculate the average age of passengers
    2. Include the count of passengers with recorded ages
    3. Show the minimum and maximum ages
    4. Handle missing values appropriately
    
    The dataset is available at: https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"""

    await run_code_generation_and_execution(task)


if __name__ == "__main__":
    asyncio.run(main())
