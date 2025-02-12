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

    # 2. Create the code generator agent (AssistantAgent)
    tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))
    code_generator = AssistantAgent(
        name="code_generator",
        model_client=model_client,
        tools=[tool],
        reflect_on_tool_use=True,
        system_message="""You are a data analysis expert and Python programmer. Follow these guidelines:
    1. When analyzing data, first inspect the data structure and content
    2. Always handle potential errors (file not found, empty data, missing columns)
    3. Use pandas for data manipulation and analysis
    4. Include helpful comments explaining your code
    5. Print descriptive messages with the results
    6. Always wrap your code in ```python code blocks
    7. For numerical analysis, handle potential NaN values and invalid data types
    8. When calculating statistics, include relevant context (count, min, max along with average)
    
    Example output format:
    ```python
    # Read and validate the data
    import pandas as pd
    
    try:
        df = pd.read_csv('data.csv')
        print(f"Successfully loaded data with {len(df)} rows")
        
        # Display basic information
        print("\nData Overview:")
        print(df.info())
        
        # Perform analysis
        result = df['column'].mean()
        print(f"\nAnalysis Result: {result}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
    ```""",
    )

    # 3. Set up the code executor with Docker

    # code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
    # await code_executor.start()

    # # 4. Create the code executor agent
    # code_runner = CodeExecutorAgent(
    #     name="code_runner",
    #     code_executor=code_executor,
    #     sources=["code_generator"],  # Only execute code from the code generator
    # )

    # 5. Create a group chat for the agents to collaborate
    # termination = TextMentionTermination("TERMINATE")
    # team = RoundRobinGroupChat(
    #     participants=[code_generator, code_runner],
    #     max_turns=3,  # Limit the conversation rounds
    #     termination_condition=termination,
    # )

    # return team, code_executor

    return code_generator


async def run_code_generation_and_execution(task: str):
    # Set up the agents
    code_executor = await setup_code_generation_and_execution()

    try:
        # Run the team chat with the given task
        print(f"Starting task: {task}")
        await Console(
            code_executor.run_stream(task=task, cancellation_token=CancellationToken())
        )
    finally:
        # Clean up: Stop the code executor
        await code_executor.stop()


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
