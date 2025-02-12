import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_core import CancellationToken, Image
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from autogen_core.model_context import BufferedChatCompletionContext

from io import BytesIO

import requests
from autogen_core import Image as AGImage
import PIL

import pandas as pd

from typing import Literal

from pydantic import BaseModel
import os


# The response format for the agent as a Pydantic base model.
class AgentResponse(BaseModel):
    thoughts: str
    response: Literal["happy", "sad", "neutral"]


async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."


# Define a tool that searches the web for information.
async def web_search(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."


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

model_client_vision = OpenAIChatCompletionClient(
    model="llama3.2-vision",
    base_url="http://127.0.0.1:11434/v1",
    api_key="NULL",
    model_info={
        "vision": True,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
    },
)

# Create an agent that uses the OpenAI GPT-4o model with the custom response format.
model_client_formated_response = OpenAIChatCompletionClient(
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
    response_format=AgentResponse,  # type: ignore
)


model_client_gemini = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key=os.environ.get("GEMINI_API_KEY"),
)


async def formatted_assistant() -> None:
    agent = AssistantAgent(
        "assistant",
        model_client=model_client_formated_response,
        system_message="Categorize the input as happy, sad, or neutral following the JSON format.",
    )
    await Console(agent.run_stream(task="I am happy."))


async def use_buffered_context() -> None:
    # Create an agent that uses only the last 5 messages in the context to generate responses.
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[web_search],
        system_message="Use tools to solve tasks.",
        model_context=BufferedChatCompletionContext(
            buffer_size=5
        ),  # Only use the last 5 messages in the context.
    )


async def streaming_run() -> None:
    streaming_assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful assistant.",
        model_client_stream=True,  # Enable streaming tokens.
    )

    # Use an async function and asyncio.run() in a script.
    async for message in streaming_assistant.on_messages_stream(  # type: ignore
        [TextMessage(content="Name two cities in South America", source="user")],
        cancellation_token=CancellationToken(),
    ):
        print(message)


async def langchain_test() -> None:
    df = pd.read_csv(
        "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
    )
    tool = LangChainToolAdapter(PythonAstREPLTool(locals={"df": df}))
    # model_client = OpenAIChatCompletionClient(model="gpt-4o")
    agent = AssistantAgent(
        "assistant",
        tools=[tool],
        model_client=model_client,
        system_message="Use tools to solve tasks. Use the `df` variable to access the dataset.",
    )
    await Console(
        agent.on_messages_stream(
            [
                TextMessage(
                    content="What is the average Age of the passengers in the data?",
                    source="user",
                )
            ],
            CancellationToken(),
        ),
        output_stats=True,
    )


async def assistant_run() -> None:
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[web_search],
        system_message="Use tools to solve tasks.",
    )

    response = await agent.on_messages(
        [TextMessage(content="Find information on AutoGen", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(response.inner_messages)
    print("\n")
    print(response.chat_message)
    print("\n")
    print(response)


async def image_test() -> None:
    agent_vision = AssistantAgent(name="assistant", model_client=model_client_vision)

    pil_image = PIL.Image.open(
        BytesIO(requests.get("https://picsum.photos/300/200").content)
    )
    img = Image(pil_image)
    multi_modal_message = MultiModalMessage(
        content=["Can you describe the content of this image?", img], source="user"
    )
    img
    response = await agent_vision.on_messages(
        [multi_modal_message], CancellationToken()
    )
    print(response)


async def assistant_run_stream() -> None:
    # Option 1: read each message from the stream (as shown in the previous example).
    # async for message in agent.on_messages_stream(
    #     [TextMessage(content="Find information on AutoGen", source="user")],
    #     cancellation_token=CancellationToken(),
    # ):
    #     print(message)

    # Option 2: use Console to print all messages as they appear.
    await Console(
        agent.on_messages_stream(
            [TextMessage(content="Find information on AutoGen", source="user")],
            cancellation_token=CancellationToken(),
        ),
        output_stats=True,  # Enable stats printing.
    )


async def create_assistant() -> None:
    agent = AssistantAgent("assistant", model_client)
    print(await agent.run(task="Say 'Hello World!'"))


async def tool_use() -> None:
    # tool use example
    agent = AssistantAgent(
        name="weather_agent",
        model_client=model_client,
        tools=[get_weather],
        system_message="You are a helpful assistant.",
        reflect_on_tool_use=True,
    )
    await Console(agent.run_stream(task="What is the weather in New York?"))


async def multimedia_test() -> None:
    # Multimodal message example
    pil_image = Image.open(
        BytesIO(requests.get("https://picsum.photos/300/200").content)
    )
    img = AGImage(pil_image)
    multi_modal_message = MultiModalMessage(
        content=["Can you describe the content of this image?", img], source="User"
    )
    print(img)


async def multi_agent() -> None:
    # Multi agent example
    assistant = AssistantAgent("assistant", model_client)
    web_surfer = MultimodalWebSurfer("web_surfer", model_client)
    user_proxy = UserProxyAgent("user_proxy")
    termination = TextMentionTermination("exit")  # Type 'exit' to end the conversation.
    team = RoundRobinGroupChat(
        [web_surfer, assistant, user_proxy], termination_condition=termination
    )
    await Console(
        team.run_stream(
            task="Find information about AutoGen and write a short summary."
        )
    )


async def test_gemini() -> None:
    response = await model_client_gemini.create(
        [UserMessage(content="What is the capital of France?", source="user")]
    )
    print(response)


async def main() -> None:
    response = model_client.create(
        [UserMessage(content="What is the capital of France?", source="user")]
    )
    print(await response)

    # text_message = TextMessage(content="Hello, world!", source="User")
    # print(text_message)


# asyncio.run(main())
# asyncio.run(assistant_run())
# asyncio.run(image_test())
# asyncio.run(assistant_run_stream())
# asyncio.run(langchain_test())
# asyncio.run(formatted_assistant())
# asyncio.run(streaming_run())
asyncio.run(test_gemini())
# await assistant_run()
# Use asyncio.run(...) when running in a script.
