from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent response"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatOpenAI(temperature=0, model="gpt-5.5")
tools = [TavilySearch()]
# if you don't put model param in agent creation, a default model name will be put automatically
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    query = "How is the weather in Tokyo?"
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(f"Entire response: {result}")
    # This field(structured_response) will only arrive if we add response format in agent creation
    print(f"Answer: {result.get("structured_response").answer}")
    print(f"Sources: {result.get("structured_response").sources}") 


if __name__ == "__main__":
    main()
