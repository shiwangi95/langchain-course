from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


llm = ChatOpenAI(temperature=0, model="gpt-5.5")
tools = [TavilySearch()]
# if you don't put model param in agent creation, a default model name will be put automatically
agent = create_agent(model=llm, tools=tools)


def main():
    query = "How is the weather in Tokyo?"
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(result)


if __name__ == "__main__":
    main()
