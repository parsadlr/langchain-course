import dotenv

dotenv.load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_react_agent

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [TavilySearch()]
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True   )


def main():
    result = agent_executor.invoke({"input": "Search for the latest news on the stock market"})
    print(result)


if __name__ == "__main__":
    main()
