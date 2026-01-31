from typing import Any


import dotenv

dotenv.load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(AgentResponse)
tools = [TavilySearch()]

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions="")

agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
exract_output = RunnableLambda(lambda x: x["output"])
chain = agent_executor | exract_output | structured_llm


def main():
    result = chain.invoke({"input": "Search for the latest news on the stock market"})
    print(result)


if __name__ == "__main__":
    main()
