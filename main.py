import dotenv

dotenv.load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [TavilySearch()]

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
exract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent | exract_output | parse_output


def main():
    result = chain.invoke({"input": "Search for the latest news on the stock market"})
    print(result)


if __name__ == "__main__":
    main()
