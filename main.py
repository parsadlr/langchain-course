from pyexpat import model
import dotenv

dotenv.load_dotenv()

from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

from schemas import AgentResponse

tools = [TavilySearch()]
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_agent(
    model=model,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "Search for the latest news on the stock market"}
        ]
    })

    print(result["structured_response"])


if __name__ == "__main__":
    main()
