import os
from re import X
from dotenv import load_dotenv
load_dotenv()

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

print("Initializing Components...")
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)
# gpt 3.5
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:
    {context}
    Question: {question}
    Answer:
    """
)

def format_docs(docs):
    """Format the documents to a string for the model"""
    return "\n\n".join([doc.page_content for doc in docs])


def retrieval_chain_without_lcel(query: str):
    """Simple retrieval chain without LLM chaining
    Manually retrieve the documents and format them for the model
    """
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content


def create_retrieval_chain_with_lcel():
    """Create a retrieval chain with LLM chaining"""
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template | llm | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":

    # ================================
    # Option 0: without context
    # ================================
    print("\n" + "="* 70)
    print("Option 0: without context")
    print("\n" + "="* 70)
    query = "What is Pinecone in machine learning?"
    results_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(results_raw.content)

    # ================================
    # Option 1: with context but without Lancgchain expression language
    # ================================
    print("\n" + "="* 70)
    print("Option 1: with context but without Lancgchain expression language")
    print("\n" + "="* 70)
    query = "What is Pinecone in machine learning?"
    results_raw = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(results_raw)


    # ================================
    # Option 2: with context and with Lancgchain expression language
    # ================================
    print("\n" + "="* 70)
    print("Option 2: with context and with Lancgchain expression language")
    print("\n" + "="* 70)
    chain = create_retrieval_chain_with_lcel()
    query = "What is Pinecone in machine learning?"
    results_raw = chain.invoke({"question": query})
    print("\nAnswer:")
    print(results_raw)