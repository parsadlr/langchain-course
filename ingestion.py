import os 
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def main():

    print("Loading data...")
    loader = TextLoader("mediumblog1.txt")
    document = loader.load()

    print("Splitting data...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(document)

    print("Ingesting data...")
    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))
    print("Data ingested successfully!")


if __name__ == "__main__":
    main()
