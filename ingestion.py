import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME
os.environ["OPENAI_API_KEY"] 
os.environ["PINECONE_API_KEY"] 

def ingest_docs() -> None:
    loader = TextLoader("features.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    PineconeVectorStore.from_documents(docs, embeddings, index_name=INDEX_NAME)
    print("-> Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()
