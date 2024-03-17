import os
from typing import Any

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone

from consts import INDEX_NAME

os.environ["OPENAI_API_KEY"] 
os.environ["PINECONE_API_KEY"] 


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})


if __name__ == "__main__":
    # print(run_llm(query="Quero que recomende features para um modelo que vou criar que prediz casa. Apenas traga uma lista de features já plugada em modelo com o mesmo contexto"))
    # As features recomendadas para um modelo de predição de casa são: ["preco", "cor", "quantidade_quartos"]
    
    # print(run_llm(query="Quero que recomende features para um modelo que vou criar que prediz se alguem sobreviu ou não ao titanic. Apenas traga uma lista de features já plugada em modelo com o mesmo contexto"))
    # As features recomendadas para um modelo de predição de sobrevivência no Titanic são ["classe", "idade"]

    print(run_llm(query="Quero que recomende features para um modelo que vou criar que prediz sobre mudanças. Apenas traga uma lista de features já plugada em modelo com o mesmo contexto"))
    # 'result': 'Desculpe, não tenho informações suficientes para recomendar features para um modelo de predição sobre mudanças com base nos contextos fornecidos.'