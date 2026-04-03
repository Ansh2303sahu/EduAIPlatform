from app.rag import retriever as rag_retriever
from app.rag.schemas import RetrievalQuery


def run_rag(query: RetrievalQuery):
    return rag_retriever.retrieve(query)
