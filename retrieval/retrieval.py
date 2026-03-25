from config.llm import llm
from retrieval.retriever.retriever import Retriever
from retrieval.generator.generator import RAGPipeline

retriever = Retriever(persist_dir="chroma_store", collection_name="documents")

rag = RAGPipeline(retriever, llm)
result = rag.query(
    "Evolution of AI definition?",
    top_k=3,
    min_score=0.1,
    stream=True,
    summarize=True
)
print("\nFinal Answer:\n", result['answer'])
print("Summary:\n", result['summary'])
print("History:\n", result['history'][-1])