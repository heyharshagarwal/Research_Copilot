import os
from langchain.tools import tool
from langchain_tavily import TavilySearch
from retrieval.retriever.retriever import Retriever
from dotenv import load_dotenv

load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

retriever_obj = Retriever(persist_dir="chroma_store", collection_name="documents")

@tool
def calc(expression: str) -> str:
    """Evaluate simple mathematical expressions (e.g., '2 + 2' or '5 * (10/2)')."""

    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def search_documents(query: str):
    """
    Look up specific information, facts, or technical details from the 
    uploaded research papers and PDF documents. Use this for deep internal knowledge.
    """
    results = retriever_obj.retrieve(query, top_k=4) 
    
    formatted_results = []
    for doc in results:
        source = doc["metadata"].get("source", "Unknown")
        page = doc["metadata"].get("page", "N/A")
        content = f"\n---\nSource: {source} (Page {page})\nContent: {doc['document']}"
        formatted_results.append(content)
        
    return "\n".join(formatted_results)

tavily_tool = TavilySearch(max_results=5)


tools = [search_documents, tavily_tool, calc]