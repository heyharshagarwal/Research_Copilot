from ingestion.vector_store.store import ChromaVectorStore

class Retriever:
    """
    A retriever that wraps vector store to provide a standard `retrieve()` interface.
    """
    def __init__(self, persist_dir="chroma_store", collection_name="documents"):
        self.vector_store = ChromaVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name
        )

    def retrieve(self, query, top_k=3, score_threshold=0.0):
        docs = self.vector_store.query(query, top_k=top_k)
        filtered_docs = [d for d in docs if d['distance'] <= score_threshold]
        return filtered_docs