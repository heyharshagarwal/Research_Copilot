import os
from typing import List, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from ingestion.embedding.embedding import EmbeddingPipeline
import uuid

class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str = "chroma_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "documents"
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


       

        self.client = chromadb.PersistentClient(path=self.persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        print(f"[INFO] Loaded embedding model: {embedding_model}")
        print(f"[INFO] Using ChromaDB collection: {collection_name}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} documents...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [{"text": text} for text in texts]
        ids = [str(uuid.uuid4()) for _ in texts]

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[INFO] Stored {len(texts)} chunks in ChromaDB")

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying for: '{query_text}'")

        query_embedding = self.model.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i]
            })

        return formatted_results