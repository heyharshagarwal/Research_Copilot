import os
from typing import List, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from ingestion.embedding.embedding import EmbeddingPipeline
import uuid
from pathlib import Path

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

        ids, metadatas, documents_text, embeddings_list = [], [], [], []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            
      
            metadata = dict(chunk.metadata).copy()
            
        
            metadata.update({
                "chunk_id": doc_id,
                "doc_index": i,
                "filename": Path(metadata.get("source", "unknown")).name,
                "paper_title": metadata.get("title") or "Untitled Research Paper",
                "author": metadata.get("author") or "Unknown Author",
                "page": metadata.get("page", 0) + 1 
            })
            
      
            ids.append(doc_id)
            metadatas.append(metadata)
            documents_text.append(chunk.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(chunks)} chunks.")
        except Exception as e:
            print(f"Error adding to Chroma: {e}")
            
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