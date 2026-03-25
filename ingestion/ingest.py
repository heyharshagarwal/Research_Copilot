from ingestion.dataLoader.dataLoader import load_all_documents
from ingestion.vector_store.store import ChromaVectorStore

def run_ingestion():
    DATA_DIR = "data"

   
    documents = load_all_documents(DATA_DIR)

    if not documents:
        print("No documents found. Exiting.")
        return

    

    print("Initializing vector store...")
    vector_store = ChromaVectorStore(
        persist_dir="chroma_store",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
        collection_name="documents"
    )

    print("Building vector store...")
    vector_store.build_from_documents(documents)

    print("Ingestion complete!")

if __name__ == "__main__":
    run_ingestion()
