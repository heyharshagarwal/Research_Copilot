from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF
    """

    data_path = Path(data_dir).resolve()
    documents = []

    pdf_files = list(data_path.glob('**/*.pdf'))
   
    for pdf_file in pdf_files:
      
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            loaded = loader.load()
           
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    return documents

