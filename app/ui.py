import streamlit as st
import os
import uuid
from agents.research_agent import agent_with_chat_history
from ingestion.dataLoader.dataLoader import load_all_documents
from ingestion.vector_store.store import ChromaVectorStore


DATA_DIR = "data"

def save_uploaded_file(uploaded_file):
    """Saves an uploaded file to the data directory."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def run_streamlit_app():
    st.set_page_config(page_title="AI Research Copilot", page_icon="🚀", layout="wide")

    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    
    with st.sidebar:
        st.title("📁 Document Center")
        
   
        uploaded_files = st.file_uploader(
            "Upload research papers (PDF)", 
            accept_multiple_files=True
        )
        
        if st.button("Process & Index Documents"):
            if uploaded_files:
                with st.spinner("Saving and indexing..."):
               
                    for uploaded_file in uploaded_files:
                        save_uploaded_file(uploaded_file)
                    
                  
                    documents = load_all_documents(DATA_DIR)
                    if documents:
                        vector_store = ChromaVectorStore(
                            persist_dir="chroma_store",
                            embedding_model="all-MiniLM-L6-v2",
                            collection_name="documents"
                        )
                        vector_store.build_from_documents(documents)
                        st.success(f"Indexed {len(documents)} document chunks!")
                    else:
                        st.error("Processing failed: No documents found.")
            else:
                st.warning("Please upload at least one file first.")

        st.divider()
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    
    st.title("AI Research Copilot")
    st.caption("Context-aware AI powered by your documents")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

  
    if prompt := st.chat_input("Ask a question about your documents..."):
        
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

     
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                config = {"configurable": {"session_id": st.session_state.session_id}}
                try:
                    response = agent_with_chat_history.invoke({"input": prompt}, config=config)
                    answer = response['output']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    run_streamlit_app()