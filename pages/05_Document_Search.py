import streamlit as st
import os
from utils import set_openai_api_key

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def file_selector():
    folder_path = "data"
    files = [f for f in os.listdir(
        folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".pdf")]
    selected_filename = st.selectbox('Select a file', files)
    return os.path.join(folder_path, selected_filename)

if __name__ == "__main__":
    set_openai_api_key()
    filepath = file_selector()
    
    if filepath:
        filename_full = filepath
        filename = filename_full.split('.')[0]
        
        with st.spinner('Loading...'):
            db_path = os.path.join("db", filename.split('/')[-1])
            try:
                db = FAISS.load_local(db_path, HuggingFaceEmbeddings())
            except RuntimeError:
                # Step 1: LOAD: document loading (txt, web, pdf)
                raw_documents = PyPDFLoader(filename_full).load_and_split()
                
                # Step 2: SPLIT: split a document into multiple chunk
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=0)
                documents = text_splitter.split_documents(raw_documents)

                # Step 3: STORE: embedding each chunk and store into vector DB
                db = FAISS.from_documents(documents, HuggingFaceEmbeddings())
                db.save_local(db_path)

        # Step 4: RETRIEVE & Generate
        question = st.text_input("Enter your question about the file.")
        if question:
            llm = ChatOpenAI(openai_api_key=st.session_state.get("OPENAI_API_KEY"))
            qa_chain = RetrievalQA.from_chain_type(
                llm, retriever=db.as_retriever())
            res = qa_chain({"query": question})
            st.write(res)
