import streamlit as st
import os
# from utils import set_openai_api_key

from langchain.document_loaders import PyPDFLoader

def file_selector(folder_path='.'):
    st.write("file_selector()")

    st.write(folder_path)
    for f in os.listdir(folder_path):
        st.write(f)

    st.write("done")

    files = [f for f in os.listdir(
        folder_path) if os.path.isfile(f) and f.endswith(".pdf")]
    st.write(files)
    selected_filename = st.selectbox('Select a file', files)
    return os.path.join(folder_path, selected_filename)

if __name__ == "__main__":
    # set_openai_api_key()
    filepath = file_selector('data')
    st.write(filepath)

    if filepath:
        filename_full = filepath
        filename = filename_full.split('.')[0]

        with st.spinner('Loading...'):
            # Step 1: LOAD: document loading (txt, web, pdf)
            raw_documents = PyPDFLoader(filename_full).load_and_split()
