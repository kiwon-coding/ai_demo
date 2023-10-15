import os
import pandas as pd
import streamlit as st
import json

from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

@st.cache_data
def load_review_data():
    json_file_path = './data/amazon_fashion_5.json'
    with open(json_file_path, 'r') as f:
        json_data = [json.loads(line) for line in f]

    return pd.DataFrame(json_data)

def prepare():
    df = load_review_data()
    product_list = df['asin'].unique()
    for p_id in product_list:
        print(p_id)
        chunk = df[df['asin'] == p_id]['reviewText']
        review_list = chunk.tolist()
        reviews = '\n'.join(review_list)

        text_splitter = CharacterTextSplitter(separator = "\n", chunk_size = 1000)
        documents = text_splitter.create_documents([reviews])

        db = FAISS.from_documents(documents, HuggingFaceEmbeddings())
        print("saving...")
        db.save_local(f"db/{p_id}")
        print("==============")

@st.cache_resource
def set_openapi_config(openai_api_key):
    llm = ChatOpenAI(openai_api_key = openai_api_key)
    return llm

def get_answer(llm, product_id, question):
    selected_db = FAISS.load_local(f"db/{product_id}", HuggingFaceEmbeddings())
    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=selected_db.as_retriever(), return_source_documents=True)
    res = qa_chain({"query": question})
    return res["result"], res["source_documents"]

if __name__ == "__main__":
    product_id = '7106116521'
    if not os.path.exists(f'db/{product_id}/index.pkl'):
        st.write("preparing..")
        prepare()

    api_key_input = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY", ""))
    api_key_button = st.button("Set OpenAI Key")
    if api_key_button:
        st.session_state["OPENAI_API_KEY"] = api_key_input

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    if openai_api_key:
        llm = set_openapi_config(openai_api_key)

        df = load_review_data()
        product_list = df['asin'].unique()
        print(product_list)
        product_id = st.selectbox("Select a product", product_list)
        if product_id:
            question = st.text_input("Ask any question about the selected product.")
            answer, source = get_answer(llm, product_id, question)
            st.write(answer)
            st.write(source)
