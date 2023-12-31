import pandas as pd
import json
import os
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

import ast
import streamlit as st
from utils import set_openai_api_key

@st.cache_data
def load_review_data():
    json_file_path = './data/amazon_fashion_20.json'
    with open(json_file_path, 'r') as f:
        json_data = [json.loads(line) for line in f]

    return pd.DataFrame(json_data)
    
def parse_string_to_list(string_data):
    try:
        parsed_list = ast.literal_eval(string_data)
        return parsed_list
    except Exception as e:
        print(string_data)
        print("Error parsing string:", e)
        return []

@st.cache_resource
def set_openai_config():
    chat = ChatOpenAI(openai_api_key=st.session_state.get("OPENAI_API_KEY"))

    template = "A tagging system that creates tags for use in an online shopping mall."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Create up to 5 tags for the given review. The result should be an python style array of strings: ```{text}```"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    return chain


def get_taggings(review_text, chain):
    res = chain.run(text = review_text)
    tags = parse_string_to_list(res)
    return tags

@st.cache_data
def load_review_tags(file_path):
    review_tags = pd.read_csv(file_path)
    return review_tags

def show_reviews(item):
    for i, row in item.iterrows():
        st.write("Review No: ", i)
        st.write(f"**Overall Score:** {row['overall']}")
        st.write(f"**Product ID:** {row['asin']}")
        st.write(f"**Reviewer Name:** {row['reviewerName']}")
        st.write(f"**Review Text:**")
        st.write(f"{row['reviewText']}")
        st.write(f"**Tags:** {row['tags']}")
        st.markdown('<hr>', unsafe_allow_html=True)

if __name__ == '__main__':
    set_openai_api_key()
    chain = set_openai_config()
    reviews = load_review_data()
    
    tag_file_path = './data/amazon_fashion_20_tags.csv'
    if not os.path.exists(tag_file_path):
        with st.spinner("generating tags.."):
            reviews['tags'] = reviews.apply(lambda x: get_taggings(x['reviewText'], chain), axis=1)
        reviews.to_csv(tag_file_path, index=False)
    else:
        with st.spinner("loading existing tags.."):
            review_tags = load_review_tags(tag_file_path)

        # creating a dictionary of tags and select the major tags
        all_tags = {}
        tag_column_df = review_tags['tags'].apply(ast.literal_eval)
        for tags in tag_column_df:
            for tag in tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1
        
        sorted_tags = dict(sorted(all_tags.items(), key=lambda item: item[1], reverse=True))
        major_keywords = list(sorted_tags.keys())[:10]
        selected_tags = st.multiselect(
            'Select tags to filter reviews', major_keywords)
        
        # show reviews containing selected tags
        # 1) find matching reviews (containing at least one tag in its tags)
        # 2) show the reviews
        if len(selected_tags) > 0:
            selected_reviews = review_tags[review_tags['tags'].apply(lambda x: all(tag in x for tag in selected_tags))]
            show_reviews(selected_reviews)