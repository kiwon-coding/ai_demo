import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler
from utils import set_openai_api_key

class StreamHandler(BaseCallbackHandler):
    def __init__(self, streamlit_container):
        self.st_container = streamlit_container
        self.text = ""
    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.st_container.write(self.text)


@st.cache_resource
def set_langchain_config():
    chat_llm = ChatOpenAI(openai_api_key=st.session_state.get("OPENAI_API_KEY"), streaming=True)

    system_template = "친절하게 대화하는 assistant"
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template)

    human_template = "{msg}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    MessagesPlaceholder(variable_name="chat_history"),
                                                    human_message_prompt])
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
    chain = LLMChain(llm=chat_llm, prompt=chat_prompt,
                     memory=memory, verbose=True)

    return chain

if __name__ == "__main__":
    set_openai_api_key()
    chain = set_langchain_config()

    # chat
    if "msg" not in st.session_state:
        st.session_state.msg = []
        
    for message in st.session_state.msg:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Say something")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
            st.session_state.msg.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            streamlit_component = StreamHandler(st.empty())
            res = chain.run(msg=prompt, callbacks=[streamlit_component])
            st.session_state.msg.append(
                {"role": "assistant", "content": res})
