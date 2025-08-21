from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from langchain.memory import ConversationBufferMemory



import os
from dotenv import load_dotenv
load_dotenv()

api_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki=WikipediaQueryRun(api_wrapper=api_wiki)

api_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv=ArxivQueryRun(api_wrapper=api_arxiv)

search=DuckDuckGoSearchRun(name="Search")


st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your groq ai api key",type="password")

st.title("LangChain -Chat with Search")


if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi,I am a chatbot who can search the web.How can i help you ?"}
    ]

# Initialize LangChain memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(api_key=api_key,model="Llama3-8b-8192",streaming=True)
    tools=[search,wiki,arxiv]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    search_agent=initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=st.session_state.memory,
        handle_parsing_errors=True,
        verbose=True
    )
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.invoke({"input":prompt},callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response["output"]})
        st.write(response["output"])
