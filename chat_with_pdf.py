import streamlit as st
import os
from openai import OpenAI
from os import environ

from rag import add_files, answer

client = OpenAI(
	api_key=os.environ["API_KEY"],
	base_url="https://api.ai.it.cornell.edu",
)

st.set_page_config(
    page_title="INFO5940 RAG Chat",
    page_icon="💬",
    layout="wide",
)
st.title("📝 File Q&A with Retrieval")

st.markdown(
    """
    Upload one or more .txt / .pdf files.  
    Ask questions in chat.  
    Answers come only from the uploaded files.
    """
)
st.sidebar.header("1. Upload documents")

uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)

if "indexed_docs" not in st.session_state:
    st.session_state["indexed_docs"] = []

if st.sidebar.button("Add to knowledge base"):
    if not uploaded_files:
        st.sidebar.warning("No files selected.")
    else:
        new_docs = add_files(uploaded_files)
        if len(new_docs) == 0:
            st.sidebar.error("No readable text found in those files.")
        else:
            st.session_state.indexed_docs.extend(new_docs)
            st.sidebar.success("Indexing Success")

st.sidebar.subheader("Indexed so far")
if len(st.session_state.indexed_docs) == 0:
    st.sidebar.write("_none yet_")
else:
    for info in st.session_state.indexed_docs:
        st.sidebar.write(f"- {info['filename']}")


st.subheader("2. Chat with your documents")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Ask a question about the uploaded documents. I will try my best to answer!",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.write(msg["content"])


user_question = st.chat_input(
    "Ask a question...",
    disabled=len(st.session_state.indexed_docs) == 0,
)


if user_question:
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )
    with st.chat_message("user"):
        st.write(user_question)

    bot_reply = answer(
        question=user_question,
        history=st.session_state.messages,
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )
    with st.chat_message("assistant"):
        st.write(bot_reply)

st.markdown("---")
st.caption(
    "Behind the scenes: files are chunked, embedded, stored in Chroma, and "
    "retrieved to answer each question."
)