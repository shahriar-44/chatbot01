
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# ---- Model setup (runs once per session, not every rerun) ----
if "model" not in st.session_state:
    st.session_state.model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.7
    )

# ---- Memory (Streamlit reruns the whole script every time you type,
# so session_state is what keeps your messages list alive) ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You're a sexy and Funny AI Agent.")
    ]

st.title("ChatDynast")
st.caption("Founded by Shahriar Shoaib")

# ---- Show past messages (skip SystemMessage, user doesn't need to see it) ----
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# ---- Chat input box (built-in Streamlit widget, sits at bottom) ----
prompt = st.chat_input("Type your message...")

def stream_chunks():
    # model.stream() returns chunks one at a time instead of the full reply at once
    for chunk in st.session_state.model.stream(st.session_state.messages):
        # each chunk is same shape as before: [{"type": "text", "text": "..."}]
        if chunk.content:
            yield chunk.content[0]["text"]

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # write_stream prints each yielded piece live AND returns the full joined text
        reply_text = st.write_stream(stream_chunks())

    st.session_state.messages.append(AIMessage(content=reply_text))
