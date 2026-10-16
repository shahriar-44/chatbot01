
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# ---- Model setup (runs once per session) ----
if "model" not in st.session_state:
    st.session_state.model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.7
    )

# ---- Chat memory ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You're a sexy and Funny AI Agent.")
    ]

# ---- App title ----
st.title("ChatDynast")
st.caption("Founded by Shahriar Shoaib")

# ---- Display conversation history ----
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# ---- Chat input ----
prompt = st.chat_input("Type your message...")


# ---- Stream Gemini response ----
def stream_chunks():
    for chunk in st.session_state.model.stream(
        st.session_state.messages
    ):
        if isinstance(chunk.content, str):
            if chunk.content:
                yield chunk.content

        elif isinstance(chunk.content, list):
            for block in chunk.content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and block.get("text"):
                        yield block["text"]


# ---- Handle user message ----
if prompt:
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        try:
            reply_text = st.write_stream(stream_chunks())

            if reply_text:
                st.session_state.messages.append(
                    AIMessage(content=reply_text)
                )
            else:
                st.warning("Gemini returned an empty response.")

        except Exception:
            # Remove the failed request from conversation memory
            st.session_state.messages.pop()

            st.error(
                "Sorry, ChatDynast couldn't get a response from Gemini. "
                "Please try again in a moment. If the problem continues, "
                "check your API key, model availability, quota, and app logs."
            )
