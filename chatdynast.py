
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# ---- App configuration ----
st.title("ChatDynast")
st.caption("Founded by Shahriar Shoaib")

# ---- Model setup ----
if "model" not in st.session_state:
    st.session_state.model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        max_retries=2,
    )

# ---- Chat memory ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content="You're a sexy and Funny AI Agent."
        )
    ]

# ---- Display conversation history ----
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# ---- Chat input ----
prompt = st.chat_input("Type your message...")


# ---- Stream Groq response ----
def stream_chunks():
    for chunk in st.session_state.model.stream(
        st.session_state.messages
    ):
        content = chunk.content

        if isinstance(content, str):
            if content:
                yield content

        elif isinstance(content, list):
            for block in content:
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
                st.warning("Groq returned an empty response.")

        except Exception as e:
            print(f"Groq error: {type(e).__name__}: {e}")

            # Remove only the failed user message
            if (
                st.session_state.messages
                and isinstance(st.session_state.messages[-1], HumanMessage)
                and st.session_state.messages[-1].content == prompt
            ):
                st.session_state.messages.pop()

            st.error(
                "ChatDynast couldn't get a response from Groq. "
                "Please check your API key, model availability, "
                "rate limits, and app logs."
            )
