
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# ---- Page configuration ----
st.set_page_config(
    page_title="ChatDynast | AI Chat",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---- Custom design ----
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(145deg, #0b1020 0%, #11152b 55%, #17112c 100%);
    color: #f1f5f9;
}

.block-container {
    max-width: 850px;
    padding-top: 2.5rem;
    padding-bottom: 7rem;
}

/* Header */
.hero {
    text-align: center;
    padding: 25px 15px 30px;
    margin-bottom: 28px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        rgba(99, 102, 241, 0.14),
        rgba(168, 85, 247, 0.09)
    );
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.18);
}

.logo {
    font-size: 35px;
    margin-bottom: 4px;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0;
    background: linear-gradient(90deg, #a5b4fc, #c084fc, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.tagline {
    color: #cbd5e1;
    font-size: 15px;
    margin-top: 10px;
}

.founder {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 15px;
}

.status {
    display: inline-block;
    margin-top: 16px;
    padding: 6px 12px;
    border-radius: 30px;
    background: rgba(34, 197, 94, 0.10);
    border: 1px solid rgba(34, 197, 94, 0.25);
    color: #86efac;
    font-size: 12px;
}

/* Chat message containers */
[data-testid="stChatMessage"] {
    background: rgba(30, 41, 59, 0.62);
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 14px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.10);
}

/* Chat input */
[data-testid="stChatInput"] {
    border: 1px solid rgba(167, 139, 250, 0.4);
    border-radius: 18px;
    background: #151a30;
    box-shadow: 0 0 22px rgba(139, 92, 246, 0.08);
}

[data-testid="stChatInput"] textarea {
    color: #f8fafc !important;
}

/* Buttons */
.stButton button {
    border-radius: 12px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 20px 0 5px;
}

/* Hide default decoration */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="hero">
    <div class="logo">✦</div>
    <h1>ChatDynast</h1>
    <div class="tagline">A little intelligence. A lot of personality.</div>
    <div class="status">● AI ASSISTANT ONLINE</div>
    <div class="founder">Crafted with care by <b>Shahriar Shoaib</b></div>
</div>
""", unsafe_allow_html=True)

# ---- Model setup ----
if "model" not in st.session_state:
    st.session_state.model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        max_retries=2
    )

# ---- Chat memory ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content="You're a sexy and Funny AI Agent."
        )
    ]

# ---- Optional welcome section ----
if not any(
    isinstance(msg, HumanMessage)
    for msg in st.session_state.messages
):
    st.markdown("""
    <div style="
        text-align:center;
        padding: 10px 0 25px;
        color:#cbd5e1;
    ">
        <div style="font-size:19px;font-weight:600;">
            Hey there! 👋
        </div>
        <div style="font-size:14px;color:#94a3b8;margin-top:7px;">
            What's on your mind today?
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---- Display conversation history ----
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="✦"):
            st.markdown(msg.content)

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

# ---- Chat input ----
prompt = st.chat_input("Message ChatDynast...")

if prompt:
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✦"):
        try:
            reply_text = st.write_stream(stream_chunks())

            if reply_text:
                st.session_state.messages.append(
                    AIMessage(content=reply_text)
                )
            else:
                st.warning("The AI returned an empty response.")

        except Exception as e:
            print(f"Groq error: {type(e).__name__}: {e}")

            if (
                st.session_state.messages
                and isinstance(st.session_state.messages[-1], HumanMessage)
                and st.session_state.messages[-1].content == prompt
            ):
                st.session_state.messages.pop()

            st.error(
                "Oops! ChatDynast couldn't connect right now. "
                "Please try again in a moment."
            )

# ---- Footer ----
st.markdown("""
<div class="footer">
    ChatDynast ✦ Powered by Groq · Made by Shahriar Shoaib
</div>
""", unsafe_allow_html=True)
