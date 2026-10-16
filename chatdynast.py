import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# ---- Page configuration (must be the first Streamlit command) ----
st.set_page_config(
    page_title="ChatDynast",
    page_icon="🪶",
    layout="centered",
)

# ---- Custom styling ----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: radial-gradient(circle at top, #1b1035 0%, #0b0b17 60%);
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Title */
    .dynast-title {
        text-align: center;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0;
        background: linear-gradient(90deg, #ff4ecd, #8b5cf6, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
    }

    /* Founder name */
    .dynast-founder {
        text-align: center;
        font-size: 1.1rem;
        color: #a5a5c0;
        margin-top: 0.2rem;
    }
    .dynast-founder .name {
        font-weight: 800;
        font-size: 1.2rem;
        color: #fcd34d;
        padding-left: 6px;
    }

    /* Divider line */
    .dynast-divider {
        height: 2px;
        margin: 14px auto 22px auto;
        width: 60%;
        background: linear-gradient(90deg, transparent, #8b5cf6, transparent);
        border-radius: 2px;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(6px);
    }

    /* Make all chat text bright and clear */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] code {
        color: #f5f5ff !important;
        font-size: 1.05rem;
        line-height: 1.65;
    }

    /* Dark bottom bar */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: #1f1f3d !important;
        border: 1px solid #6d4fd8 !important;
        border-radius: 16px !important;
    }
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] div[data-baseweb="textarea"],
    [data-testid="stChatInput"] div[data-baseweb="base-input"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #ffffff !important;
        border-radius: 0 !important;
    }
    [data-testid="stChatInputSubmitButton"] {
        background: #6d4fd8 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatInputSubmitButton"] svg {
        color: #ffffff !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #b8b8d6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header ----
st.markdown('<h1 class="dynast-title">🪶 ChatDynast</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="dynast-founder">Founded by <span class="name">Shahriar Shoaib</span></p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="dynast-divider"></div>', unsafe_allow_html=True)

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
        st.chat_message("user", avatar="🧑‍💻").write(msg.content)

    elif isinstance(msg, AIMessage):
        st.chat_message("assistant", avatar="🪶").write(msg.content)

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

    st.chat_message("user", avatar="🧑‍💻").write(prompt)

    with st.chat_message("assistant", avatar="🪶"):
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
