import os
from datetime import date
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

    /* Access code form + alerts */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
    }
    [data-testid="stForm"] label p,
    [data-testid="stAlert"] *,
    [data-testid="stCaptionContainer"] * {
        color: #f5f5ff !important;
    }
    [data-testid="stAlert"] {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
    }
    [data-testid="stTextInput"] input {
        background: #1f1f3d !important;
        color: #ffffff !important;
        border: 1px solid #6d4fd8 !important;
    }
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #8b5cf6, #38bdf8);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        font-weight: 600;
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

# ---- Usage limits (protects your free API key) ----
FREE_MESSAGES = 3      # free messages per visitor
DAILY_CAP = 200        # total free messages for ALL visitors per day


def get_access_code():
    """Reads ACCESS_CODE from Streamlit secrets or your .env / environment."""
    try:
        return st.secrets["ACCESS_CODE"]
    except Exception:
        return os.getenv("ACCESS_CODE", "")


@st.cache_resource
def _usage():
    return {"day": date.today(), "count": 0}


def count_today():
    """One shared counter for everyone. Resets every new day."""
    usage = _usage()
    if usage["day"] != date.today():
        usage["day"] = date.today()
        usage["count"] = 0
    return usage


def unlock_form():
    with st.form("unlock_form"):
        code = st.text_input("Access code", type="password")
        if st.form_submit_button("Unlock"):
            real = get_access_code()
            if real and code.strip() == real:
                st.session_state.unlocked = True
                st.rerun()
            else:
                st.error("Wrong code. Try again.")


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

if "unlocked" not in st.session_state:
    st.session_state.unlocked = False   # becomes True after a correct code
    st.session_state.used = 0           # free messages used so far

# ---- Display conversation history ----
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user", avatar="🧑‍💻").write(msg.content)

    elif isinstance(msg, AIMessage):
        st.chat_message("assistant", avatar="🪶").write(msg.content)

# ---- Free limit check ----
if not st.session_state.unlocked:
    if count_today()["count"] >= DAILY_CAP:
        st.info("ChatDynast is resting for today because a lot of people used it. "
                "Come back tomorrow! 🪶")
        unlock_form()
        st.stop()
    if st.session_state.used >= FREE_MESSAGES:
        st.warning("You've used your free messages. Enter an access code to keep chatting.")
        unlock_form()
        st.stop()
    st.caption(f"🪶 Free messages left: {FREE_MESSAGES - st.session_state.used}")

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
                if not st.session_state.unlocked:
                    st.session_state.used += 1
                    count_today()["count"] += 1
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
