import docx
import PyPDF2
import streamlit as st

from module.general_chatbot import show_general_chatbot
from module.law_chatbot import show_law_chatbot
from module.login import login_ui
from prediction_model import predict_judgment


def inject_global_ui():
    st.markdown(
        """
        <style>
        /* === Layout tweaks === */
        .stApp {
            background: linear-gradient(180deg, #0b1020 0%, #0b0f1a 100%);
        }
        header[data-testid="stHeader"] {
            background: rgba(13,18,28,0.55);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .app-title-bar {
            position: sticky; top: 0; z-index: 999;
            background: rgba(13,18,28,0.55);
            backdrop-filter: blur(8px);
            padding: 12px 14px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .app-title {
            font-weight: 700; font-size: 18px; letter-spacing: 0.2px;
            color: #e8eefc; display: inline-flex; gap: 10px; align-items: center;
        }
        .app-subtle {
            color: #9fb4ff; font-weight: 500; opacity: .9;
        }

        /* === Sidebar === */
        section[data-testid="stSidebar"] {
            background: rgba(13,18,28,0.55) !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        /* === Messages (chat bubble feel on top of st.chat_message) === */
        .element-container:has(div[data-testid="stChatMessage"]) {
            margin-bottom: 8px;
        }
        div[data-testid="stChatMessage"] {
            padding: 0;
        }
        div[data-testid="stChatMessage"] > div:nth-child(2) {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 12px 14px;
        }

        /* Make code blocks crisp */
        pre, code, .stCodeBlock {
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            background: rgba(10,14,25,0.9) !important;
        }

        /* === Inputs === */
        textarea, input, .stTextInput input, .stTextArea textarea {
            border-radius: 12px !important;
        }

        /* === Footer hide === */
        footer, #MainMenu {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="app-title-bar">
          <div class="app-title">⚖️ Jublie <span class="app-subtle">• Legal & General Assistant</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------
# FILE PROCESSOR
# -------------------------------------------------------
def extract_text_from_file(uploaded_file) -> str:
    """Extracts text from TXT, DOCX, or PDF files (detected by extension)."""
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    try:
        if ext == "txt":
            return uploaded_file.read().decode("utf-8", errors="replace")
        if ext == "docx":
            return "\n".join(p.text for p in docx.Document(uploaded_file).paragraphs if p.text.strip())
        if ext == "pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
    return ""


# -------------------------------------------------------
# PREDICTION UI
# -------------------------------------------------------
def show_prediction_ui():
    """UI for hybrid ML + LLM legal prediction."""
    st.subheader("⚖️ Legal Judgment Prediction")

    scenario_text = st.text_area("📝 Enter scenario description", height=200)
    uploaded_file = st.file_uploader("📂 Or upload a case file (TXT, DOCX, PDF)", type=["txt", "docx", "pdf"])

    if uploaded_file:
        file_text = extract_text_from_file(uploaded_file)
        if file_text:
            scenario_text = f"{scenario_text}\n{file_text}".strip()

    c1, _ = st.columns([1, 5])
    with c1:
        predict = st.button("🔮 Predict Judgment", use_container_width=True)
    if not predict:
        return
    if not scenario_text.strip():
        st.error("⚠️ Please enter a scenario or upload a file.")
        return

    with st.spinner("Analyzing scenario..."):
        try:
            result = predict_judgment(scenario_text)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return

    for w in result["warnings"]:
        st.warning(w)
    st.success("✅ Analysis complete!")
    st.balloons()

    st.subheader("📜 Case Summary")
    st.markdown(result["summary"] or "No summary available.")

    st.subheader("🔮 Predicted Outcome")
    st.markdown(result["prediction"] or "No prediction available.")

    st.subheader("📊 Confidence")
    st.markdown(f"{result['confidence']}%")
    if result["ml"]:
        st.caption(f"ML classifier: {result['ml']['predicted_decision']} ({result['ml']['confidence']}%)")

    st.subheader("📌 Reasons")
    st.markdown("\n".join(f"- {r}" for r in result["reasons"]) or "No reasons available.")


# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Jublie Chatbot", page_icon="⚖️", layout="wide")
inject_global_ui()

defaults = {
    "authenticated": False,
    "current_page": "login",
    "username": "",
    "law_messages": [],
    "general_messages": []
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# -------------------------------------------------------
# AUTH CHECK
# -------------------------------------------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------------------------------------------
# NAVIGATION SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.title("🌐 Navigation")
    nav_choice = st.radio(
        "Go to:",
        ("Law-Jublie ⚖️", "Com-Jublie 💬", "Logout 🚪"),
        index=0 if st.session_state.current_page == "law" else 1
    )

if nav_choice.startswith("Law"):
    st.session_state.current_page = "law"
elif nav_choice.startswith("Com"):
    st.session_state.current_page = "general"
else:
    st.session_state.clear()
    st.rerun()

# -------------------------------------------------------
# PAGE RENDER
# -------------------------------------------------------
if st.session_state.current_page == "law":
    show_law_chatbot()
    st.markdown("---")
    show_prediction_ui()
else:
    show_general_chatbot()
