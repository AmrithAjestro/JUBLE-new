import random
import re

import streamlit as st
from deep_translator import GoogleTranslator

from module.llm import LLM_MODEL, chat

# ------------------------------
# Fun facts database
# ------------------------------
FUN_FACTS = [
    "🍯 Honey never spoils! Archaeologists found edible honey over 3,000 years old in Egyptian tombs.",
    "🐙 Octopuses have three hearts — two for the gills, one for the body.",
    "🍌 Bananas are berries, but strawberries aren't!",
    "🌍 A day on Venus is longer than a year on Venus.",
    "🦈 Sharks existed before trees."
]

# ------------------------------
# Settings
# ------------------------------
LANG_MAP = {
    "No translation": None,
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Hindi": "hi",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-CN",
    "Russian": "ru",
    "Tamil": "ta",
    "Telugu": "te"
}

# Supported Groq models (module.llm.chat falls back automatically if one is retired)
MODEL_CHOICES = list(dict.fromkeys([
    LLM_MODEL,
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]))

TONE_PRESETS = {
    "Neutral": "neutral and concise",
    "Friendly": "warm, encouraging, and friendly",
    "Formal": "formal and precise",
    "Playful": "light, witty, and playful"
}

RESPONSE_LEN = {
    "Short": "Keep it within 2-3 sentences.",
    "Normal": "Answer with a short paragraph.",
    "Detailed": "Provide a detailed, multi-paragraph explanation when helpful."
}

HELP_TEXT = (
    "**Commands**\n"
    "- `/funfact` — random fun fact\n"
    "- `/summarize <text>` — short summary\n"
    "- `/translate:<lang> <text>` — translate text (e.g., `/translate:hi hello`)\n"
    "- `/setlang:<lang>` — set default translation language\n"
    "- `/remember <fact>` — store a fact in memory\n"
    "- `/forget <fact>` — remove a fact\n"
    "- `/memory` — show stored facts\n"
    "- `/clear` — clear chat"
)

ARTICLE_PATTERN = re.compile(r"\barticle\s*\d+[a-z]?\b", re.IGNORECASE)
CODE_MARKERS = ("#include", "int main(", "def ", "class ", "function ", "console.log",
                "import ", "public static void", "<?php", "<html", "SELECT ")


def translate_text(text: str, target_code: str | None) -> str:
    if not target_code:
        return text
    try:
        return GoogleTranslator(source="auto", target=target_code).translate(text)
    except Exception as e:
        return f"{text}\n\n> ⚠️ Translation error: {e}"


def fence_code_if_needed(s: str) -> str:
    if "```" not in s and any(key in s for key in CODE_MARKERS):
        return f"```\n{s}\n```"
    return s


def chat_as_markdown(messages) -> str:
    return "".join(f"**{m['role'].capitalize()}:**\n\n{m['content']}\n\n---\n\n" for m in messages)


_BOLD = re.compile(r"\*\*(.*?)\*\*")


def chat_as_text(messages) -> str:
    return "".join(m["role"].upper() + ": " + _BOLD.sub(r"\1", m["content"]) + "\n\n" for m in messages)


def limit_context(msgs, max_items=16, max_chars=12000):
    """Keep the most recent messages that fit within max_items / max_chars (always at least one)."""
    out, total = [], 0
    for m in reversed(msgs[-max_items:]):
        total += len(m["content"])
        if out and total > max_chars:
            break
        out.append(m)
    return list(reversed(out))


def make_system_prompt(tone: str, resp_len_hint: str):
    return (
        f"You are **Jublie**, a helpful AI assistant.\n"
        f"Tone: {tone}. {resp_len_hint}\n"
        "If asked your name, reply exactly: 'My name is Jublie.'\n"
        "If asked your role, say you're a helpful AI assistant.\n"
        "For queries that mention a specific Article number of the Constitution (e.g., Article 40, 47, 102), "
        "provide only a short 2 - 3 line summary.\n"
        "Do not reveal model/provider details. Be natural and polite. Use Markdown formatting when helpful."
    )


def safe_chat(model, messages, temperature, max_tokens) -> str:
    try:
        return chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return str(e)


def handle_command(user_msg: str, model_choice, temperature, max_tokens) -> str | None:
    """Returns the reply for a /command, or None if the message is not a command."""
    cmd = user_msg.strip()
    memory = st.session_state.general_memory

    if cmd.startswith("/help"):
        return HELP_TEXT
    if cmd.startswith("/funfact"):
        return random.choice(FUN_FACTS)
    if cmd.startswith("/summarize"):
        target_text = cmd[len("/summarize"):].strip()
        if not target_text:
            return "Provide text after `/summarize`."
        with st.spinner("📝 Summarizing…"):
            return safe_chat(model_choice, [
                {"role": "system", "content": "Summarize the following text in 3–5 sentences. Be clear and faithful."},
                {"role": "user", "content": target_text},
            ], temperature, max_tokens)
    if cmd.startswith("/translate:"):
        m = re.match(r"^/translate:([a-zA-Z\-]+)\s+(.+)$", cmd, re.DOTALL)
        if not m:
            return "Usage: `/translate:<lang> your text` (e.g., `/translate:hi Hello`)"
        lang, text = m.groups()
        with st.spinner(f"🌐 Translating to {lang}…"):
            try:
                return f"**Translated ({lang})**:\n\n{GoogleTranslator(source='auto', target=lang).translate(text)}"
            except Exception as e:
                return f"⚠️ Translation error: {e}"
    if cmd.startswith("/setlang:"):
        m = re.match(r"^/setlang:([a-zA-Z\-]+)$", cmd)
        if not m:
            return "Usage: `/setlang:<lang>` (e.g., `/setlang:hi`)"
        st.session_state.general_lang_override = m.group(1)
        return f"✅ Default translation language set to `{m.group(1)}`."
    if cmd.startswith("/remember"):
        fact = cmd[len("/remember"):].strip()
        if not fact:
            return "Add a fact after `/remember`."
        memory.append(fact)
        return "🧠 Noted."
    if cmd.startswith("/forget"):
        fact = cmd[len("/forget"):].strip()
        if not fact:
            return "Provide the fact after `/forget` to remove it."
        if fact in memory:
            memory.remove(fact)
            return "🗑️ Removed."
        return "I couldn’t find that fact in memory."
    if cmd.startswith("/memory"):
        return "**Memory:**\n" + "\n".join(f"- {m}" for m in memory) if memory else "_Memory is empty._"
    return None


# ------------------------------
# General Chatbot UI
# ------------------------------
def show_general_chatbot():
    st.subheader("🤖 General Chatbot — Jublie")

    st.session_state.setdefault("general_messages", [])
    st.session_state.setdefault("general_memory", [])

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        model_choice = st.selectbox("Model", MODEL_CHOICES, index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
        max_tokens = st.slider("Max tokens", 128, 4096, 768, 64)
        tone_name = st.selectbox("Tone", list(TONE_PRESETS.keys()), index=1)
        resp_len_name = st.selectbox("Response length", list(RESPONSE_LEN.keys()), index=1)

        st.markdown("### 🌐 Translation")
        lang_name = st.selectbox("Translate responses into", list(LANG_MAP.keys()), index=0)
        always_translate = st.toggle("Always translate", value=False)

        st.markdown("### 🧠 Memory")
        st.caption("Add facts with /remember; list with /memory.")
        if st.button("Clear memory"):
            st.session_state.general_memory = []

        st.markdown("### 🧹 Chat controls")
        if st.button("Clear chat (/clear)"):
            st.session_state.general_messages = []
            st.rerun()

        st.markdown("### ⬇️ Export")
        if st.session_state.general_messages:
            st.download_button("⬇️ Download chat (.md)", data=chat_as_markdown(st.session_state.general_messages),
                               file_name="general_chat.md", mime="text/markdown")
            st.download_button("⬇️ Download chat (.txt)", data=chat_as_text(st.session_state.general_messages),
                               file_name="general_chat.txt", mime="text/plain")

        st.markdown("---")
        st.markdown("**Commands**: `/help`, `/funfact`, `/summarize <text>`, `/translate:<lang> <text>`, "
                    "`/setlang:<lang>`, `/remember <fact>`, `/forget <fact>`, `/memory`, `/clear`")

    for msg in st.session_state.general_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask anything… (use /help for commands)")
    if not user_msg:
        return

    if user_msg.strip().startswith("/clear"):
        st.session_state.general_messages = []
        st.rerun()

    st.session_state.general_messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    content = handle_command(user_msg, model_choice, temperature, max_tokens)

    if content is None:
        with st.spinner("💭 Jublie is thinking…"):
            system_prompt = make_system_prompt(TONE_PRESETS[tone_name], RESPONSE_LEN[resp_len_name])
            messages = [{"role": "system", "content": system_prompt}]
            if ARTICLE_PATTERN.search(user_msg):
                messages.append({"role": "system",
                                 "content": "The user asked about a specific Article. Keep answer to 2–3 lines."})
            if st.session_state.general_memory:
                mem_text = "\n".join(f"- {m}" for m in st.session_state.general_memory)
                messages.append({"role": "system",
                                 "content": f"User-provided memory/facts to respect in answers:\n{mem_text}"})
            messages += limit_context(st.session_state.general_messages, max_items=18, max_chars=14000)

            content = fence_code_if_needed(safe_chat(model_choice, messages, temperature, max_tokens))

            target_code = st.session_state.get("general_lang_override") or LANG_MAP.get(lang_name)
            if always_translate and target_code:
                content = translate_text(content, target_code)

    st.session_state.general_messages.append({"role": "assistant", "content": content})
    with st.chat_message("assistant"):
        st.markdown(content)
