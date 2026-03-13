"""
app.py
SmartChat — Streamlit chatbot built on LangChain + Groq.
Features:
  • RAG (upload documents → embed → retrieve relevant chunks)
  • Live Web Search (DuckDuckGo, no API key needed)
  • Response Modes (Concise vs Detailed)

Run:  streamlit run app.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config.config import (
    APP_TITLE, DEFAULT_RESPONSE_MODE,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, MAX_HISTORY,
)
from models.llm import get_chatgroq_model, build_system_prompt
from utils.rag_utils import (
    VectorStore, extract_text_from_file,
    build_vector_store, retrieve_relevant_chunks, format_context,
)
from utils.search_utils import web_search, format_search_results
from utils.chat_utils import build_augmented_prompt, trim_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.4rem 2rem; border-radius: 12px; margin-bottom: 1.2rem; color: white;
}
.feature-badge {
    display: inline-block; background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3); color: white;
    padding: 0.2rem 0.7rem; border-radius: 20px;
    font-size: 0.75rem; margin: 0.2rem 0.3rem 0 0;
}
</style>
""", unsafe_allow_html=True)


# ── Core chat function (same as your original) ────────────────────────────────

def get_chat_response(chat_model, messages: list[dict], system_prompt: str) -> str:
    """Get response from the Groq chat model using LangChain message format."""
    try:
        formatted = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted.append(HumanMessage(content=msg["content"]))
            else:
                formatted.append(AIMessage(content=msg["content"]))
        response = chat_model.invoke(formatted)
        return response.content
    except Exception as exc:
        logger.error("Chat response error: %s", exc)
        return f"❌ Error: {exc}"


# ── Session state ──────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "messages": [],
        "vector_store": VectorStore(),
        "response_mode": DEFAULT_RESPONSE_MODE,
        "use_rag": False,
        "use_web_search": False,
        "uploaded_doc_names": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Instructions page ──────────────────────────────────────────────────────────

def instructions_page():
    st.title("📖 SmartChat — Setup & Instructions")
    st.markdown("""
Welcome! Follow the steps below to get the chatbot running.

## 🔧 Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## 🔑 API Key Setup

Create a `.env` file in the project root and add your Groq key:

```
GROQ_API_KEY=your_groq_key_here
```

Get your free Groq key at [console.groq.com/keys](https://console.groq.com/keys).

> For **Streamlit Cloud** deployment, add the key under **Settings → Secrets** instead of a `.env` file.

## 🚀 Features

| Feature | How to use |
|---------|-----------|
| **💬 Response Mode** | Switch Concise ↔ Detailed in the sidebar |
| **📚 RAG** | Toggle on RAG, then upload `.txt`, `.md`, or `.pdf` files |
| **🌐 Web Search** | Toggle on Web Search for real-time internet results |

### 💬 Response Modes
- **Concise** — Short, direct answers (2-4 sentences). Best for quick questions.
- **Detailed** — In-depth, structured answers with bullet points. Best for research.

### 📚 RAG (Document Q&A)
Upload your own documents and the chatbot will search through them to answer your questions. Great for asking about reports, notes, or any private files.

### 🌐 Web Search
When enabled, the bot searches DuckDuckGo in real time before answering. Useful for current events or anything that may be outside the model's training data. No API key required.

## 💡 Tips
- Both RAG and Web Search can be on at the same time.
- If no documents are uploaded, RAG has no effect even when toggled on.
- Use the **Clear Chat** button to start a fresh conversation.

---
Navigate to **Chat** in the sidebar to start! 🎉
""")


# ── Chat page ──────────────────────────────────────────────────────────────────

def chat_page():
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;font-size:1.8rem;">🤖 SmartChat</h1>
        <p style="margin:0.3rem 0 0.6rem;opacity:0.85;font-size:0.9rem;">
            Groq · LangChain · RAG · Live Web Search · Concise / Detailed Modes
        </p>
        <span class="feature-badge">📚 RAG</span>
        <span class="feature-badge">🌐 Web Search</span>
        <span class="feature-badge">⚡ Concise / 📖 Detailed</span>
    </div>
    """, unsafe_allow_html=True)

    # Status indicators
    c1, c2, c3 = st.columns(3)
    c1.metric("Response Mode", "⚡ Concise" if st.session_state.response_mode == "concise" else "📖 Detailed")
    c2.metric("Knowledge Base", f"{len(st.session_state.uploaded_doc_names)} doc(s) loaded" if st.session_state.uploaded_doc_names else "No docs uploaded")
    c3.metric("Web Search", "✅ Active" if st.session_state.use_web_search else "Disabled (toggle in sidebar)")
    st.divider()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📌 Sources used"):
                    st.markdown(msg["sources"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # ── RAG retrieval ──────────────────────────────────────────────────────
        rag_context = ""
        search_context = ""
        source_parts: list[str] = []

        if st.session_state.use_rag and not st.session_state.vector_store.is_empty():
            with st.spinner("🔍 Searching your documents…"):
                try:
                    chunks = retrieve_relevant_chunks(
                        prompt, st.session_state.vector_store,
                        top_k=TOP_K_RESULTS, model_name=EMBEDDING_MODEL,
                    )
                    if chunks:
                        rag_context = format_context(chunks)
                        srcs = list({c.source for c in chunks})
                        source_parts.append("📚 **From your documents:** " + ", ".join(f"`{s}`" for s in srcs))
                except Exception as exc:
                    st.warning(f"⚠️ Document search failed: {exc}")

        # ── Web search ─────────────────────────────────────────────────────────
        if st.session_state.use_web_search:
            with st.spinner("🌐 Searching the web…"):
                try:
                    results = web_search(prompt, max_results=4)
                    if results:
                        search_context = format_search_results(results)
                        source_parts.append(
                            "🌐 **Web results:**\n" +
                            "\n".join(f"- [{r.title}]({r.url})" for r in results[:3])
                        )
                except Exception as exc:
                    st.warning(f"⚠️ Web search failed: {exc}")

        # ── Build final prompt ─────────────────────────────────────────────────
        final_prompt = build_augmented_prompt(
            user_message=prompt,
            rag_context=rag_context or None,
            search_context=search_context or None,
        )

        system_prompt = build_system_prompt(mode=st.session_state.response_mode)

        # ── Get Groq response ──────────────────────────────────────────────────
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ Thinking…")
            try:
                chat_model = get_chatgroq_model()
                history = trim_history(st.session_state.messages[:-1], max_turns=MAX_HISTORY // 2)
                history.append({"role": "user", "content": final_prompt})
                reply = get_chat_response(chat_model, history, system_prompt)
                placeholder.markdown(reply)

                sources_md = "\n\n".join(source_parts)
                if sources_md:
                    with st.expander("📌 Sources used"):
                        st.markdown(sources_md)

            except Exception as exc:
                reply = f"❌ **Error:** {exc}\n\nMake sure your `GROQ_API_KEY` is set in `.env`."
                placeholder.markdown(reply)
                sources_md = ""

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "sources": sources_md if source_parts else "",
        })


# ── Sidebar ────────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)

        st.divider()

        # ── Response Mode ──────────────────────────────────────────────────────
        st.markdown("### 💬 Response Mode")
        mode = st.radio(
            "How should the bot reply?",
            ["concise", "detailed"],
            index=0 if st.session_state.response_mode == "concise" else 1,
            format_func=lambda x: "⚡ Concise — short & direct" if x == "concise" else "📖 Detailed — in-depth",
        )
        st.session_state.response_mode = mode

        st.divider()

        # ── RAG ────────────────────────────────────────────────────────────────
        st.markdown("### 📚 Document Q&A (RAG)")
        st.caption("Upload files so the bot can answer questions from them.")
        st.session_state.use_rag = st.toggle("Enable RAG", value=st.session_state.use_rag)

        if st.session_state.use_rag:
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=["txt", "md", "pdf"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                new_names = [f.name for f in uploaded_files]
                if set(new_names) != set(st.session_state.uploaded_doc_names):
                    with st.spinner("🔄 Reading & indexing documents…"):
                        try:
                            texts, names = [], []
                            for f in uploaded_files:
                                texts.append(extract_text_from_file(f))
                                names.append(f.name)
                            st.session_state.vector_store = build_vector_store(
                                texts, names,
                                model_name=EMBEDDING_MODEL,
                                chunk_size=CHUNK_SIZE,
                                overlap=CHUNK_OVERLAP,
                            )
                            st.session_state.uploaded_doc_names = new_names
                            st.success(f"✅ {len(texts)} document(s) ready")
                        except Exception as exc:
                            st.error(f"❌ Indexing failed: {exc}")

            if st.session_state.uploaded_doc_names:
                for n in st.session_state.uploaded_doc_names:
                    st.markdown(f"- 📄 `{n}`")

        st.divider()

        # ── Web Search ─────────────────────────────────────────────────────────
        st.markdown("### 🌐 Live Web Search")
        st.caption("When ON, the bot searches DuckDuckGo before answering — great for current events.")
        st.session_state.use_web_search = st.toggle(
            "Enable Web Search",
            value=st.session_state.use_web_search,
        )
        if st.session_state.use_web_search:
            st.success("✅ Web search is active")
        else:
            st.info("💡 Turn on to get real-time web results")

        st.divider()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return page


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    page = sidebar()
    if page == "Instructions":
        instructions_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()