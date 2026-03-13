# SmartChat 🤖

An AI chatbot built with Streamlit and LangChain (Groq) featuring RAG, live web search, and response modes.

## Features

- **RAG** — Upload `.txt`, `.md`, or `.pdf` files and ask questions about them
- **Web Search** — Toggle DuckDuckGo search for real-time answers (no API key needed)
- **Response Modes** — Switch between Concise (2-4 sentences) and Detailed replies

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/iChetanRaval/SmartChat_Chatbot
cd SmartChat_Chatbot

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# 5. Run
streamlit run app.py
```

Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys).

## Project Structure

```
SmartChat_Chatbot/
├── config/config.py        # API keys and settings
├── models/llm.py           # Groq + LangChain wrapper
├── models/embeddings.py    # Sentence-transformers for RAG
├── utils/rag_utils.py      # Document chunking and retrieval
├── utils/search_utils.py   # DuckDuckGo web search
├── utils/chat_utils.py     # Prompt assembly
├── app.py                  # Streamlit UI
└── requirements.txt
```
