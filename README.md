---
title: RAG Chatbot
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: RAG chatbot powered by Groq with TF-IDF & embeddings.
---

# 🤖 RAG Chatbot — Powered by Groq

A **Retrieval-Augmented Generation (RAG)** chatbot built with Python, FastAPI, and the Groq API. Upload your own documents and get fast, grounded answers with source citations — all in a beautiful dark-mode UI.

---

## ✨ Features

- 🚀 **Ultra-fast inference** via [Groq API](https://groq.com) (Llama 3, Mixtral, Gemma 2)
- 📄 **Document support** — Upload `.txt`, `.md`, and `.pdf` files
- 🔍 **Flexible retrieval**:
  - **Local TF-IDF** — works fully offline, no extra API keys needed
  - **HuggingFace Embeddings** — free semantic search via HF Inference API
  - **OpenAI Embeddings** — high-quality vector search
- 📌 **Source citations** — every answer shows the exact retrieved chunks and relevance scores
- 💾 **Persistent index** — documents survive server restarts (stored in `data/index.json`)
- 🎨 **Premium UI** — glassmorphism dark-mode design with smooth animations

---

## 🖼️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **LLM Inference** | Groq API |
| **Retrieval** | TF-IDF (local) / HuggingFace / OpenAI |
| **PDF Parsing** | pypdf |
| **Frontend** | Vanilla HTML, CSS, JavaScript |
| **Deployment** | Docker (Hugging Face Spaces) |

---

## 🚀 Using the App

1. Enter your **Groq API Key** in the sidebar (get one free at [console.groq.com](https://console.groq.com))
2. Upload a `.txt`, `.md`, or `.pdf` document
3. Start chatting with your document!

> **Note:** The document index is stored in-memory on Hugging Face Spaces. Uploaded documents will be cleared when the Space restarts.

---

## ⚙️ Configuration

All settings are configured directly in the UI sidebar:

1. **Groq API Key** — Get yours free at [console.groq.com](https://console.groq.com)
2. **LLM Model** — Choose from Llama 3.3 70B, Mixtral 8x7B, Gemma 2 9B, and more
3. **Retrieval Mode** — Select TF-IDF (no key needed), HuggingFace, or OpenAI
4. **Retrieved Chunks** — Control how many context chunks are retrieved per query (1–8)

---

## 📁 Project Structure

```
RAG-CHATBOT/
├── app.py              # FastAPI backend (API endpoints)
├── rag_engine.py       # RAG core: chunking, indexing, retrieval
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker build config for HF Spaces
├── static/
│   ├── index.html      # Frontend UI
│   ├── style.css       # Glassmorphism dark-mode styles
│   └── app.js          # Frontend logic
└── data/               # Auto-created: stores the document index
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the frontend UI |
| `GET` | `/documents` | List all indexed documents |
| `POST` | `/upload` | Upload and index a document |
| `DELETE` | `/documents/{id}` | Remove a document |
| `POST` | `/chat` | Query the RAG chatbot |
| `POST` | `/validate` | Validate the Groq API key |

---

## 🏗️ Run Locally

```bash
git clone https://github.com/aswinas608-cell/RAG-CHATBOT.git
cd RAG-CHATBOT
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open: **http://127.0.0.1:8000**

---

## 🤗 Hugging Face Spaces Notes

| Topic | Detail |
|---|---|
| **Storage** | Free-tier Spaces have no persistent disk. Uploaded documents are stored in `/tmp` and **reset on restart**. |
| **File size** | Uploads are capped at **10 MB** per file. |
| **Port** | The app binds to port `7860` as required by HF Spaces. |
| **Index path** | Set the `RAG_INDEX_DIR` env var in Space Settings to change where the index is stored. |

---

## 📜 License

MIT License — feel free to use, modify, and distribute.
