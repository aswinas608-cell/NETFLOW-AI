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
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **LLM Inference** | Groq API |
| **Retrieval** | TF-IDF (local) / HuggingFace / OpenAI |
| **PDF Parsing** | pypdf |
| **Frontend** | Vanilla HTML, CSS, JavaScript |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/aswinas608-cell/RAG-CHATBOT.git
cd RAG-CHATBOT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the server
```bash
python -m uvicorn app:app --reload
```

### 4. Open in browser
Navigate to: **http://127.0.0.1:8000**

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
├── static/
│   ├── index.html      # Frontend UI
│   ├── style.css       # Glassmorphism dark-mode styles
│   └── app.js          # Frontend logic
└── data/               # Auto-created: stores the document index (gitignored)
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

## 📜 License

MIT License — feel free to use, modify, and distribute.
