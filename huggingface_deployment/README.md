---
title: NetFlow AI
emoji: 🌐
colorFrom: cyan
colorTo: violet
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Industrial network traffic predictor and congestion forecaster using LSTM & Groq.
---

# 🌐 NetFlow AI — Industrial Network Traffic Predictor

> **Predictive Traffic Forecasting & Congestion Prediction** using LSTM neural networks + Groq-powered AI insights

NetFlow AI is a full-stack application that forecasts industrial network traffic patterns and predicts congestion before it happens. It combines a PyTorch LSTM model for time-series prediction with Groq's lightning-fast LLM inference for generating actionable mitigation recommendations.

Deployed on **Hugging Face Spaces** using Docker!

---

## ✨ Features

- **LSTM Forecasting** — PyTorch-based deep learning model predicts bandwidth, packet counts, and latency
- **Groq AI Insights** — Intelligent congestion analysis and mitigation recommendations via Groq LLM
- **Real-Time Dashboard** — Stunning glassmorphism dark-mode UI with Chart.js visualizations
- **Synthetic Data Generator** — Built-in tool to create realistic industrial traffic patterns
- **Docker Ready** — One-command deployment with Docker Compose or Hugging Face Spaces
- **RESTful API** — Clean FastAPI backend with interactive Swagger docs
- **Demo Mode** — Works out-of-the-box without any trained model (random weights for demo)
- **`import antigravity`** — Classic Python Easter egg 🚀

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend Dashboard                     │
│          (HTML + CSS + Chart.js Visualizations)           │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/REST
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  LSTM Model  │  │  Data Utils   │  │  Groq Client   │  │
│  │  (PyTorch)   │  │  (Pandas)     │  │  (LLM API)     │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘  │
│         │                │                   │            │
│    predictions     preprocessing       AI insights       │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
network_predictor/
├── backend/
│   ├── main.py              # FastAPI server + all API endpoints
│   ├── model.py             # LSTM model definition (PyTorch)
│   ├── data_utils.py        # Data processing & synthetic generation
│   ├── requirements.txt     # Python dependencies
│   ├── .env.example         # Environment variable template
│   └── Dockerfile           # Backend container
├── frontend/
│   ├── index.html           # Dashboard UI
│   ├── style.css            # Premium dark-mode glassmorphism styles
│   └── app.js               # Chart.js visualizations + API integration
├── data/
│   ├── sample_traffic.csv   # Pre-generated synthetic dataset (500 rows)
│   └── generate_data.py     # Data generation script
├── models/                  # Saved trained model checkpoints
│   └── .gitkeep
├── docker-compose.yml       # Full-stack orchestration
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (Local Run)

### Prerequisites

- **Python 3.11+** with pip
- **Free Groq API key** — Get one at [console.groq.com](https://console.groq.com) (optional, for AI insights)

### Run via virtual environment

```bash
cd network_predictor

# Create virtual env
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r backend/requirements.txt

# Start backend (serves frontend as well on port 8000!)
python backend/main.py
```

Then open **http://localhost:8000** in your browser.

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the frontend UI dashboard |
| `GET` | `/api/antigravity` | Easter egg (`import antigravity` 🚀) |
| `GET` | `/health` | Health check (model status, device info) |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/api/sample-data` | Returns sample traffic CSV as JSON array |
| `POST` | `/api/predict` | Run LSTM prediction on historical data |
| `POST` | `/api/insights` | Get Groq-powered congestion analysis |
| `POST` | `/api/forecast` | Combined predict + insights in one call |
| `POST` | `/api/train` | Train/fine-tune the LSTM model |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
