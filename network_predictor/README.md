# 🌐 NetFlow AI — Industrial Network Traffic Predictor

> **Predictive Traffic Forecasting & Congestion Prediction** using LSTM neural networks + Groq-powered AI insights

NetFlow AI is a full-stack application that forecasts industrial network traffic patterns and predicts congestion before it happens. It combines a PyTorch LSTM model for time-series prediction with Groq's lightning-fast LLM inference for generating actionable mitigation recommendations.

---

## ✨ Features

- **LSTM Forecasting** — PyTorch-based deep learning model predicts bandwidth, packet counts, and latency
- **Groq AI Insights** — Intelligent congestion analysis and mitigation recommendations via Groq LLM
- **Real-Time Dashboard** — Stunning glassmorphism dark-mode UI with Chart.js visualizations
- **Synthetic Data Generator** — Built-in tool to create realistic industrial traffic patterns
- **Docker Ready** — One-command deployment with Docker Compose
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

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** with pip
- **Free Groq API key** — Get one at [console.groq.com](https://console.groq.com) (optional, for AI insights)

### 1. Backend Setup

```bash
cd network_predictor/backend

# Create environment file
cp .env.example .env
# Edit .env and add your Groq API key (optional)

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

The API will be running at **http://localhost:8000**. Visit `/docs` for interactive Swagger documentation.

### 2. Frontend Setup

The frontend is vanilla HTML/CSS/JS — no build step required. Serve it with Python's built-in HTTP server:

```bash
cd network_predictor/frontend

# Serve on port 3000
python -m http.server 3000
```

Open **http://localhost:3000** in your browser.

> **Note:** Don't open `index.html` directly via `file://` — CORS restrictions will block API calls. Always serve via HTTP.

### 3. Using the Dashboard

1. Click **"Load Sample Data"** to load the pre-generated traffic dataset
2. The historical traffic chart will populate with bandwidth, packet, and latency data
3. Adjust the **Prediction Horizon** slider (6–48 steps, where each step = 5 minutes)
4. Optionally enter your **Groq API Key** for AI-powered insights
5. Click **"Generate Prediction & Insights"** to run LSTM forecasting
6. View predictions on the forecast chart and AI analysis in the insights panel

---

### 🐳 Docker (Alternative)

```bash
cd network_predictor

# Build and start all services
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Easter egg (`import antigravity` 🚀) |
| `GET` | `/health` | Health check (model status, device info) |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/api/sample-data` | Returns sample traffic CSV as JSON array |
| `POST` | `/api/predict` | Run LSTM prediction on historical data |
| `POST` | `/api/insights` | Get Groq-powered congestion analysis |
| `POST` | `/api/forecast` | Combined predict + insights in one call |
| `POST` | `/api/train` | Train/fine-tune the LSTM model |

### POST `/api/predict`

**Request:**
```json
{
  "historical_data": [
    {"bandwidth_utilization": 65.3, "packet_count": 9800, "latency_ms": 14.2},
    {"bandwidth_utilization": 72.1, "packet_count": 10800, "latency_ms": 18.5}
  ],
  "horizon": 12
}
```

**Response:**
```json
{
  "predictions": [...],
  "congestion_probability": 0.35,
  "risk_level": "medium"
}
```

### POST `/api/forecast`

**Request:**
```json
{
  "historical_data": [...],
  "horizon": 12,
  "groq_api_key": "gsk_...",
  "model": "llama-3.3-70b-versatile"
}
```

**Response:**
```json
{
  "predictions": [...],
  "congestion_probability": 0.35,
  "risk_level": "medium",
  "analysis": "Based on the traffic patterns...",
  "recommendations": ["Implement QoS policies...", "..."],
  "severity": "warning"
}
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(none)* | Your Groq API key for AI insights |
| `MODEL_PATH` | `../models/traffic_lstm.pth` | Path to saved LSTM checkpoint |

---

## 🧠 Model Details

### LSTM Architecture

| Parameter | Value |
|-----------|-------|
| Input features | 3 (bandwidth, packets, latency) |
| Hidden size | 64 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| Output features | 3 |
| Optimizer | Adam (lr=0.001) |
| Loss function | MSE |
| Gradient clipping | max_norm=1.0 |

The model uses **autoregressive forecasting** — predictions are fed back as input to generate multi-step forecasts.

---

## 🔮 Extending the Project

### Real Data Sources
- **Kafka/MQTT** — Stream real-time network telemetry
- **OPC UA** — Industrial protocol integration
- **Prometheus/Grafana** — Metrics collection pipeline
- **InfluxDB/TimescaleDB** — Time-series database backend

### Better Models
- **Hugging Face Transformers** — Use pre-trained time-series models (TimesFM, Chronos)
- **ONNX Export** — Convert PyTorch model for edge deployment
- **Ensemble Methods** — Combine LSTM with statistical methods (ARIMA, Prophet)

### Production Features
- **Alerting** — Webhook/email/Slack notifications on congestion prediction
- **Edge Deployment** — Run on Raspberry Pi or industrial gateways
- **A/B Testing** — Compare model versions in production
- **Auto-retraining** — Scheduled model updates with new data

---

## 📊 Sample Data

The included `data/sample_traffic.csv` contains 500 rows of synthetic industrial network traffic:

- **5-minute intervals** starting from January 15, 2025
- **Diurnal patterns** — higher traffic during business hours (08:00–18:00)
- **Congestion spikes** — ~10% chance during peak hours (bandwidth 85–98%)
- **Correlated metrics** — packet count and latency track bandwidth realistically

To regenerate: `python data/generate_data.py`

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <code>import antigravity</code> 🚀
</p>
