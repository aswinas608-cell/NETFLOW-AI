"""
main.py — FastAPI Application for Industrial Network Traffic Predictive Forecaster

This is the main server entrypoint. It exposes REST endpoints for:
  - LSTM-based traffic prediction (bandwidth, packets, latency)
  - Groq LLM-powered insights and recommendations
  - Combined forecast (predict → analyze) pipeline
  - Model training from uploaded data
  - Sample data retrieval

The LSTM model operates in "demo mode" with random weights when no
saved checkpoint is found, ensuring the API is always functional.
"""

import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_utils import (
    calculate_congestion_probability,
    create_sequences,
    denormalize_data,
    generate_synthetic_data,
    load_csv,
    normalize_data,
)
from model import TrafficLSTM, load_model, predict_future, save_model, train_model

# ──────────────────────────────────────────────────────────────
# Configuration & Global State
# ──────────────────────────────────────────────────────────────

# Load environment variables from .env file (if present)
load_dotenv()

# Resolve device — prefer CUDA when available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the model checkpoint
MODEL_PATH: str = os.getenv("MODEL_PATH", "../models/traffic_lstm.pth")

# Feature columns used by the LSTM
FEATURE_COLUMNS: List[str] = ["bandwidth_utilization", "packet_count", "latency_ms"]

# Sequence length the model expects as input
SEQ_LENGTH: int = 24


def _initialize_model() -> TrafficLSTM:
    """
    Attempt to load a saved checkpoint; fall back to a fresh model
    with random weights (demo mode) if loading fails.
    """
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH, device=DEVICE)
            print(f"[INFO] Model loaded from checkpoint: {MODEL_PATH}")
            return model
    except Exception as exc:
        print(f"[WARN] Failed to load model from {MODEL_PATH}: {exc}")

    print("[INFO] Initializing fresh TrafficLSTM model (demo mode - random weights)")
    model = TrafficLSTM().to(DEVICE)
    model.eval()
    return model


# Global model instance
lstm_model: TrafficLSTM = _initialize_model()

# Track whether a real checkpoint was loaded
model_loaded_from_checkpoint: bool = os.path.exists(MODEL_PATH)


# ──────────────────────────────────────────────────────────────
# Pydantic Request / Response Schemas
# ──────────────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """Request body for the /api/predict endpoint."""

    historical_data: List[Dict[str, Any]] = Field(
        ...,
        description="List of dicts with keys: bandwidth_utilization, packet_count, latency_ms",
        min_length=1,
    )
    horizon: int = Field(
        default=12,
        ge=1,
        le=96,
        description="Number of future time steps to forecast",
    )


class InsightRequest(BaseModel):
    """Request body for the /api/insights endpoint."""

    predictions: List[Dict[str, float]] = Field(
        ...,
        description="Predicted metric values from the LSTM model",
    )
    current_metrics: Dict[str, Any] = Field(
        ...,
        description="Current/latest observed network metrics",
    )
    groq_api_key: str = Field(
        ...,
        description="Groq API key for LLM inference",
    )
    model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model identifier",
    )


class ForecastRequest(BaseModel):
    """Combined predict + insights request for /api/forecast."""

    historical_data: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
    )
    horizon: int = Field(default=12, ge=1, le=96)
    groq_api_key: str = Field(
        ...,
        description="Groq API key for LLM inference",
    )
    model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model identifier",
    )


class TrainRequest(BaseModel):
    """Request body for the /api/train endpoint."""

    training_data: List[Dict[str, Any]] = Field(
        ...,
        description="Training samples with bandwidth_utilization, packet_count, latency_ms",
        min_length=50,
    )
    epochs: int = Field(default=50, ge=1, le=500)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)


# ──────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Industrial Network Traffic Predictive Forecaster",
    description=(
        "LSTM + Groq LLM powered API for predicting network traffic metrics "
        "and generating actionable congestion-mitigation insights."
    ),
    version="1.0.0",
)

# Enable CORS for all origins (frontend dev convenience)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────


def _classify_risk(probability: float) -> str:
    """Map a congestion probability to a human-readable risk level."""
    if probability >= 0.7:
        return "critical"
    elif probability >= 0.4:
        return "high"
    elif probability >= 0.2:
        return "moderate"
    return "low"


def _build_groq_messages(
    predictions: List[Dict[str, float]],
    current_metrics: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Construct the system + user messages for the Groq LLM.
    """
    system_prompt = (
        "You are an expert industrial network engineer AI assistant. "
        "Analyze the provided network traffic predictions and current metrics. "
        "Identify potential congestion events, bandwidth bottlenecks, and "
        "latency anomalies. Provide:\n"
        "1. A concise analysis paragraph.\n"
        "2. A JSON array of actionable recommendations.\n"
        "3. An overall severity level: 'low', 'moderate', 'high', or 'critical'.\n\n"
        "Respond ONLY with valid JSON in this format:\n"
        '{"analysis": "...", "recommendations": ["...", "..."], "severity": "..."}'
    )

    user_content = (
        f"## Current Metrics\n{json.dumps(current_metrics, indent=2)}\n\n"
        f"## Predicted Metrics (next {len(predictions)} steps)\n"
        f"{json.dumps(predictions, indent=2)}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


async def _call_groq(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str = "llama-3.3-70b-versatile",
) -> Dict[str, Any]:
    """
    Call the Groq API for chat completions.

    Returns parsed JSON from the LLM response, or a fallback dict
    if parsing fails.
    """
    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.3,
            max_tokens=1024,
        )

        raw_content: str = chat_completion.choices[0].message.content or ""

        # Attempt to extract JSON from the response
        # The LLM may wrap the JSON in markdown code fences
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            # Remove code fences
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            )

        parsed = json.loads(cleaned)
        return parsed

    except json.JSONDecodeError:
        # LLM returned non-JSON — wrap raw text as analysis
        return {
            "analysis": raw_content,
            "recommendations": [
                "Review the raw LLM output for detailed recommendations."
            ],
            "severity": "moderate",
        }
    except Exception as exc:
        return {
            "analysis": f"Groq API error: {str(exc)}",
            "recommendations": [
                "Verify your Groq API key is valid.",
                "Check network connectivity to api.groq.com.",
            ],
            "severity": "unknown",
        }


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────


@app.get("/api/antigravity", tags=["meta"])
async def root() -> Dict[str, str]:
    """Easter egg endpoint."""
    return {
        "message": "import antigravity  # 🚀 You've discovered the Easter egg!",
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
async def health() -> Dict[str, Any]:
    """Health check — confirms the server and model are operational."""
    return {
        "status": "ok",
        "model_loaded": model_loaded_from_checkpoint,
        "device": str(DEVICE),
    }


@app.get("/api/sample-data", tags=["data"])
async def sample_data() -> List[Dict[str, Any]]:
    """
    Return sample traffic data from the CSV in ../data/.

    Falls back to generating synthetic data if the CSV is missing.
    """
    csv_path = Path(__file__).resolve().parent.parent / "data" / "sample_traffic.csv"

    try:
        if csv_path.exists():
            df = load_csv(str(csv_path))
        else:
            print(f"[WARN] {csv_path} not found - generating synthetic data")
            df = generate_synthetic_data(n_points=200)
    except Exception as exc:
        print(f"[WARN] Error loading CSV: {exc} - generating synthetic data")
        df = generate_synthetic_data(n_points=200)

    # Convert timestamps to ISO strings for JSON serialization
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str)

    return df.to_dict(orient="records")


@app.post("/api/predict", tags=["inference"])
async def predict(request: PredictionRequest) -> Dict[str, Any]:
    """
    Run LSTM inference to forecast future traffic metrics.

    Accepts historical data, normalizes it, feeds it through the model,
    and returns denormalized predictions with congestion probability.
    """
    global lstm_model

    try:
        import pandas as pd

        # Validate that input has enough data points for a full sequence
        if len(request.historical_data) < SEQ_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"At least {SEQ_LENGTH} historical data points are required. "
                    f"Got {len(request.historical_data)}."
                ),
            )

        # Build DataFrame from request
        df = pd.DataFrame(request.historical_data)

        # Ensure required columns are present
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required column: {col}",
                )

        # Normalize
        df_norm, scaler = normalize_data(df, FEATURE_COLUMNS)

        # Extract the last SEQ_LENGTH rows as the input window
        input_data = df_norm[FEATURE_COLUMNS].values[-SEQ_LENGTH:]
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(DEVICE)

        # Autoregressive prediction
        raw_predictions = predict_future(
            lstm_model, input_tensor, n_steps=request.horizon, device=DEVICE
        )

        # Denormalize predictions back to original scale
        pred_np = raw_predictions.cpu().numpy()
        predictions: List[Dict[str, float]] = []

        for step in pred_np:
            pred_dict: Dict[str, float] = {}
            for i, col in enumerate(FEATURE_COLUMNS):
                value = float(denormalize_data(np.array([step[i]]), scaler, col)[0])
                pred_dict[col] = round(value, 2)
            predictions.append(pred_dict)

        # Calculate congestion probability across the forecast horizon
        congestion_prob = calculate_congestion_probability(predictions)
        risk_level = _classify_risk(congestion_prob)

        return {
            "predictions": predictions,
            "congestion_probability": congestion_prob,
            "risk_level": risk_level,
        }

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


@app.post("/api/insights", tags=["inference"])
async def insights(request: InsightRequest) -> Dict[str, Any]:
    """
    Generate AI-powered insights using the Groq LLM.

    Takes LSTM predictions and current metrics, sends them to the LLM
    for analysis, and returns structured recommendations.
    """
    if not request.groq_api_key or request.groq_api_key == "your_groq_api_key_here":
        raise HTTPException(
            status_code=400,
            detail="A valid Groq API key is required for insights.",
        )

    try:
        messages = _build_groq_messages(request.predictions, request.current_metrics)
        result = await _call_groq(
            api_key=request.groq_api_key,
            messages=messages,
            model=request.model,
        )
        return result

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Insight generation failed: {str(exc)}",
        )


@app.post("/api/forecast", tags=["inference"])
async def forecast(request: ForecastRequest) -> Dict[str, Any]:
    """
    Combined endpoint: runs LSTM prediction then Groq LLM analysis.

    This is a convenience endpoint that chains /api/predict and
    /api/insights into a single request.
    """
    # Step 1: Run prediction
    predict_request = PredictionRequest(
        historical_data=request.historical_data,
        horizon=request.horizon,
    )
    prediction_result = await predict(predict_request)

    # Step 2: Build current metrics from the last historical data point
    current_metrics = request.historical_data[-1] if request.historical_data else {}

    # Step 3: Run Groq insights (graceful if key is missing)
    insight_result: Dict[str, Any] = {}
    if request.groq_api_key and request.groq_api_key != "your_groq_api_key_here":
        try:
            insight_request = InsightRequest(
                predictions=prediction_result["predictions"],
                current_metrics=current_metrics,
                groq_api_key=request.groq_api_key,
                model=request.model,
            )
            insight_result = await insights(insight_request)
        except Exception as exc:
            insight_result = {
                "analysis": f"Insight generation failed: {str(exc)}",
                "recommendations": [],
                "severity": "unknown",
            }
    else:
        insight_result = {
            "analysis": "No Groq API key provided — skipping LLM insights.",
            "recommendations": [
                "Provide a valid Groq API key to enable AI-powered analysis."
            ],
            "severity": "unknown",
        }

    # Merge prediction and insight responses
    return {
        **prediction_result,
        **insight_result,
    }


@app.post("/api/train", tags=["training"])
async def train(request: TrainRequest) -> Dict[str, Any]:
    """
    Train (or fine-tune) the LSTM model on provided data.

    Accepts raw training samples, normalizes them, creates sequences,
    trains the model, saves a checkpoint, and returns loss metrics.
    """
    global lstm_model, model_loaded_from_checkpoint

    try:
        import pandas as pd

        df = pd.DataFrame(request.training_data)

        # Validate columns
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required column in training data: {col}",
                )

        if len(df) < SEQ_LENGTH + 10:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Need at least {SEQ_LENGTH + 10} training samples. "
                    f"Got {len(df)}."
                ),
            )

        # Normalize
        df_norm, scaler = normalize_data(df, FEATURE_COLUMNS)
        data_np = df_norm[FEATURE_COLUMNS].values

        # Create sequences
        X, y = create_sequences(data_np, seq_length=SEQ_LENGTH)

        print(
            f"[INFO] Training on {len(X)} sequences for {request.epochs} epochs "
            f"(lr={request.learning_rate})"
        )

        # Train
        losses = train_model(
            model=lstm_model,
            train_data=(X, y),
            epochs=request.epochs,
            lr=request.learning_rate,
            device=DEVICE,
        )

        # Save checkpoint
        save_dir = Path(MODEL_PATH).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        save_model(lstm_model, MODEL_PATH)
        model_loaded_from_checkpoint = True

        return {
            "success": True,
            "epochs": request.epochs,
            "final_loss": round(losses[-1], 6) if losses else 0.0,
            "loss_history": [round(l, 6) for l in losses],
        }

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(exc)}",
        )
# Serve frontend static files
# Mount this at the end so it doesn't override explicit API routes
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ──────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
