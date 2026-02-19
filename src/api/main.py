from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
from typing import Dict

app = FastAPI(
    title="Bati Bank BNPL Credit Risk API – Week 12 Capstone Task 6",
    description="Credit risk prediction API (mirrors dashboard logic)",
    version="1.0.0"
)

# Same model URI as your working dashboard
MODEL_URI = "runs:/509e75e100aa43e99fafd4a978549b33/model"

# Load model at startup (will fail on Render, fallback to demo)
model = None
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

# Same feature list as dashboard
FEATURE_COLUMNS = [
    "total_amount",
    "avg_amount",
    "transaction_count",
    "std_amount",
    "avg_hour",
    "avg_day"
]

class CreditRequest(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    avg_hour: float
    avg_day: float

@app.get("/health")
def health_check():
    mode = "real" if model is not None else "demo (fallback)"
    return {"status": "healthy", "mode": mode}

@app.post("/predict", response_model=Dict)
def predict_credit_risk(data: CreditRequest):
    try:
        # Convert input to DataFrame (same as dashboard)
        input_data = pd.DataFrame([data.dict()])

        if model is not None:
            # Real model prediction (same as dashboard)
            input_data = input_data[model.feature_names_in_]
            prob = model.predict_proba(input_data)[0][1]
            mode = "real"
        else:
            # Exact same fallback as your dashboard
            rfm_score = (data.transaction_count / 50.0) + (data.total_amount / 20000.0)
            volatility_penalty = data.std_amount / max(abs(data.total_amount), 1)
            time_score = (data.avg_hour / 24.0) + (data.avg_day / 31.0)
            combined = (rfm_score - volatility_penalty) * 0.7 + time_score * 0.3
            prob = max(0.05, min(0.90, 0.55 - combined * 0.45))
            mode = "demo (fallback)"

        # Same score calculation as dashboard
        score = max(300, min(850, int(850 - prob * 550)))

        # Same decision logic as dashboard
        decision = "Decline/Strict" if prob > 0.35 else "Approve"
        risk_level = "High" if prob > 0.35 else "Low-Moderate"

        return {
            "default_probability": round(float(prob), 4),
            "credit_score": score,
            "decision": decision,
            "risk_level": risk_level,
            "mode": mode,
            "message": "High risk detected - strict terms or decline recommended" if prob > 0.35 else "Low-moderate risk - standard approval possible"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    mode = "real" if model is not None else "demo (fallback)"
    return {
        "message": "Bati Bank BNPL Credit Risk API (Week 12 Capstone - Task 6)",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "mode": mode
    }