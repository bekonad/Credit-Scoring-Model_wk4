from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

app = FastAPI(
    title="Bati Bank BNPL Credit Risk API – Week 12 Task 6",
    description="Minimal API for credit risk prediction (demo mode)",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    """Health check endpoint – required for deployment"""
    return {"status": "healthy", "mode": "demo"}

class CreditRequest(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    avg_hour: float
    avg_day: float

@app.post("/predict", response_model=Dict)
def predict_credit_risk(data: CreditRequest):
    """Predict default probability and recommendation (demo mode)"""
    try:
        # Same fallback as dashboard when model load fails
        rfm_score = (data.transaction_count / 50.0) + (data.total_amount / 20000.0)
        volatility_penalty = data.std_amount / max(abs(data.total_amount), 1)
        time_score = (data.avg_hour / 24.0) + (data.avg_day / 31.0)
        combined = (rfm_score - volatility_penalty) * 0.7 + time_score * 0.3
        prob = max(0.05, min(0.90, 0.55 - combined * 0.45))

        score = max(300, min(850, int(850 - prob * 550)))
        decision = "Decline/Strict" if prob > 0.35 else "Approve"
        risk_level = "High" if prob > 0.35 else "Low-Moderate"

        return {
            "default_probability": round(float(prob), 4),
            "credit_score": score,
            "decision": decision,
            "risk_level": risk_level,
            "message": "High risk detected - strict terms or decline recommended (demo mode)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "message": "Bati Bank BNPL Credit Risk API (Week 12 Capstone - Task 6)",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "mode": "demo (fallback heuristic)"
    }