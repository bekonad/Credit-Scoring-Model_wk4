from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
from typing import Dict
import traceback

app = FastAPI(
    title="Bati Bank BNPL Credit Risk API – Week 12 Task 6",
    description="Real model with fallback",
    version="1.0.0"
)

MODEL_URI = "runs:/509e75e100aa43e99fafd4a978549b33/model"

model = None
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("SUCCESS: Real model loaded from", MODEL_URI)
except Exception as e:
    print("ERROR: Failed to load real model")
    print(str(e))
    print(traceback.format_exc())
    model = None

@app.get("/health")
def health_check():
    mode = "real" if model is not None else "demo (fallback)"
    return {"status": "healthy", "mode": mode}

class CreditRequest(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    avg_hour: float
    avg_day: float

@app.post("/predict", response_model=Dict)
def predict_credit_risk(data: CreditRequest):
    try:
        input_df = pd.DataFrame([data.dict()])

        if model is not None:
            input_df = input_df[model.feature_names_in_]
            prob = model.predict_proba(input_df)[0][1]
            mode = "real"
        else:
            # fallback
            rfm_score = (data.transaction_count / 50.0) + (data.total_amount / 20000.0)
            volatility_penalty = data.std_amount / max(abs(data.total_amount), 1)
            time_score = (data.avg_hour / 24.0) + (data.avg_day / 31.0)
            combined = (rfm_score - volatility_penalty) * 0.7 + time_score * 0.3
            prob = max(0.05, min(0.90, 0.55 - combined * 0.45))
            mode = "demo (fallback)"

        score = max(300, min(850, int(850 - prob * 550)))
        decision = "Decline/Strict" if prob > 0.35 else "Approve"
        risk_level = "High" if prob > 0.35 else "Low-Moderate"

        return {
            "default_probability": round(float(prob), 4),
            "credit_score": score,
            "decision": decision,
            "risk_level": risk_level,
            "mode": mode
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))