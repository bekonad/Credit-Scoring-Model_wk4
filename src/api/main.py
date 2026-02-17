from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="Bati Bank BNPL Credit Risk API – Week 12 Capstone")

# Your re-logged Run ID
MODEL_URI = "runs:/509e75e100aa43e99fafd4a978549b33/model"

# Load model at startup
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

class CreditInput(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    avg_hour: float
    avg_day: float

@app.post("/predict")
def predict_credit_risk(data: CreditInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        # Align columns exactly as model expects
        input_df = input_df[model.feature_names_in_]
        prob = model.predict_proba(input_df)[0][1]
        score = max(300, min(850, int(850 - prob * 550)))
        decision = "Decline/Strict" if prob > 0.35 else "Approve"

        return {
            "default_probability": round(float(prob), 4),
            "credit_score": score,
            "decision": decision,
            "risk_level": "High" if prob > 0.35 else "Low-Moderate"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}