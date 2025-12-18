from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn

# Load champion model (Random Forest or XGBoost)
MODEL_URI = "models:/CreditRisk_RandomForest/Production"  # or local path
model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(title="Bati Bank BNPL Credit Risk API")

# Pydantic input model
class CustomerInput(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    avg_hour: float
    avg_day: float

# Pydantic output model
class RiskOutput(BaseModel):
    probability_of_default: float
    credit_score: float
    decision: str

# Mapping PD → credit score
def map_pd_to_score(pd: float) -> float:
    return 300 + 550 * (1 - pd)

@app.post("/predict", response_model=RiskOutput)
def predict_risk(data: CustomerInput):
    df = pd.DataFrame([data.dict()])
    pd_prob = model.predict_proba(df)[:, 1][0]
    score = map_pd_to_score(pd_prob)
    decision = "APPROVE" if pd_prob < 0.5 else "REJECT"
    return RiskOutput(
        probability_of_default=pd_prob,
        credit_score=score,
        decision=decision
    )
