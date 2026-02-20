# Bati Bank BNPL Credit Risk Model – Week 12 Capstone

**Author**: Bereket Feleke  
**Email**: bereketfeleke003@gmail.com  
**Project**: Week 4 – Credit Risk Probability Model for Alternative Data (re-imagined as Week 12 capstone)  
**Goal**: Transform the Week 4 credit risk project into a production-grade portfolio piece demonstrating **reliability**, **transparency**, and **business impact** for finance recruiters.

**Live Dashboard** (real predictions + SHAP explainability):  
https://bekonad-credit-scoring-model-wk4.streamlit.app  

**Live API** (Task 6 – demo fallback mode):  
https://credit-scoring-model-wk4.onrender.com/docs

**Repository**: https://github.com/bekonad/Credit-Scoring-Model_wk4

## Business Problem

Bati Bank wants to offer buy-now-pay-later credit to e-commerce customers in emerging markets like Ethiopia. Many applicants lack formal credit history, leading to high default rates (revenue loss) or overly strict approvals (missed opportunities). Traditional scoring ignores behavioral signals from transaction data (Recency, Frequency, Monetary + time patterns).

**This solution** uses alternative data to predict default probability (`is_high_risk`) and recommend safe loan decisions.

**Why it matters**  
- Reduces defaults while approving more low-risk customers  
- Transparent predictions build trust with regulators and risk teams  
- Fast, interactive tool enables better decision-making

## Key Improvements (Week 12 Capstone)

- Interactive Streamlit dashboard for real-time risk scoring  
- Real Random Forest predictions (re-logged model to fix MLflow issues)  
- SHAP waterfall explainability – shows feature contributions to risk  
- Business recommendation logic (Approve / Moderate / Decline/Strict)  
- Task 6: FastAPI `/predict` endpoint + Dockerfile (deployed on Render)  
- Professional README with screenshots, metrics, live links

**Model Performance** (Random Forest – re-logged)

| Metric      | Value   | Interpretation                              |
|-------------|---------|---------------------------------------------|
| ROC-AUC     | 0.8292  | Excellent risk separation                   |
| Accuracy    | 75.17%  | Strong overall correctness                  |
| Precision   | 69.57%  | Low false positives                         |
| Recall      | 61.75%  | Good default detection                      |
| F1-Score    | 65.43%  | Balanced precision/recall                   |

**Example Dashboard Output**  
Default Probability: 38.42% → Credit Score: 638 → **High Risk – Decline/Strict**

## Screenshots

![Dashboard Overview](plots/dashboard_overview.png)  
*Interactive sliders + real-time results*

![High-Risk Prediction + SHAP Waterfall](plots/shap_waterfall_high_risk.png)  
*SHAP explains why model predicted 38.42% default risk*

![API Swagger UI](plots/api_docs.png)  
*Interactive API docs (/docs)*

![API /predict Response](plots/api_predict_response.png)  
*Live prediction response*

## How to Run Locally

```bash
git clone https://github.com/bekonad/Credit-Scoring-Model_wk4.git
cd Credit-Scoring-Model_wk4

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard.py

# Run API (optional)
uvicorn src.api.main:app --reload

Open:

Dashboard: http://localhost:8501
API Docs: http://127.0.0.1:8000/docs

Tech Stack

Streamlit (dashboard)
MLflow (model tracking & loading)
scikit-learn Random Forest
SHAP (explainability)
FastAPI (API)
Docker (containerization)
Pandas, Matplotlib, NumPy

Task 6 – API Deployment (FastAPI)
Branch: task6-api-deployment-week12 (merged into main)
Endpoint: POST /predict
Mode: Demo (fallback heuristic – real model loading requires remote MLflow server)
Sample Request
JSON{
  "total_amount": 5000.0,
  "avg_amount": 1000.0,
  "transaction_count": 10,
  "std_amount": 500.0,
  "avg_hour": 12.0,
  "avg_day": 15.0
}
Sample Response (live)
JSON{
  "default_probability": 0.3069,
  "credit_score": 681,
  "decision": "Approve",
  "risk_level": "Low-Moderate",
  "message": "High risk detected - strict terms or decline recommended (demo mode)"
}
Deployed on: Render.com (free tier, Docker-based)
Challenge Alignment (Week 12)
This project transforms Week 4 credit risk model into a finance-ready portfolio piece:

Reliability: Re-logged model + fallback heuristic
Transparency: SHAP waterfall explainability
Business impact: Risk-based decisions using alternative data
Advanced skills: Interactive dashboard + deployed API + Docker

Limitations

API uses demo fallback (real model loading requires remote MLflow server)
Model uses proxy target (is_high_risk) – real default labels would improve accuracy

Lessons Learned

Local MLflow runs do not work remotely → fallback ensures robustness
SHAP v0.20+ requires explicit class selection → fixed with [:, :, 1]
Demo mode allows deployment while real model shines in local dashboard

Repository: https://github.com/bekonad/Credit-Scoring-Model_wk4
Live Dashboard: https://bekonad-credit-scoring-model-wk4.streamlit.app
Live API: https://credit-scoring-model-wk4.onrender.com/docs
© 2026 Bereket Feleke – 10 Academy KAIM Week 12 Capstone