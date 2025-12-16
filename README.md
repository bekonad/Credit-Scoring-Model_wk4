# Credit Risk Week 4 — Bati Bank BNPL Model
Author: Bereket Feleke

This project implements:
- Credit risk probability model using XGBoost
- Loan amount & tenure prediction using LightGBM
- Alternative-data-driven credit scoring
- Full DVC + MLflow MLOps pipeline

Project status:
- Repo initialized
- Environment ready
- Codebase to be added next
---

Credit scoring is a critical process in financial services, used to quantify the likelihood that a borrower will default on a loan. In the context of Bati Bank’s partnership with an eCommerce platform, the goal is to evaluate customers for a buy-now-pay-later service using alternative transaction data.

### Basel II and Model Interpretability

The Basel II Capital Accord emphasizes that banks must measure and manage credit risk in a consistent, well-documented manner. This regulatory framework drives the need for interpretable models because financial institutions must justify credit decisions to regulators. Transparent models, such as logistic regression with Weight of Evidence (WoE) encoding, allow clear insights into which features drive predictions and facilitate compliance with Basel II standards.

### Proxy Variable Necessity

Our dataset lacks an explicit "default" label. To address this, we define a proxy variable that identifies high-risk customers based on behavioral patterns, such as Recency, Frequency, and Monetary (RFM) metrics. This proxy allows the model to predict potential defaults, but it introduces business risk: misclassification could lead to denying credit to creditworthy customers or extending credit to high-risk individuals. Hence, careful clustering and analysis are essential to minimize these risks.

### Model Trade-offs

There are trade-offs between simple, interpretable models and complex, high-performance models:

* **Simple models (e.g., Logistic Regression with WoE)**:

  * Pros: Highly interpretable, easier to document, regulatory-friendly.
  * Cons: May have lower predictive performance for complex patterns in alternative data.

* **Complex models (e.g., Gradient Boosting, Random Forests)**:

  * Pros: Can capture non-linear relationships and improve predictive accuracy.
  * Cons: Less interpretable, harder to explain to regulators, potential overfitting if not carefully tuned.

In a regulated financial context, the choice of model balances interpretability, regulatory compliance, and predictive performance. Often, an ensemble approach or hybrid workflow is used to maintain both accuracy and explainability.

---
# Feature Engineering (Task 3)
We transformed raw transactional data into customer-level features using
automated sklearn pipelines. Aggregate behavioral and time-based features
were created to ensure reproducibility and model readiness.

---
## Next Steps
- **Task 4 - Proxy Target:**  
  Define high-risk customers with RFM clustering; create `is_high_risk` target.

- **Task 5 - Model Training:**  
  Train and tune multiple models, track experiments with MLflow, evaluate metrics, write unit tests.

- **Task 6 - Deployment & CI/CD:**  
  Build FastAPI service, containerize with Docker, configure automated testing and CI/CD.