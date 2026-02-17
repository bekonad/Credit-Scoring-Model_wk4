import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

mlflow.set_experiment("Credit_Risk_Week12_Capstone")

# Load final modeling data
df = pd.read_csv("data/processed/processed.csv")

# Target and features
target = "is_high_risk"
exclude = [target, "CustomerId"]
features = [col for col in df.columns if col not in exclude]

X = df[features]
y = df[target]

print("Features used:", features)
print("Target distribution:\n", y.value_counts(normalize=True))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

with mlflow.start_run(run_name="RF_Real_Relog_Week12") as run:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    probs = rf.predict_proba(X_test)[:, 1]
    preds = rf.predict(X_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, probs),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    }

    mlflow.log_metrics(metrics)
    mlflow.log_params({"n_estimators": 200, "max_depth": 12})
    mlflow.sklearn.log_model(rf, artifact_path="model")  # correct path

    print("\n" + "="*60)
    print("Model logged successfully!")
    print("New Run ID:", run.info.run_id)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")