import mlflow.sklearn
import pandas as pd

MODEL_URI = "models:/m-f4846715846a446294434d5198253b2c"

def load_model():
    return mlflow.sklearn.load_model(MODEL_URI)

def predict(model, input_df):
    return model.predict_proba(input_df)[:, 1]

if __name__ == "__main__":
    print("Loading model...")
    model = load_model()

    sample_input = pd.DataFrame([{
        "total_amount": 1200.0,
        "avg_amount": 300.0,
        "transaction_count": 4,
        "std_amount": 150.0,
        "avg_hour": 14.0,
        "avg_day": 12.0
    }])

    prob = predict(model, sample_input)
    print("Predicted credit risk probability:", prob[0])
