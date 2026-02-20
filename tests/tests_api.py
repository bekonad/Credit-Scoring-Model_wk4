import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    payload = {
        "total_amount": 5000.0,
        "avg_amount": 1000.0,
        "transaction_count": 10,
        "std_amount": 500.0,
        "avg_hour": 12.0,
        "avg_day": 15.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "credit_score" in data
    assert "decision" in data
    assert data["mode"] == "demo (fallback heuristic)"

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "mode" in response.json()