import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Enable mock mode for testing to avoid downloading heavy models
os.environ["MOCK_LLM"] = "true"

from fastapi.testclient import TestClient
from app.api import app

# Create FastAPI TestClient
client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "is_mock" in data


def test_patients_endpoint():
    """Test presets retrieval endpoint."""
    response = client.get("/patients")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert len(data) > 0
    assert "patient_001.json" in data


def test_guidelines_endpoint():
    """Test guidelines retrieval endpoint."""
    response = client.get("/guidelines")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert len(data) > 0
    assert any("Diabetes" in title for title in data)


def test_parse_patient_endpoint():
    """Test parser endpoint with valid and invalid data."""
    # Valid patient data
    valid_payload = {
        "patient_id": "PTest",
        "patient": {"age": 45, "gender": "female"},
        "conditions": ["Asthma"],
        "medications": ["Albuterol"]
    }
    response = client.post("/patient", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "PTest"
    assert "Age: 45" in data["context"]
    assert "Gender: female" in data["context"]
    assert "Asthma" in data["context"]

    # Invalid patient data (missing age)
    invalid_payload = {
        "patient_id": "PTest",
        "patient": {"gender": "female"},
    }
    response = client.post("/patient", json=invalid_payload)
    assert response.status_code == 400  # FHIR validation error (Bad Request)


def test_query_endpoint():
    """Test clinical query (Q&A) endpoint."""
    payload = {
        "question": "What is the recommended treatment for patient's asthma?",
        "patient_data": {
            "patient_id": "PTest",
            "patient": {"age": 45, "gender": "female"},
            "conditions": ["Asthma"],
            "medications": ["Albuterol"]
        }
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["retrieved_chunks"]) > 0


def test_summary_endpoint():
    """Test clinical summary endpoint."""
    payload = {
        "patient_data": {
            "patient_id": "PTest",
            "patient": {"age": 45, "gender": "female"},
            "conditions": ["Asthma"],
            "medications": ["Albuterol"]
        }
    }
    response = client.post("/summary", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "Patient ID" in data["summary"] or "PTest" in data["summary"]


if __name__ == "__main__":
    print("Running API tests...")
    
    # Run tests sequentially
    tests = [
        ("test_health_endpoint", test_health_endpoint),
        ("test_patients_endpoint", test_patients_endpoint),
        ("test_guidelines_endpoint", test_guidelines_endpoint),
        ("test_parse_patient_endpoint", test_parse_patient_endpoint),
        ("test_query_endpoint", test_query_endpoint),
        ("test_summary_endpoint", test_summary_endpoint),
    ]
    
    passed_count = 0
    import traceback
    for name, test_func in tests:
        try:
            print(f"Running {name}...", end=" ")
            test_func()
            print("PASSED")
            passed_count += 1
        except Exception as e:
            print("FAILED")
            traceback.print_exc()
            
    print(f"\nAPI Test Results: {passed_count}/{len(tests)} tests passed.")
    if passed_count == len(tests):
        print("All API endpoints verified successfully!")
        sys.exit(0)
    else:
        sys.exit(1)
