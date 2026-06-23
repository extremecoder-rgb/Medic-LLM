import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.fhir_parser import FHIRParser, PatientContext


def test_valid_patient():
    """Test parsing a valid patient record."""
    parser = FHIRParser()

    data = {
        "patient": {
            "age": 65,
            "gender": "male"
        },
        "conditions": ["Type 2 Diabetes", "Hypertension"],
        "medications": ["Metformin", "Lisinopril"]
    }

    result = parser.parse(data)

    assert isinstance(result, PatientContext)
    assert result.age == 65
    assert result.gender == "male"
    assert result.conditions == ["Type 2 Diabetes", "Hypertension"]
    assert result.medications == ["Metformin", "Lisinopril"]


def test_missing_patient():
    """Test validation catches missing patient field."""
    parser = FHIRParser()

    data = {
        "conditions": ["Diabetes"]
    }

    errors = parser.validate(data)
    assert len(errors) > 0
    assert "patient" in errors[0]


def test_invalid_age():
    """Test validation catches invalid age."""
    parser = FHIRParser()

    data = {
        "patient": {
            "age": -5,
            "gender": "male"
        }
    }

    errors = parser.validate(data)
    assert len(errors) > 0


def test_to_prompt():
    """Test prompt generation."""
    parser = FHIRParser()

    data = {
        "patient": {
            "age": 65,
            "gender": "male"
        },
        "conditions": ["Type 2 Diabetes"],
        "medications": ["Metformin"]
    }

    result = parser.parse(data)
    prompt = result.to_prompt()

    assert "Age: 65" in prompt
    assert "Gender: male" in prompt
    assert "Type 2 Diabetes" in prompt
    assert "Metformin" in prompt


if __name__ == "__main__":
    test_valid_patient()
    test_missing_patient()
    test_invalid_age()
    test_to_prompt()
    print("All tests passed!")
