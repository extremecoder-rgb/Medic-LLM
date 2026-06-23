from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class PatientContext:
    patient_id: str
    age: int
    gender: str
    conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    encounters: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        lines = [f"Patient ID: {self.patient_id}"]
        lines.append(f"Age: {self.age}")
        lines.append(f"Gender: {self.gender}")

        if self.conditions:
            lines.append(f"Conditions: {', '.join(self.conditions)}")
        if self.medications:
            lines.append(f"Medications: {', '.join(self.medications)}")
        if self.observations:
            lines.append(f"Observations: {', '.join(self.observations)}")
        if self.allergies:
            lines.append(f"Allergies: {', '.join(self.allergies)}")
        if self.encounters:
            lines.append(f"Recent Encounters: {', '.join(self.encounters)}")

        return "\n".join(lines)


class FHIRParser:
    REQUIRED_FIELDS = ["patient"]
    PATIENT_FIELDS = ["age", "gender"]

    def validate(self, data: dict) -> List[str]:
        errors = []

        for field_name in self.REQUIRED_FIELDS:
            if field_name not in data:
                errors.append(f"Missing required field: '{field_name}'")

        if "patient" in data:
            patient = data["patient"]
            if not isinstance(patient, dict):
                errors.append("'patient' must be a JSON object")
            else:
                for field_name in self.PATIENT_FIELDS:
                    if field_name not in patient:
                        errors.append(f"Missing patient field: '{field_name}'")

                if "age" in patient:
                    age = patient["age"]
                    if not isinstance(age, (int, float)):
                        errors.append(f"'age' must be a number, got {type(age).__name__}")
                    elif age < 0 or age > 150:
                        errors.append(f"'age' must be between 0 and 150, got {age}")

                if "gender" in patient:
                    gender = patient["gender"]
                    if gender.lower() not in ["male", "female", "other", "unknown"]:
                        errors.append(f"'gender' must be male/female/other/unknown, got '{gender}'")

        return errors

    def parse(self, data: dict) -> PatientContext:
        errors = self.validate(data)
        if errors:
            raise ValueError(f"Validation failed:\n" + "\n".join(errors))

        patient = data["patient"]

        return PatientContext(
            patient_id=data.get("patient_id", "unknown"),
            age=int(patient["age"]),
            gender=patient["gender"].lower(),
            conditions=data.get("conditions", []),
            medications=data.get("medications", []),
            observations=data.get("observations", []),
            allergies=data.get("allergies", []),
            encounters=data.get("encounters", []),
        )

    def parse_file(self, file_path: str) -> PatientContext:
        with open(file_path, "r") as f:
            data = json.load(f)
        return self.parse(data)
