from pydantic import BaseModel
from typing import List, Optional


class PatientData(BaseModel):
    patient_id: Optional[str] = "unknown"
    patient: dict
    conditions: Optional[List[str]] = []
    medications: Optional[List[str]] = []
    observations: Optional[List[str]] = []
    allergies: Optional[List[str]] = []
    encounters: Optional[List[str]] = []


class QueryRequest(BaseModel):
    question: str
    patient_data: Optional[PatientData] = None


class SummaryRequest(BaseModel):
    patient_data: PatientData


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: Optional[List[str]] = []


class SummaryResponse(BaseModel):
    summary: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
