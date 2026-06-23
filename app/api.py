from fastapi import FastAPI, HTTPException
from app.schemas import (
    PatientData, QueryRequest, SummaryRequest,
    QueryResponse, SummaryResponse, ErrorResponse
)
from app.fhir_parser import FHIRParser
from app.rag import MedicalRAG
from app.inference import ClinicalInference

app = FastAPI(title="MediTwin API", version="1.0.0")
parser = FHIRParser()
rag = MedicalRAG()

try:
    rag.build_index("data/medical_docs")
except Exception as e:
    print(f"RAG index not built: {e}")

try:
    model = ClinicalInference()
except Exception as e:
    print(f"Model not loaded: {e}")
    model = None


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/patient", response_model=dict)
def parse_patient(data: PatientData):
    try:
        context = parser.parse(data.model_dump())
        return {"patient_id": context.patient_id, "context": context.to_prompt()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        chunks = rag.retrieve(request.question, k=3)

        context_prompt = ""
        if request.patient_data:
            patient = parser.parse(request.patient_data.model_dump())
            context_prompt = patient.to_prompt()

        combined_context = f"Patient:\n{context_prompt}\n\nMedical Context:\n" + "\n\n".join(chunks)

        answer = model.answer_clinical_question(request.question, combined_context)
        return QueryResponse(answer=answer, retrieved_chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summary", response_model=SummaryResponse)
def summary(request: SummaryRequest):
    try:
        patient = parser.parse(request.patient_data.model_dump())
        patient_text = patient.to_prompt()

        chunks = rag.retrieve(f"Patient summary for {patient_text[:100]}", k=2)
        context = f"Patient:\n{patient_text}\n\nRelevant Guidelines:\n" + "\n\n".join(chunks)

        response = model.summarize_patient(context)
        return SummaryResponse(summary=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
