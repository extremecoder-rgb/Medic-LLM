from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.schemas import (
    PatientData, QueryRequest, SummaryRequest,
    QueryResponse, SummaryResponse, ErrorResponse
)
from app.fhir_parser import FHIRParser
from app.rag import MedicalRAG
from app.inference import ClinicalInference
import os
import json

app = FastAPI(title="MediTwin API", version="1.0.0")

# Enable CORS for local testing on different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    is_mock = getattr(model, "is_mock", True) if model else True
    is_fine_tuned = getattr(model, "is_fine_tuned", False) if model else False
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "is_mock": is_mock,
        "is_fine_tuned": is_fine_tuned
    }


@app.get("/patients", response_model=dict)
def get_patients():
    """Load and return all sample patient JSONs from the data directory."""
    patients = {}
    data_dir = "data"
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.startswith("patient_") and filename.endswith(".json"):
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        patients[filename] = json.load(f)
                except Exception as e:
                    print(f"Error loading patient file {filename}: {e}")
    return patients


@app.get("/guidelines", response_model=dict)
def get_guidelines():
    """Retrieve all plain text medical reference documents in the medical_docs directory."""
    guidelines = {}
    docs_dir = "data/medical_docs"
    if os.path.exists(docs_dir):
        for filename in os.listdir(docs_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(docs_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        title = filename.replace("_", " ").replace(".txt", "").title()
                        guidelines[title] = f.read()
                except Exception as e:
                    print(f"Error loading guideline file {filename}: {e}")
    return guidelines


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

        if model is None:
            raise HTTPException(status_code=500, detail="Inference model is not initialized.")

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

        if model is None:
            raise HTTPException(status_code=500, detail="Inference model is not initialized.")

        response = model.summarize_patient(context)
        return SummaryResponse(summary=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the single-page premium clinical dashboard."""
    dashboard_path = os.path.join("app", "static", "index.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # Fallback if UI is not generated yet
        return """
        <html>
            <head><title>MediTwin Loading</title></head>
            <body style="font-family:sans-serif; text-align:center; padding-top:50px;">
                <h1>MediTwin API is Running</h1>
                <p>Dashboard UI is currently initializing. Please refresh in a moment.</p>
            </body>
        </html>
        """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
