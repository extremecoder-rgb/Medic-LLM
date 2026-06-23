# MediTwin

FHIR-Aware Clinical RAG Assistant

## Architecture

```
Patient JSON → FHIR Parser → Patient Context
                                    ↓
User Question → RAG (FAISS) → Retrieved Docs
                                    ↓
                        LLM (Fine-tuned Qwen 2.5)
                                    ↓
                              Clinical Answer
```

## Setup

```bash
uv venv
.venv\Scripts\activate
uv sync
```

## Project Structure

```
├── app/
│   ├── api.py            # FastAPI endpoints
│   ├── fhir_parser.py    # FHIR JSON parser
│   ├── inference.py      # LLM inference
│   ├── rag.py            # RAG pipeline
│   └── schemas.py        # Pydantic models
├── training/
│   ├── config.py         # Training config
│   ├── dataset.py        # Dataset loading
│   ├── train.py          # Fine-tuning script
│   └── evaluate.py       # Evaluation metrics
├── data/                 # Sample data
├── gradio_app.py         # Web UI
└── pyproject.toml        # Dependencies
```

## Usage

### 1. Start API

```bash
uv run python -m app.api
```

### 2. Start UI

```bash
uv run python gradio_app.py
```

### 3. Ask Questions

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What should be monitored for diabetes?"}'
```

## Fine-Tuning

Run on Kaggle with GPU:

```bash
python training/train.py
```

## Evaluation

```bash
uv run python training/evaluate.py
```
