<div align="center">

# 🏥 MedicLLM

### FHIR-Aware Clinical RAG Assistant

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-FF5252?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

A clinical decision support system that combines **FHIR-compliant patient data parsing**, **retrieval-augmented generation (RAG)**, and a **fine-tuned Qwen 2.5** language model to deliver accurate, context-aware medical responses.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📋 **FHIR Patient Parser** | Validates and structures patient JSON into clinical context |
| 🔍 **RAG Pipeline** | FAISS-powered vector search over medical documents |
| 🧠 **Fine-tuned LLM** | Qwen 2.5 with LoRA adapters trained on medical QA data |
| 🌐 **REST API** | FastAPI endpoints for patient parsing, queries, and summaries |
| 🖥️ **Web Interface** | Gradio UI for interactive clinical question-answering |
| 📊 **Evaluation Suite** | ROUGE, BLEU, and Exact Match metrics |

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Patient JSON    │────▶│  FHIR Parser │────▶│ Patient Context  │
│  (FHIR-compat)  │     │              │     │                  │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                      │
┌─────────────────┐     ┌──────────────┐              │
│  User Question  │────▶│ RAG (FAISS)  │──────────────┤
│                 │     │              │              │
└─────────────────┘     └──────────────┘              ▼
                                         ┌────────────────────┐
                                         │  Qwen 2.5 (LoRA)   │
                                         │  Clinical Inference │
                                         └─────────┬──────────┘
                                                   │
                                                   ▼
                                         ┌────────────────────┐
                                         │  Clinical Answer    │
                                         └────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MedicLLM.git
cd MedicLLM

# Create virtual environment and install dependencies
uv venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux
uv sync
```

## 📁 Project Structure

```
MedicLLM/
├── app/
│   ├── api.py              # FastAPI REST endpoints
│   ├── fhir_parser.py      # FHIR patient data parser
│   ├── inference.py        # LLM inference with LoRA
│   ├── rag.py              # RAG pipeline (FAISS + embeddings)
│   └── schemas.py          # Pydantic request/response models
├── training/
│   ├── config.py           # Training hyperparameters
│   ├── dataset.py          # Medical QA dataset loader
│   ├── train.py            # Fine-tuning with TRL + PEFT
│   └── evaluate.py         # ROUGE / BLEU evaluation
├── data/
│   ├── medical_docs/       # Medical reference documents
│   ├── medical_qa.json     # Training dataset
│   └── patient_*.json      # Sample patient records
├── tests/
│   └── test_fhir_parser.py
├── gradio_app.py           # Interactive web UI
└── pyproject.toml          # Project configuration
```

## 💡 Usage

### 1. Start the API Server

```bash
uv run python -m app.api
```

> Server runs at `http://localhost:8000`

### 2. Launch the Web UI

```bash
uv run python gradio_app.py
```

> Gradio interface opens at `http://localhost:7860`

### 3. Query via API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Ask a Clinical Question:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What should be monitored for this patient?",
    "patient_data": {
      "patient_id": "P1001",
      "patient": {"age": 65, "gender": "male"},
      "conditions": ["Type 2 Diabetes", "Hypertension"],
      "medications": ["Metformin 500mg", "Lisinopril 10mg"],
      "observations": ["HbA1c: 7.2%"],
      "allergies": ["Penicillin"]
    }
  }'
```

**Generate Patient Summary:**
```bash
curl -X POST http://localhost:8000/summary \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "patient_id": "P1001",
      "patient": {"age": 65, "gender": "male"},
      "conditions": ["Type 2 Diabetes"],
      "medications": ["Metformin 500mg"]
    }
  }'
```

## 🧪 Fine-Tuning

Train on [Kaggle](https://www.kaggle.com/) with GPU acceleration:

```bash
cd training
python train.py
```

**Training Configuration** (`training/config.py`):

| Parameter | Value |
|-----------|-------|
| Base Model | `Qwen/Qwen2.5-3B-Instruct` |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Epochs | 3 |
| Learning Rate | 2e-4 |
| Batch Size | 4 |

## 📊 Evaluation

```bash
uv run python training/evaluate.py
```

Computes:
- **ROUGE-1 / ROUGE-2 / ROUGE-L** — n-gram overlap with reference answers
- **BLEU** — translation-quality scoring
- **Exact Match** — strict correctness metric

## 🔧 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/patient` | POST | Parse and validate patient JSON |
| `/query` | POST | Ask a clinical question with optional patient context |
| `/summary` | POST | Generate a clinical patient summary |

## 🛠️ Tech Stack

- **Model**: [Qwen 2.5 3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) with LoRA via PEFT
- **Training**: TRL SFTTrainer, Hugging Face Accelerate
- **RAG**: FAISS, Sentence Transformers (`all-MiniLM-L6-v2`)
- **API**: FastAPI + Uvicorn
- **UI**: Gradio
- **Evaluation**: rouge-score, NLTK BLEU

---

<div align="center">
  <sub>Built with care for clinical intelligence</sub>
</div>
