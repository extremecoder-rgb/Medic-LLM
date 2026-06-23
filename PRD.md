# MediTwin

### FHIR-Aware Clinical RAG Assistant

---

# Overview

MediTwin is a healthcare AI assistant that combines structured clinical data, retrieval-augmented generation (RAG), and a fine-tuned medical language model to help clinicians generate patient summaries and answer clinical questions.

The goal is to demonstrate practical skills in:

* Python
* PyTorch
* Hugging Face Transformers
* PEFT / LoRA
* TRL
* Retrieval-Augmented Generation
* HL7 FHIR
* Healthcare AI workflows
* Experiment tracking
* API deployment

This project is intentionally scoped for a research internship and focuses on depth rather than attempting to recreate a full hospital AI platform.

---

# Problem

Clinical information is often fragmented across:

* Patient records
* Diagnoses
* Observations
* Clinical guidelines
* Medical literature

Clinicians spend significant time gathering information before making decisions.

MediTwin aims to:

1. Ingest structured patient data.
2. Retrieve relevant medical knowledge.
3. Generate grounded clinical responses.

---

# Goals

## Primary Goals

* Build a healthcare-focused LLM application.
* Demonstrate LoRA fine-tuning.
* Demonstrate TRL usage.
* Demonstrate healthcare data understanding.
* Demonstrate PyTorch competency.
* Demonstrate ML engineering skills.

## Non-Goals

* Training a foundation model from scratch.
* Real hospital deployment.
* Distributed training at scale.
* Regulatory compliance.

---

# User Stories

### Clinical Summary

As a clinician,

I want to provide patient information

So that I can receive a concise patient summary.

---

### Clinical Question Answering

As a clinician,

I want to ask questions about a patient

So that I receive evidence-grounded responses.

---

### Guideline Retrieval

As a clinician,

I want the system to retrieve relevant context

So that answers are grounded in medical knowledge.

---

# Functional Requirements

## Feature 1 — FHIR Patient Ingestion

### Input

FHIR-inspired JSON.

Example:

```json
{
  "patient": {
    "age": 65,
    "gender": "male"
  },
  "conditions": [
    "Type 2 Diabetes",
    "Hypertension"
  ],
  "medications": [
    "Metformin"
  ]
}
```

### Responsibilities

* Validate schema
* Parse records
* Normalize data
* Generate patient context

### Output

Structured patient context object.

---

## Feature 2 — Medical Knowledge Retrieval

### Responsibilities

* Store medical documents
* Generate embeddings
* Retrieve relevant context

### Sources

* MedQA
* PubMedQA
* Public medical guidelines

### Output

Top-k relevant passages.

---

## Feature 3 — Clinical LLM

### Base Model

Choose one:

* Qwen 2.5 3B Instruct
* Llama 3.2 3B Instruct

### Fine-Tuning

Use:

* Transformers
* PEFT
* LoRA
* TRL SFTTrainer

### Tasks

* Clinical QA
* Patient summarization
* Medical reasoning

---

## Feature 4 — Clinical Chat Interface

User can:

* Upload patient JSON
* Ask questions
* View generated answers

Example:

"What should be monitored for this patient?"

---

## Feature 5 — Evaluation Suite

Metrics:

### Generation

* ROUGE
* BLEU
* Exact Match

### Retrieval

* Recall@K

### Qualitative

* Hallucination examples
* Failure cases
* Error analysis

Output:

Evaluation report.

---

## Feature 6 — Experiment Tracking

Track:

* Learning rate
* Batch size
* Training loss
* Validation loss
* Checkpoints

Tool:

Weights & Biases

---

## Feature 7 — REST API

Endpoints:

### POST /patient

Store patient context.

### POST /query

Ask clinical question.

### POST /summary

Generate patient summary.

---

# Technical Architecture

## Frontend

Gradio

Responsibilities:

* Patient upload
* Chat interface
* Result visualization

---

## Backend

FastAPI

Responsibilities:

* Request handling
* Retrieval pipeline
* Model orchestration

---

## Retrieval Layer

FAISS

Responsibilities:

* Embedding storage
* Similarity search

---

## Model Layer

PyTorch

Transformers

PEFT

TRL

Responsibilities:

* Fine-tuning
* Inference
* Evaluation

---

# PyTorch Usage

PyTorch will be used for:

### Fine-Tuning

* Forward pass
* Backpropagation
* Gradient updates

### LoRA Training

* Adapter training
* Optimization

### Inference

* Tensor operations
* GPU execution

### Evaluation

* Batch processing
* Metric computation

---

# Repository Structure

```text
meditwin/

├── app/
│   ├── api.py
│   ├── fhir_parser.py
│   ├── rag.py
│   ├── inference.py
│   └── schemas.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   ├── dataset.py
│   └── config.py
│
├── data/
│
├── notebooks/
│
├── configs/
│
├── tests/
│
├── gradio_app.py
│
├── pyproject.toml
│
├── uv.lock
│
├── README.md
│
└── PRD.md
```

---

# Development Environment

Package Manager:

uv

Examples:

```bash
uv init

uv venv

source .venv/bin/activate

uv add torch transformers datasets peft trl accelerate

uv add fastapi uvicorn gradio

uv add faiss-cpu sentence-transformers

uv add wandb
```

---

# Success Criteria

The project is successful if:

* FHIR records are parsed correctly.
* Medical documents are retrieved accurately.
* LoRA fine-tuning completes successfully.
* Clinical questions receive grounded responses.
* Experiments are tracked.
* API endpoints function correctly.
* A complete end-to-end demo is available.

---

# Stretch Goal (Optional)

Add a lightweight medical image workflow:

Input:

* Chest X-ray

Model:

* CLIP

Output:

* Image findings added to clinical context.

This is optional and only pursued after the core system is complete.

---

# Deliverables

1. GitHub Repository
2. Technical README
3. Architecture Diagram
4. Trained LoRA Adapter
5. Evaluation Report
6. Demo Video
7. API Documentation

---

# Expected Internship Signal

After reviewing this project, an ML research engineer should conclude:

"This candidate understands modern LLM workflows, PEFT/LoRA training, healthcare data structures, retrieval systems, PyTorch fundamentals, and can independently build an end-to-end ML application."
