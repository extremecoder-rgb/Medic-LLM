import gradio as gr
import json
import requests

API_URL = "http://localhost:8000"


def parse_patient(json_input: str) -> str:
    try:
        data = json.loads(json_input)
        resp = requests.post(f"{API_URL}/patient", json=data)
        if resp.status_code == 200:
            return resp.json().get("context", "Parsed successfully")
        return f"Error: {resp.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"Error: {str(e)}"


def ask_question(json_input: str, question: str) -> str:
    try:
        patient_data = json.loads(json_input) if json_input.strip() else None
        payload = {"question": question}
        if patient_data:
            payload["patient_data"] = patient_data
        resp = requests.post(f"{API_URL}/query", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("answer", "No answer")
            chunks = data.get("retrieved_chunks", [])
            if chunks:
                result += "\n\n--- Retrieved Context ---\n"
                for i, c in enumerate(chunks, 1):
                    result += f"\n[{i}] {c[:200]}..."
            return result
        return f"Error: {resp.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"Error: {str(e)}"


def generate_summary(json_input: str) -> str:
    try:
        data = json.loads(json_input)
        resp = requests.post(f"{API_URL}/summary", json={"patient_data": data})
        if resp.status_code == 200:
            return resp.json().get("summary", "No summary")
        return f"Error: {resp.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(title="MediTwin - Clinical RAG Assistant", theme="soft") as demo:
    gr.Markdown("# 🏥 MediTwin")
    gr.Markdown("FHIR-Aware Clinical RAG Assistant — Upload patient data and ask clinical questions.")

    sample_json = json.dumps({
        "patient_id": "P1001",
        "patient": {"age": 65, "gender": "male"},
        "conditions": ["Type 2 Diabetes", "Hypertension"],
        "medications": ["Metformin 500mg", "Lisinopril 10mg"],
        "observations": ["HbA1c: 7.2%", "BP: 140/90"],
        "allergies": ["Penicillin"],
        "encounters": ["2024-01-15: Follow-up"]
    }, indent=2)

    with gr.Tab("Patient Parser"):
        json_input = gr.Textbox(label="Patient JSON", lines=12, value=sample_json)
        parse_btn = gr.Button("Parse Patient")
        parse_output = gr.Textbox(label="Parsed Context", lines=8)
        parse_btn.click(parse_patient, inputs=json_input, outputs=parse_output)

    with gr.Tab("Clinical Query"):
        query_json = gr.Textbox(label="Patient JSON (optional)", lines=8, value=sample_json)
        question = gr.Textbox(label="Your Question", lines=2, placeholder="What should be monitored for this patient?")
        ask_btn = gr.Button("Ask Question")
        query_output = gr.Textbox(label="Answer", lines=10)
        ask_btn.click(ask_question, inputs=[query_json, question], outputs=query_output)

    with gr.Tab("Patient Summary"):
        summary_json = gr.Textbox(label="Patient JSON", lines=12, value=sample_json)
        summary_btn = gr.Button("Generate Summary")
        summary_output = gr.Textbox(label="Clinical Summary", lines=10)
        summary_btn.click(generate_summary, inputs=summary_json, outputs=summary_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
