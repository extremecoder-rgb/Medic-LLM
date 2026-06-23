import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class ClinicalInference:
    """Load fine-tuned model and generate clinical responses."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: str = "./results",
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading base model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        try:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"Loaded LoRA adapter from {adapter_path}")
            self.is_fine_tuned = True
        except Exception:
            print("No LoRA adapter found. Using base model.")
            self.is_fine_tuned = False

        if device == "cpu":
            self.model = self.model.float()
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        return response

    def answer_clinical_question(self, question: str, context: str = None) -> str:
        if context:
            prompt = f"""### Instruction:
{question}

### Input:
{context}

### Response:
"""
        else:
            prompt = f"""### Instruction:
{question}

### Response:
"""
        return self.generate(prompt)

    def summarize_patient(self, patient_context: str) -> str:
        prompt = f"""### Instruction:
Provide a concise clinical summary of this patient.

### Input:
{patient_context}

### Response:
"""
        return self.generate(prompt, max_new_tokens=512)
