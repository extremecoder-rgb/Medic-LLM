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
        import os
        # Check if mock mode is forced via environment variable
        self.is_mock = os.getenv("MOCK_LLM", "false").lower() == "true"
        
        if self.is_mock:
            print("MOCK_LLM is enabled. Running in simulated (mock) inference mode.")
            self.is_fine_tuned = True
            return

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        try:
            print(f"Loading base model: {base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )

            try:
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                print(f"Loaded LoRA adapter from {adapter_path}")
                self.is_fine_tuned = True
            except Exception as peft_err:
                print(f"No LoRA adapter found or error loading: {peft_err}. Using base model.")
                self.is_fine_tuned = False

            if device == "cpu":
                self.model = self.model.float()
            self.model.eval()
            self.is_mock = False
        except Exception as e:
            print(f"Failed to load Hugging Face model: {e}")
            print("Falling back to simulated (mock) inference mode.")
            self.is_mock = True
            self.is_fine_tuned = True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        if self.is_mock:
            return self._generate_mock_response(prompt)

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

    def _generate_mock_response(self, prompt: str) -> str:
        # Parse instruction and input from the structured prompt
        instruction = ""
        input_data = ""
        
        if "### Instruction:" in prompt:
            parts = prompt.split("### Instruction:")
            if len(parts) > 1:
                inst_part = parts[1]
                if "### Input:" in inst_part:
                    inst_subparts = inst_part.split("### Input:")
                    instruction = inst_subparts[0].strip()
                    if "### Response:" in inst_subparts[1]:
                        input_data = inst_subparts[1].split("### Response:")[0].strip()
                    else:
                        input_data = inst_subparts[1].strip()
                elif "### Response:" in inst_part:
                    instruction = inst_part.split("### Response:")[0].strip()
                else:
                    instruction = inst_part.strip()

        # Parse patient context from input_data
        patient_id = "unknown"
        age = "unknown"
        gender = "unknown"
        conditions = []
        medications = []
        observations = []
        allergies = []
        encounters = []
        
        patient_info = input_data
        medical_context = ""
        if "Medical Context:" in input_data:
            parts = input_data.split("Medical Context:")
            patient_info = parts[0].strip()
            medical_context = parts[1].strip()
        elif "Relevant Guidelines:" in input_data:
            parts = input_data.split("Relevant Guidelines:")
            patient_info = parts[0].strip()
            medical_context = parts[1].strip()

        for line in patient_info.split("\n"):
            line = line.strip()
            if line.startswith("Patient ID:"):
                patient_id = line.replace("Patient ID:", "").strip()
            elif line.startswith("Age:"):
                age = line.replace("Age:", "").strip()
            elif line.startswith("Gender:"):
                gender = line.replace("Gender:", "").strip()
            elif line.startswith("Conditions:"):
                conditions = [c.strip() for c in line.replace("Conditions:", "").split(",") if c.strip()]
            elif line.startswith("Medications:"):
                medications = [m.strip() for m in line.replace("Medications:", "").split(",") if m.strip()]
            elif line.startswith("Observations:"):
                observations = [o.strip() for o in line.replace("Observations:", "").split(",") if o.strip()]
            elif line.startswith("Allergies:"):
                allergies = [a.strip() for a in line.replace("Allergies:", "").split(",") if a.strip()]
            elif line.startswith("Recent Encounters:"):
                encounters = [e.strip() for e in line.replace("Recent Encounters:", "").split(",") if e.strip()]

        # Generate output based on instruction type
        is_summary = "summary" in instruction.lower() or "summarize" in instruction.lower()
        
        # Extract matching/relevant guidelines
        matched_guidelines = []
        if medical_context:
            guideline_items = []
            for chunk in medical_context.split("\n\n"):
                for line in chunk.split("\n"):
                    line = line.strip()
                    if line and (line.startswith("-") or line.startswith("*") or len(line) > 20):
                        guideline_items.append(line.strip("-* "))
            
            keywords = [w.lower() for w in instruction.split() if len(w) > 3]
            for item in guideline_items:
                if any(kw in item.lower() for kw in keywords):
                    if item not in matched_guidelines:
                        matched_guidelines.append(item)
            # Fill up with general guidelines if none matched specifically
            for item in guideline_items:
                if len(matched_guidelines) < 3 and item not in matched_guidelines:
                    matched_guidelines.append(item)

        if is_summary:
            summary_lines = [
                "### 📋 Patient Clinical Summary (Simulated Support)",
                f"**Demographic Profile:**",
                f"- **Patient ID:** `{patient_id}`",
                f"- **Age / Gender:** {age} years / {gender.capitalize()}",
            ]
            
            if conditions:
                summary_lines.append(f"- **Active Conditions:** {', '.join(conditions)}")
            if medications:
                summary_lines.append(f"- **Current Medications:** {', '.join(medications)}")
            if observations:
                summary_lines.append(f"- **Clinical Observations:** {', '.join(observations)}")
            if allergies:
                summary_lines.append(f"- **Allergies & Sensitivities:** {', '.join(allergies)}")
                
            summary_lines.append("\n**Clinical Care Guidelines & Key Interventions:**")
            
            if allergies:
                summary_lines.append(f"- ⚠️ **Allergy Warning:** Cross-reference all new prescriptions against known allergy: *{', '.join(allergies)}*.")
            
            # Add general guideline summaries
            for g in matched_guidelines[:3]:
                summary_lines.append(f"- {g}")
                
            # Heuristics based on conditions
            for cond in conditions:
                cond_l = cond.lower()
                if "diabetes" in cond_l:
                    summary_lines.append("- **Diabetes Management:** Monitor HbA1c periodically (target < 7.0%). Focus on diabetic nephropathy, retinopathy, and cardiovascular risk factors.")
                if "hypertension" in cond_l:
                    summary_lines.append("- **Blood Pressure Control:** Target BP < 130/80 mmHg. Lifestyle modifications (DASH diet, exercise) and pharmacological compliance are critical.")
                if "asthma" in cond_l:
                    summary_lines.append("- **Asthma Control:** Review inhaler technique and trigger avoidance. Recommend peak flow monitoring and asthma action plan.")
                if "heart failure" in cond_l:
                    summary_lines.append("- **Heart Failure Protocol:** Daily weight monitoring (report gain of >3 lbs/day). Optimize beta-blocker and ACE-i/ARB dosages.")

            return "\n".join(summary_lines)
            
        else:
            # Query answer
            ans_lines = [
                "### 🩺 Clinical Decision Support Response (Simulated Support)",
                f"**Clinical Context for Patient `{patient_id}` ({age}yo {gender}):**"
            ]
            
            # Analyze query for specific questions
            q_lower = instruction.lower()
            
            ans_lines.append("\n**Evidence-Based Assessment:**")
            if matched_guidelines:
                for g in matched_guidelines[:3]:
                    ans_lines.append(f"- *Guideline:* {g}")
            else:
                ans_lines.append("- No direct guidelines found in local documents database. Please consult primary clinical literature.")

            ans_lines.append("\n**Actionable Recommendations:**")
            
            # Answer specifics based on question keywords
            if "monitor" in q_lower or "check" in q_lower or "test" in q_lower:
                ans_lines.append("- **Routine Monitoring:** Establish baseline and regular intervals for diagnostic panels (e.g., HbA1c for diabetics, serum potassium/creatinine for ACE-inhibitors).")
                ans_lines.append("- **Vital Signs:** Instruct the patient on home monitoring logs (blood pressure or glucose logs).")
            elif "medication" in q_lower or "drug" in q_lower or "treatment" in q_lower:
                ans_lines.append("- **Therapy Compliance:** Confirm patient understands drug indications, correct dosing, and potential side effects.")
                ans_lines.append("- **Interaction Audit:** Perform a comprehensive drug-reconciliation, especially if adding new agents.")
            elif "allergy" in q_lower or "allergies" in q_lower:
                if allergies:
                    ans_lines.append(f"- **Active Sensitivities:** Ensure Electronic Health Record (EHR) flag is set for: {', '.join(allergies)}.")
                else:
                    ans_lines.append("- No drug or food allergies documented in the current patient profile.")
            else:
                ans_lines.append(f"- Integrate clinical findings with the patient's conditions ({', '.join(conditions) if conditions else 'none documented'}).")
                ans_lines.append("- Follow patient-centered care and review guidelines for appropriate next steps.")
                
            return "\n".join(ans_lines)

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
