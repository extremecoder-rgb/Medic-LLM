import json
import os
from typing import List, Dict
from datasets import Dataset


def load_kaggle_medical_qa(file_path: str) -> List[Dict[str, str]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    for item in raw_data:
        question = item.get("input", "")
        answer = item.get("answer_chatgpt") or item.get("answer_chatdoctor") or item.get("answer_icliniq") or ""
        if question and answer:
            sample = {"instruction": question, "input": "", "output": answer}
            data.append(sample)
    return data


def load_training_data(file_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    print(f"Loading dataset from {file_path}...")
    data = load_kaggle_medical_qa(file_path)
    print(f"Loaded {len(data)} training samples")
    return data


def format_instruction(sample: Dict[str, str]) -> str:
    text = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
    return text


def format_dataset(data: List[Dict[str, str]]) -> Dataset:
    formatted = []
    for sample in data:
        formatted.append({"text": format_instruction(sample)})
    return Dataset.from_list(formatted)
