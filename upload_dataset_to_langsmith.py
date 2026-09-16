"""
Upload ground_truth.json to LangSmith as a Dataset.

Each entry in ground_truth.json becomes one example in the dataset:
  - inputs  : { "query": "...", "contexts": [...] }
  - outputs : { "answer": "...", "ground_truth": "..." }

Run once:
    python upload_dataset_to_langsmith.py
"""

import json
import os
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables from .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
DATASET_NAME      = "qa-rag-golden-dataset"
GT_FILE           = os.path.join(BASE_DIR, "ground_truth.json")

if not LANGSMITH_API_KEY or LANGSMITH_API_KEY == "your_langsmith_api_key_here":
    raise RuntimeError("LANGSMITH_API_KEY is not set in your .env file.")

# Load ground truth entries
with open(GT_FILE, "r", encoding="utf-8") as f:
    entries = json.load(f)

print(f"[DATASET] Loaded {len(entries)} entries from ground_truth.json")

# Connect to LangSmith
client = Client(api_key=LANGSMITH_API_KEY)

# Create or reuse the dataset
existing_datasets = [d.name for d in client.list_datasets()]
if DATASET_NAME in existing_datasets:
    print(f"[DATASET] '{DATASET_NAME}' already exists — adding examples to it.")
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
else:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Golden Q&A dataset for RAG evaluation — SmartSkills document"
    )
    print(f"[DATASET] Created new dataset: '{DATASET_NAME}'")

# Upload each entry as an example
inputs  = []
outputs = []

for entry in entries:
    inputs.append({
        "query":    entry.get("query", ""),
        "contexts": entry.get("contexts", [])
    })
    outputs.append({
        "answer":       entry.get("answer", ""),
        "ground_truth": entry.get("ground_truth", "")
    })

client.create_examples(
    inputs=inputs,
    outputs=outputs,
    dataset_id=dataset.id
)

print(f"[DATASET] Successfully uploaded {len(inputs)} examples to '{DATASET_NAME}'")
print(f"[DATASET] View it at: https://smith.langchain.com")
