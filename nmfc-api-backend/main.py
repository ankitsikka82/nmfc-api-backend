from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import faiss
import numpy as np
import json

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Dummy NMFC items
nmfc_items = [
    {"code": "156600", "description": "Plastic bottles, 1L, filled, boxed"},
    {"code": "114620", "description": "Electric motors, 2HP, palletized"},
    {"code": "79320", "description": "Flat-packed wooden desks, crated"},
]

# Create FAISS index
index = faiss.IndexFlatL2(384)
item_texts = [item["description"] for item in nmfc_items]

# Load OpenAI Embedding once
def embed(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return np.array(response['data'][0]['embedding']).astype('float32')

# Vectorize item descriptions
item_vectors = np.array([embed(desc) for desc in item_texts])
index.add(item_vectors)

class InputData(BaseModel):
    material: str
    volume_per_unit: str
    filled: str
    contents: str = ""
    packaging: str
    fragile: str
    total_weight: str
    dimensions: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify")
def classify(data: InputData):
    description = (
        f"{data.filled} {data.material} bottles, {data.volume_per_unit} each, "
        f"containing {data.contents}, {data.packaging}, {data.total_weight}, size {data.dimensions}, "
        f"fragile: {data.fragile}"
    )

    embedding = embed(description)
    D, I = index.search(np.array([embedding]), k=1)
    match = nmfc_items[I[0][0]]

    return {
        "input_description": description,
        "matched_nmfc": match["code"],
        "matched_description": match["description"],
        "confidence_score": float(D[0][0])
    }
