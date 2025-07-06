from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import faiss
import numpy as np

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nmfc-chatbot-ui-live.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

nmfc_items = [
    {"code": "156600", "description": "Plastic bottles, 1L, filled, boxed"},
    {"code": "114620", "description": "Electric motors, 2HP, palletized"},
    {"code": "79320", "description": "Flat-packed wooden desks, crated"}
]

def embed(text):
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return np.array(response['data'][0]['embedding'], dtype='float32')
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {str(e)}")

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
    try:
        description = (
            f"{data.filled} {data.material} bottles, {data.volume_per_unit} each, "
            f"containing {data.contents}, {data.packaging}, {data.total_weight}, size {data.dimensions}, "
            f"fragile: {data.fragile}"
        )

        query_vec = embed(description).reshape(1, -1)

        item_texts = [item["description"] for item in nmfc_items]
        item_vecs = np.array([embed(text) for text in item_texts])
        index = faiss.IndexFlatL2(item_vecs.shape[1])
        index.add(item_vecs)

        D, I = index.search(query_vec, k=1)
        match = nmfc_items[I[0][0]]

        return {
            "input_description": description,
            "matched_nmfc": match["code"],
            "matched_description": match["description"],
            "confidence_score": float(D[0][0])
        }

    except Exception as e:
        return {
            "error": "Failed to classify.",
            "details": str(e)
        }
