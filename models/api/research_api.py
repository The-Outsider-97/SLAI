from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, Dict
from fastapi.middleware.cors import CORSMiddleware

from models.research_model import ResearchModel

app = FastAPI(title="SLAI Research API", version="1.0.0")
model = ResearchModel()

# Allow CORS for local frontend or tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    prompt: str

@app.post("/research", tags=["Research"])
async def run_research(request: QueryRequest):
    result = model.run(request.prompt)
    return result

@app.get("/", tags=["Health"])
async def root():
    return {"status": "SLAI Research API is running"}
