from fastapi import FastAPI, Request
from pydantic import BaseModel
from modules.deployment.model_deployer import ModelDeployer

app = FastAPI()
deployer = ModelDeployer("models/random_forest_model.pkl")
deployer.load()

class InferenceRequest(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(input: InferenceRequest):
    result = deployer.predict(input.features)
    return result

@app.get("/")
async def root():
    return {"status": "Model is live"}
