import uvicorn
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.collaboration.collaboration_manager import CollaborationManager
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, FastAPI, Request, HTTPException
from pydantic import BaseModel

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != "your_token_here":  # Replace with real validation
        raise HTTPException(status_code=401, detail="Invalid token")
        
app = FastAPI(title="SLAI API Gateway", version="1.0")

# Initialize the Collaboration Manager
collab_manager = CollaborationManager()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SLAI-API")

# Define schema for external input
class StateInput(BaseModel):
    npc_id: str
    current_state: dict
    environment: dict
    task_type: str = "default"

class FeedbackInput(BaseModel):
    npc_id: str
    feedback: dict  # reward, correction, user action, etc.

@app.post("/slai/decide/")
async def decide(state: StateInput):
    try:
        logger.info(f"Received state for NPC {state.npc_id}")
        action = collab_manager.process_state(state.current_state, state.environment, state.task_type)
        return {"npc_id": state.npc_id, "action": action}
    except Exception as e:
        logger.error(f"Error during decision for NPC {state.npc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/slai/feedback/")
async def feedback(feedback: FeedbackInput):
    try:
        logger.info(f"Received feedback for NPC {feedback.npc_id}")
        collab_manager.process_feedback(feedback.npc_id, feedback.feedback)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing feedback for NPC {feedback.npc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "SLAI API Gateway is running."}

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run("integration.api_gateway:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    run_api_server()
