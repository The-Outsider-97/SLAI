import asyncio
import websockets
import json
from slai.core.collaboration import CollaborationManager

collab = CollaborationManager()

async def handler(websocket, path):
    async for message in websocket:
        try:
            data = json.loads(message)
            npc_id = data.get("npc_id", "unknown")
            state = data["current_state"]
            env = data["environment"]
            task_type = data.get("task_type", "default")
            action = collab.process_state(state, env, task_type)
            await websocket.send(json.dumps({"npc_id": npc_id, "action": action}))
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    asyncio.run(websockets.serve(handler, "0.0.0.0", 8765))
