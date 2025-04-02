import aiohttp
import asyncio
import logging

class SLAIClient:
    def __init__(self, api_url="http://localhost:8000", auth_token: str = None):
        self.api_url = api_url
        self.auth_token = auth_token
        self.headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        self.session = aiohttp.ClientSession()

    async def send_state(self, npc_id: str, state: dict, environment: dict, task_type="default") -> dict:
        url = f"{self.api_url}/slai/decide/"
        payload = {
            "npc_id": npc_id,
            "current_state": state,
            "environment": environment,
            "task_type": task_type
        }
        async with self.session.post(url, json=payload, headers=self.headers) as resp:
            if resp.status != 200:
                raise Exception(f"[{npc_id}] SLAI error: {resp.status} - {await resp.text()}")
            return await resp.json()

    async def send_feedback(self, npc_id: str, feedback: dict) -> dict:
        url = f"{self.api_url}/slai/feedback/"
        payload = {
            "npc_id": npc_id,
            "feedback": feedback
        }
        async with self.session.post(url, json=payload, headers=self.headers) as resp:
            if resp.status != 200:
                raise Exception(f"[{npc_id}] Feedback error: {resp.status} - {await resp.text()}")
            return await resp.json()

    async def close(self):
        await self.session.close()

# Example usage: run this only for testing or from a script
#if __name__ == "__main__":
#    async def main():
#        client = SLAIClient(auth_token="your_token_here")

#        tasks = []
#        for npc_id in ["npc1", "npc2", "npc3"]:
#            tasks.append(client.send_state(
#                npc_id,
#                state={"health": 80, "enemy_visible": True},
#                environment={"weather": "rain", "time": "night"},
#                task_type="combat"
#            ))

#        results = await asyncio.gather(*tasks, return_exceptions=True)
#        for npc, result in zip(["npc1", "npc2", "npc3"], results):
#            print(f"{npc}: {result}")

#        await client.close()

#    asyncio.run(main())
