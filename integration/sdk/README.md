### How to use wrapper in Unity

- Attach the script to a GameObject in your scene.
- Call methods from your NPC script like this:

```C++
Dictionary<string, object> state = new Dictionary<string, object>
{
    { "health", 75 },
    { "enemyVisible", true }
};

Dictionary<string, object> env = new Dictionary<string, object>
{
    { "timeOfDay", "night" },
    { "weather", "storm" }
};

slaiClient.RequestNPCDecision("npc42", state, env, "patrol",
    onSuccess: (result) => Debug.Log("Action: " + result),
    onError: (err) => Debug.LogError(err));
```
---

This Unity project shows how to connect game NPCs to SLAI using WebSockets.

## Requirements
- Unity 2021.3 LTS or newer
- WebSocketSharp (place in Assets/Plugins or install manually)
- Python running `slai_ws_gateway.py`

## Setup
1. Open in Unity.
2. Add an empty GameObject with `SLAIManager`.
3. Add an NPC GameObject with `NPCController`.
4. Link scripts.
5. Start Python WebSocket backend on port 8765.
