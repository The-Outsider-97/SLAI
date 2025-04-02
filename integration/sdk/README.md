### How to Use in Unity

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
