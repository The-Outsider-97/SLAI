using UnityEngine;
using WebSocketSharp;
using System;
using System.Collections.Generic;

public class SLAIWebSocketClient : MonoBehaviour
{
    public string wsUrl = "ws://localhost:8765"; // default WebSocket endpoint
    private WebSocket ws;

    public Action<string> OnMessageReceived;

    public void Connect()
    {
        ws = new WebSocket(wsUrl);
        ws.OnMessage += (sender, e) =>
        {
            Debug.Log("SLAI Message: " + e.Data);
            OnMessageReceived?.Invoke(e.Data);
        };
        ws.Connect();
    }

    public void SendState(string npcId, Dictionary<string, object> state, Dictionary<string, object> env, string taskType = "default")
    {
        var payload = new Dictionary<string, object>
        {
            { "npc_id", npcId },
            { "current_state", state },
            { "environment", env },
            { "task_type", taskType }
        };
        string json = JsonUtility.ToJson(new Wrapper(payload));
        ws.Send(json);
    }

    public void Close() => ws?.Close();

    [Serializable]
    private class Wrapper
    {
        public Dictionary<string, object> data;
        public Wrapper(Dictionary<string, object> dict) => data = dict;
    }
}
