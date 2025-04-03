using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System.Text;
using System;

public class SLAIUnityClient : MonoBehaviour
{
    [Header("SLAI API Settings")]
    public string apiUrl = "http://localhost:8000";
    public string authToken = "your_token_here";

    // Public method to request a decision from SLAI
    public void RequestNPCDecision(string npcId, Dictionary<string, object> currentState, Dictionary<string, object> environment, string taskType, Action<string> onSuccess, Action<string> onError)
    {
        string endpoint = "/slai/decide/";
        Dictionary<string, object> payload = new Dictionary<string, object>
        {
            { "npc_id", npcId },
            { "current_state", currentState },
            { "environment", environment },
            { "task_type", taskType }
        };

        StartCoroutine(PostJson(apiUrl + endpoint, payload, onSuccess, onError));
    }

    // Public method to send feedback to SLAI
    public void SendFeedback(string npcId, Dictionary<string, object> feedbackData, Action<string> onSuccess, Action<string> onError)
    {
        string endpoint = "/slai/feedback/";
        Dictionary<string, object> payload = new Dictionary<string, object>
        {
            { "npc_id", npcId },
            { "feedback", feedbackData }
        };

        StartCoroutine(PostJson(apiUrl + endpoint, payload, onSuccess, onError));
    }

    // Generic JSON POST method with auth
    private IEnumerator PostJson(string url, Dictionary<string, object> payload, Action<string> onSuccess, Action<string> onError)
    {
        string jsonData = JsonUtility.ToJson(new Wrapper(payload));
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);

        UnityWebRequest request = new UnityWebRequest(url, "POST");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");
        request.SetRequestHeader("Authorization", $"Bearer {authToken}");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            onSuccess?.Invoke(request.downloadHandler.text);
        }
        else
        {
            onError?.Invoke($"Error: {request.responseCode} - {request.error}\n{request.downloadHandler.text}");
        }
    }

    // Unity can't directly serialize Dictionary<string, object> → helper
    [Serializable]
    private class Wrapper
    {
        public Dictionary<string, object> data;

        public Wrapper(Dictionary<string, object> dict)
        {
            data = dict;
        }
    }
}
