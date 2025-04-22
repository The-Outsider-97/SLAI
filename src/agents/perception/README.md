
```mermaid
graph TD  
  A[perception_agent.py] --> B[modules/transformer.py]  
  A --> C[encoders/audio_encoder.py]  
  C --> B  
  B --> D[modules/attention.py]  

end
```
