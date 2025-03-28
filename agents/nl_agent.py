from transformers import pipeline

class LanguageAgent:
    def __init__(self):
        self.generator = pipeline("text-generation", model="gpt2")

    def generate_response(self, prompt: str) -> str:
        result = self.generator(prompt, max_length=100, num_return_sequences=1)
        return result[0]['generated_text']
