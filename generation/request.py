import json
from abc import ABC, abstractmethod

import openai
import requests
from huggingface_hub import InferenceClient


class Client(ABC):
    def __init__(self, model, max_new_tokens=2048, temperature=1.0, do_sample=False):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    @abstractmethod
    def textgen(self, prompt) -> str:
        pass


# test generation
class TGIClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = InferenceClient(model=self.model)

    def textgen(self, prompt, **kwargs) -> str:
        # def textgen(self, question, n=1, max_tokens=4096, temperature=0.0, system_msg=None):
        url = f"{self.model}/text_generation"
        headers = {"Content-Type": "application/json"}
        system_msg = "You are a highly skilled software engineer. Your task is to generate high-quality, efficient, and well-commented code based on the user's prompt. The user will provide a prompt describing a task or functionality, and you should respond with the appropriate code in the specified programming language. Include only the code in your response, and avoid any unnecessary explanations unless explicitly requested. Use best practices for coding, including meaningful variable names, comments, and proper formatting."
        data = {
            "question": prompt,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "system_msg": system_msg,
        }

        # response = requests.post(url, headers=headers, json=data)

        # response.raise_for_status()
        # return response.json().get("generated_texts", [])[0].strip('[]')

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            print("DEBUG response.json():", response.json())
            return response.json().get("generated_texts", [])[0].strip("[]")
        except Exception as e:
            return f"Model Generation Error: {type(e).__name__}"


if __name__ == "__main__":
    client = TGIClient(model="http://10.192.122.120:8080")

    output = client.textgen(prompt="Write a code for snake game")
    print(output)
