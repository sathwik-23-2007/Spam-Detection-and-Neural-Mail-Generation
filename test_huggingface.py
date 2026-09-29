import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv(override=True)
api_key = os.getenv("HUGGINGFACE_API_KEY")

try:
    hf_client = InferenceClient(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": "Write a short formal email about a project update.",
        }
    ]
    response = hf_client.chat_completion(
        messages=messages, model="Qwen/Qwen2.5-72B-Instruct", max_tokens=50
    )
    print("Success! Response:\n", response.choices[0].message.content)
except Exception as e:
    print("Error:", e)
