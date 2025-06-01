import os
import requests
from typing import List, Optional
from langchain.llms.base import LLM
from pydantic import Field
import os
# —————— Config ——————
API_URL = "https://router.huggingface.co/together/v1/chat/completions"
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set your HF token in HUGGINGFACE_HUB_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct-Turbo"


class HuggingFaceRouterLLM(LLM):
    model: str = Field(default=DEFAULT_MODEL)
    temperature: float = Field(default=0.0)

    @property
    def _llm_type(self) -> str:
        return "huggingface-together"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        resp = requests.post(API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
