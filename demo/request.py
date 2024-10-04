from litellm import completion
import os
from typing import List, Dict

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

def make_request(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    max_tokens: int = 512
) -> str:
    response = completion(
        model=model, 
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
