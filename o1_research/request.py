from litellm import completion
from openai import OpenAI
from pydantic import BaseModel
import os
from typing import List, Dict

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

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

# Only supports latest GPT-4o and GPT-4o Mini models from OpenAI
def make_request_structured_output(
    model: str,
    messages: List[Dict[str, str]],
    response_format: BaseModel,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> str:
    client = OpenAI()

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return completion.choices[0].message.parsed.dict()
