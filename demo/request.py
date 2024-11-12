from litellm import completion
from openai import OpenAI
from pydantic import BaseModel
import os
from typing import List, Dict

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['ANTHROPIC_API_KEY'] = os.getenv("ANTHROPIC_API_KEY")
os.environ['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")
os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['MISTRAL_API_KEY'] = os.getenv("MISTRAL_API_KEY")

# Helper function to extract JSON from a response string that may contain a ```json ... ``` wrapper.
def extract_json_from_response(response: str) -> str:
    """
    Extracts JSON content from a response string that may contain a ```json ... ``` wrapper.
    """
    import re
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return response  # Return the original response if no wrapper is found

def make_request(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    max_tokens: int = 512,
    json_mode: bool = False
) -> str:
    response = completion(
        model=model, 
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    res = response.choices[0].message.content
    if json_mode:
        return extract_json_from_response(res)
    else:
        return res

# Only supports latest GPT-4o and GPT-4o Mini models from OpenAI
def make_request_structured_output(
    model: str,
    messages: List[Dict[str, str]],
    response_format: BaseModel,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> Dict:
    client = OpenAI()

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return completion.choices[0].message.parsed.dict()
