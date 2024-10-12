import tiktoken
import textwrap

def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding)
    tokens = encoding.encode(text)
    return len(tokens)

def format_prompt(prompt: str) -> str:
    return textwrap.dedent(prompt).strip()
