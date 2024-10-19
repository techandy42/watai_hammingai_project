import tiktoken
import textwrap

def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding)
    tokens = encoding.encode(text)
    return len(tokens)

def format_prompt(prompt: str) -> str:
    return textwrap.dedent(prompt).strip()

class MBPPRequestId:
    @staticmethod
    def create_request_id(task_id: int) -> str:
        return f"mbpp_{task_id}"

    @staticmethod
    def extract_request_id(request_id: str) -> int:
        return int(request_id.split("_")[-1])
