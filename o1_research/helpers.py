import tiktoken
import textwrap

def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding)
    tokens = encoding.encode(text)
    return len(tokens)

def format_prompt(prompt: str) -> str:
    return textwrap.dedent(prompt).strip()

def calc_cost(input_token_count: int, output_token_count: int, price_per_mill_input: float, price_per_mill_output: float) -> float:
    return ((input_token_count/1_000_000) * price_per_mill_input + (output_token_count/1_000_000) * price_per_mill_output)

class MBPPRequestId:
    @staticmethod
    def create_request_id(task_id: int) -> str:
        return f"mbpp_{task_id}"

    @staticmethod
    def extract_request_id(request_id: str) -> int:
        return int(request_id.split("_")[-1])
