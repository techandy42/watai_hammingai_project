import ast 
from datasets import load_dataset
from helpers import get_tested_function, remove_python_code_tags
from request import make_request
from prompts import CodegenPrompts
from typing import Optional
from helpers import split_assert_statements
import tiktoken

def get_function_name(row):
    function_name = get_tested_function(row["test_list"])
    return function_name

def get_codegen_prompt(row):
    text = row["text"]
    function_name = row["function_name"]
    test_list = row["test_list"]
    codegen_prompt = CodegenPrompts.get_codegen_prompt(text, function_name, test_list[0])
    return codegen_prompt

def add_comma_to_newline(row, column_name):
    return str(row[column_name]).replace("\n", ",\n")

def get_split_assert_statements(row, column_name):
    test_list = row[column_name]
    test_list = split_assert_statements(test_list)
    return test_list

def calculate_token_count(row, column_name, encoding):
    return len(encoding.encode(row[column_name]))

def calc_cost(read_cost: float, write_cost: float) -> float:
    ### Same as the codegen code
    dataset = load_dataset('google-research-datasets/mbpp')
    test_dataset = dataset["test"]
    df_test = test_dataset.to_pandas()

    df_test["test_list"] = df_test.apply(add_comma_to_newline, axis=1, column_name="test_list")
    df_test["challenge_test_list"] = df_test.apply(add_comma_to_newline, axis=1, column_name="challenge_test_list")
    df_test["test_list"] = df_test['test_list'].apply(ast.literal_eval)
    df_test["challenge_test_list"] = df_test['challenge_test_list'].apply(ast.literal_eval)
    df_test["test_list"] = df_test.apply(get_split_assert_statements, axis=1, column_name="test_list")
    df_test["challenge_test_list"] = df_test.apply(get_split_assert_statements, axis=1, column_name="challenge_test_list")

    df_test["function_name"] = df_test.apply(get_function_name, axis=1)
    df_test["codegen_prompt"] = df_test.apply(get_codegen_prompt, axis=1)
    ###

    encoding = tiktoken.get_encoding("cl100k_base")
    df_test["codegen_prompt_token_count"] = df_test.apply(calculate_token_count, axis=1, column_name="codegen_prompt", encoding=encoding)
    df_test["code_token_count"] = df_test.apply(calculate_token_count, axis=1, column_name="code", encoding=encoding)
    total_read_token_count = df_test["codegen_prompt_token_count"].sum()
    total_write_token_count = df_test["code_token_count"].sum()
    estimated_cost = (total_read_token_count / 1_000_000) * read_cost + (total_write_token_count / 1_000_000) * write_cost
    return total_read_token_count, total_write_token_count, estimated_cost

if __name__ == "__main__":
    MODEL_NAME = "gpt-4o-mini-2024-07-18"
    READ_COST = 0.150
    WRITE_COST = 0.600
    read_token_count, write_token_count, estimated_cost = calc_cost(read_cost=READ_COST, write_cost=WRITE_COST)
    print(f"Model: {MODEL_NAME}")
    print(f"Read: {read_token_count} tokens / Write: {write_token_count} tokens")
    print(f"Cost: ${estimated_cost:.2f} USD")
