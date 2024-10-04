from datasets import load_dataset
from helpers import get_tested_function, remove_python_code_tags
from request import make_request
from prompts import CodegenPrompts
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd
from tqdm import tqdm
from typing import Optional

def get_function_name(row):
    function_name = get_tested_function(row["test_list"])
    return function_name

def get_codegen_prompt(row):
    text = row["text"]
    function_name = row["function_name"]
    codegen_prompt = CodegenPrompts.get_codegen_prompt(text, function_name)
    return codegen_prompt

def get_pred_code(row):
    task_id = row["task_id"]
    codegen_prompt = row["codegen_prompt"]
    model = "groq/llama-3.1-70b-versatile"
    messages = [
        {
            "content": codegen_prompt,
            "role": "user",
        }
    ]

    num_retry = 3

    code = None

    for i in range(num_retry):
        try:
            # print(f"Running sample no.{task_id} for the {i+1}th time...")
            code = make_request(
                model=model,
                messages=messages,
            )
            break
        except Exception as e:
            print(f"Error occurred: {str(e)}")
    
    if code is None:
        code = ""  # or handle it as needed

    cleaned_code = remove_python_code_tags(code)

    return cleaned_code

def run_codegen(num_processes: Optional[int] = None):
    dataset = load_dataset('google-research-datasets/mbpp')
    test_dataset = dataset["test"]
    df_test = test_dataset.to_pandas()

    # Apply functions sequentially for preparation
    df_test["function_name"] = df_test.apply(get_function_name, axis=1)
    df_test["codegen_prompt"] = df_test.apply(get_codegen_prompt, axis=1)

    # Prepare rows for multiprocessing
    rows = [row for _, row in df_test.iterrows()]

    # Define the number of processes
    if not num_processes:
        num_processes = cpu_count()

    # Initialize the multiprocessing pool
    with Pool(processes=num_processes) as pool:
        # Use tqdm for progress bar (optional)
        pred_codes = list(tqdm(pool.imap(get_pred_code, rows), total=len(rows)))

    # Assign the results back to the DataFrame
    df_test["pred_code"] = pred_codes

    return df_test

def main():
    df_test = run_codegen(num_processes=1)
    df_test.to_csv("mbpp_hammingai.csv", index=False)

if __name__ == "__main__":
    main()
