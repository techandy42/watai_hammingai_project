import argparse
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from prompts import CodevalTemplates
from process import run_python_code
import re
from datasets import load_dataset
import random
import json


tqdm.pandas()

def get_code_template(row):
    code_template = CodevalTemplates.get_codeval_template(row["pred_code"], row["test_list"], row["challenge_test_list"])
    return code_template

def get_function_name(function_str):
    match = re.match(r"def (\w+)\(", function_str)
    if match:
        return match.group(1)
    return None

def run_code(row):
    code = row["code_template"]
    stdout_lines, status_code = run_python_code(code)
    error_funcs = []
    if status_code == 0: 
        error_funcs.append(code)
    return stdout_lines, status_code, error_funcs

def run_codeval(file_path: str) -> Tuple[pd.DataFrame, float]:
    df_test = pd.read_json(file_path, orient='records', lines=True)
    df_test["code_template"] = df_test.apply(get_code_template, axis=1)
    print("Running unit tests...")

    # Initialize an empty list to store all error functions
    all_error_funcs = []

    def collect_error_funcs(row):
        stdout_lines, status_code, error_funcs = run_code(row)
        # Extend the large list with the error functions from the current row
        all_error_funcs.extend(error_funcs)
        return stdout_lines, status_code, error_funcs

    # Apply the updated function that collects error functions
    df_test[['stdout_lines', 'status_code', 'error_funcs']] = df_test.progress_apply(collect_error_funcs, axis=1, result_type="expand")

    num_success = df_test[df_test['status_code'] == 1].shape[0]
    total_num = df_test.shape[0]
    accuracy = num_success / total_num

    # Return the dataframe and the accuracy, along with the collected error functions
    return df_test, accuracy, all_error_funcs

def generate_dataset(all_error_funcs, context_length, target_depth, results_file):
    # Load the MBPP dataset
    dataset = load_dataset('google-research-datasets/mbpp')
    
    # Open the results file in append mode
    with open(results_file, 'a') as f:
        # Iterate through the buggy functions in all_error_funcs
        for buggy_function in all_error_funcs:
            # Select a random number of functions from the MBPP dataset based on target_depth
            selected_functions = []
            for _ in range(target_depth):
                # Randomly select an example from the MBPP dataset
                random_example = random.choice(dataset['train'])['code']
                selected_functions.append(random_example)
            
            # Combine the buggy function with the selected functions from MBPP
            code_stack = buggy_function + "\n" + "\n".join(selected_functions)
            
            # Trim the code stack to fit within the context length (if necessary)
            tokens = code_stack.split()  # Assuming words are the tokens
            if len(tokens) > context_length:
                code_stack = " ".join(tokens[:context_length])

            # Prepare the entry to write to the JSONL file
            entry = {
                "code": code_stack,
                "func_error": get_function_name(buggy_function)
            }

            # Write the entry as a JSON object in the results file
            f.write(json.dumps(entry) + '\n')

def main():
    input_file_path = "mbpp_hammingai.csv"
    output_file_path = "mbpp_hammingai_validated.csv"
    df_test, accuracy, all_error_funcs = run_codeval(input_file_path)
    df_test.to_csv(output_file_path, index=False)
    print(f"Accuracy: {accuracy*100:.2f}")
    print(all_error_funcs)

    context_length = 1024  # Example context length in tokens
    target_depth = 3  # Example target depth (number of functions from MBPP)
    results_file = "bug_in_codestack_dataset.jsonl"

    generate_dataset(all_error_funcs, context_length, target_depth, results_file)

if __name__ == "__main__":
    main()
