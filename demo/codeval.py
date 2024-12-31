import argparse
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from prompts import CodevalTemplates
from process import run_python_code
import re
import ast
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
    # df_test = pd.read_json(file_path, orient='records', lines=True)
    df_test = pd.read_csv(file_path)
    df_test["test_list"] = df_test['test_list'].apply(ast.literal_eval)
    df_test["challenge_test_list"] = df_test['challenge_test_list'].apply(ast.literal_eval)
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

def run_tests(all_error_funcs, context_sizes, depth_sizes, results_file):
    # Load the MBPP dataset
    dataset = load_dataset('google-research-datasets/mbpp')

    # Extract all functions from the dataset
    dataset_functions = [example['code'] for example in dataset['train']]
    dataset_size = len(dataset_functions)

    # Open the results file in append mode
    with open(results_file, 'a') as f:
        for buggy_function in tqdm(all_error_funcs, desc="Processing buggy functions"):
            for context_length in tqdm(context_sizes, desc="Processing context sizes", leave=False):
                # Ensure context_length doesn't exceed dataset size
                adjusted_context_length = min(context_length, dataset_size + 1)

                # Add a progress bar for depth sizes
                depth_bar = tqdm(depth_sizes, leave=False)
                for depth_percentage in depth_bar:
                    # Update the description dynamically
                    depth_bar.set_description(
                        f"Buggy: {get_function_name(buggy_function)}, Context: {context_length}, Depth: {depth_percentage}%"
                    )
                    # Calculate target depth
                    target_depth = max(1, min(adjusted_context_length, int(depth_percentage / 100 * adjusted_context_length)))

                    # Sample random functions
                    sample_size = adjusted_context_length - 1
                    selected_functions = random.sample(dataset_functions, sample_size)

                    # Insert the buggy function
                    selected_functions.insert(target_depth - 1, buggy_function)

                    # Create the code stack
                    code_stack = "\n".join(selected_functions)

                    # Prepare the entry
                    entry = {
                        "code": code_stack,
                        "func_error": get_function_name(buggy_function),
                        "context_length": adjusted_context_length,
                        "depth_percentage": depth_percentage
                    }

                    # Write the entry to the results file
                    f.write(json.dumps(entry) + '\n')    

def main():
    input_file_path = "mbpp_hammingai.csv"
    output_file_path = "mbpp_hammingai_validated.csv"
    df_test, accuracy, all_error_funcs = run_codeval(input_file_path)
    df_test.to_csv(output_file_path, index=False)
    print(f"Accuracy: {accuracy*100:.2f}")

    all_error_funcs = [item for item in all_error_funcs if item is not None]
    # save all error funcs
    with open('all_error_funcs.json', 'w') as f:
        json.dump(all_error_funcs, f)

    # func_names = []
    # for func in all_error_funcs:
    #     func_names.append(get_function_name(func))
    # print(func_names)

    # context_sizes = [500, 1000, 2000, 4000, 8000, 16000]
    # depth_sizes = [0, 25, 50, 75, 100]  # Percentages as integers
    # results_file = "bug_in_codestack_dataset.jsonl"

    # run_tests(all_error_funcs, context_sizes, depth_sizes, results_file)

if __name__ == "__main__":
    main()