import argparse
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from demo.prompts import CodevalTemplates
from demo.process import run_python_code

tqdm.pandas()

def get_code_template(row):
    code_template = CodevalTemplates.get_codeval_template(row["pred_code"], row["test_list"], row["challenge_test_list"])
    return code_template

def run_code(row):
    code = row["code_template"]
    stdout_lines, status_code = run_python_code(code)
    return stdout_lines, status_code

def run_codeval(file_path: str) -> Tuple[pd.DataFrame, float]:
    df_test = pd.read_json(file_path, orient='records', lines=True)
    df_test["code_template"] = df_test.apply(get_code_template, axis=1)
    print("Running unit tests...")
    df_test[['stdout_lines', 'status_code']] = df_test.progress_apply(run_code, axis=1, result_type="expand")
    num_success = df_test[df_test['status_code'] == 1].shape[0]
    total_num = df_test.shape[0]
    accuracy = num_success / total_num
    return df_test, accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate MBPP code generation results')
    parser.add_argument('--src_file', type=str, required=True, help='Source JSONL file containing models to evaluate')
    
    args = parser.parse_args()

    tgt_file = args.src_file.replace(".jsonl", "_validated.jsonl")
    df_test, accuracy = run_codeval(f"./eval_results/{args.src_file}")
    df_test.to_json(f"./eval_results/{tgt_file}", orient='records', lines=True)
    print(f"Accuracy: {accuracy*100:.2f}")

if __name__ == "__main__":
    main()
