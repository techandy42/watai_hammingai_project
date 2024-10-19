import pandas as pd
import ast
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
    df_test = pd.read_csv(file_path)
    df_test["test_list"] = df_test['test_list'].apply(ast.literal_eval)
    df_test["challenge_test_list"] = df_test['challenge_test_list'].apply(ast.literal_eval)
    df_test["code_template"] = df_test.apply(get_code_template, axis=1)
    print("Running unit tests...")
    df_test[['stdout_lines', 'status_code']] = df_test.progress_apply(run_code, axis=1, result_type="expand")
    num_success = df_test[df_test['status_code'] == 1].shape[0]
    total_num = df_test.shape[0]
    accuracy = num_success / total_num
    return df_test, accuracy

def main():
    input_file_path = "mbpp_hammingai.csv"
    output_file_path = "mbpp_hammingai_validated.csv"
    df_test, accuracy = run_codeval(input_file_path)
    df_test.to_csv(output_file_path, index=False)
    print(f"Accuracy: {accuracy*100:.2f}")

if __name__ == "__main__":
    main()
