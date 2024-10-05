import pandas as pd
import ast
from tqdm import tqdm
from prompts import CodevalTemplates
from process import run_python_code

tqdm.pandas()

def get_code_template(row):
    code_template = CodevalTemplates.get_codeval_template(row["pred_code"], row["test_list"], row["challenge_test_list"])
    return code_template

def run_code(row):
    code = row["code_template"]
    stdout_lines, status_code = run_python_code(code)
    return stdout_lines, status_code

def run_codeval():
    df_test = pd.read_csv("mbpp_hammingai.csv")
    df_test["test_list"] = df_test['test_list'].apply(ast.literal_eval)
    df_test["challenge_test_list"] = df_test['challenge_test_list'].apply(ast.literal_eval)
    df_test["code_template"] = df_test.apply(get_code_template, axis=1)
    print("Running unit tests...")
    df_test[['stdout_lines', 'status_code']] = df_test.progress_apply(run_code, axis=1, result_type="expand")
    return df_test

def main():
    df_test = run_codeval()
    df_test.to_csv("mbpp_hammingai_validated.csv", index=False)

if __name__ == "__main__":
    main()
