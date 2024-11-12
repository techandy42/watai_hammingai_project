import argparse
import ast 
import concurrent.futures
from tqdm import tqdm
from typing import Optional, List, Tuple
from pydantic import BaseModel
from datasets import load_dataset
from demo.helpers import remove_python_code_tags, split_assert_statements
from demo.request import make_request, make_request_structured_output
from demo.prompts import CodegenPrompts

def get_codegen_prompt(row):
    text = row["text"]
    test_list = row["test_list"]
    io_struct = row["io_struct"]
    codegen_prompt = CodegenPrompts.get_codegen_prompt(text, io_struct)
    return codegen_prompt

def get_pred_code(row, model):
    codegen_prompt = row["codegen_prompt"]
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
            code = make_request(
                model=model,
                messages=messages,
            )
            break
        except Exception as e:
            print(f"Error occurred: {str(e)}")
    
    if code is None:
        code = ""

    cleaned_code = remove_python_code_tags(code)

    return cleaned_code

def get_io_struct(row, model):
    io_struct_prompt = row["io_struct_prompt"]
    messages = [
        {
            "content": io_struct_prompt,
            "role": "user",
        }
    ]
    class IOStruct(BaseModel):
        function_name: str
        input: List[str]
        output: str
        specific_output: bool
        specific_output_values: List[str]

    num_retry = 3

    io_struct = None

    for i in range(num_retry):
        try:
            io_struct = make_request_structured_output(
                model=model,
                messages=messages,
                response_format=IOStruct,
            )
            break
        except Exception as e:
            print(f"Error occurred: {str(e)}")
    
    if io_struct is None:
        io_struct = {}

    return io_struct

def add_comma_to_newline(row, column_name):
    return str(row[column_name]).replace("\n", ",\n")

def get_split_assert_statements(row, column_name):
    test_list = row[column_name]
    test_list = split_assert_statements(test_list)
    return test_list

def get_io_struct_prompt(row):
    test_list = row["test_list"]
    challenge_test_list = row["challenge_test_list"]
    combined_test_list = [*test_list, *challenge_test_list]
    io_struct_prompt = CodegenPrompts.get_io_struct_extraction_prompt_markdown(combined_test_list)
    return io_struct_prompt

def run_codegen(file_path: str, section: str, num_threads: Optional[int] = None, range: Optional[Tuple[int, int]] = None):
    extraction_model = "gpt-4o-mini-2024-07-18"
    codegen_model = "gpt-4o-mini-2024-07-18"
    dataset = load_dataset('google-research-datasets/mbpp')
    test_dataset = dataset[section]
    df_test = test_dataset.to_pandas()
    if range is None:
        range = (0, len(df_test)+1)
    df_test = df_test.iloc[range[0]:range[1]]

    df_test["test_list"] = df_test.apply(add_comma_to_newline, axis=1, column_name="test_list")
    df_test["challenge_test_list"] = df_test.apply(add_comma_to_newline, axis=1, column_name="challenge_test_list")
    df_test["test_list"] = df_test['test_list'].apply(ast.literal_eval)
    df_test["challenge_test_list"] = df_test['challenge_test_list'].apply(ast.literal_eval)
    df_test["test_list"] = df_test.apply(get_split_assert_statements, axis=1, column_name="test_list")
    df_test["challenge_test_list"] = df_test.apply(get_split_assert_statements, axis=1, column_name="challenge_test_list")

    df_test["io_struct_prompt"] = df_test.apply(get_io_struct_prompt, axis=1)

    print("Running IO Struct extraction...")
    rows = [row for _, row in df_test.iterrows()]
    if not num_threads:
        num_threads = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        io_structs = list(tqdm(executor.map(lambda row: get_io_struct(row, extraction_model), rows), total=len(rows)))
    df_test["io_struct"] = io_structs

    df_test["codegen_prompt"] = df_test.apply(get_codegen_prompt, axis=1)

    print("Running code generation...")
    rows = [row for _, row in df_test.iterrows()]
    if not num_threads:
        num_threads = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        pred_codes = list(tqdm(executor.map(lambda row: get_pred_code(row, codegen_model), rows), total=len(rows)))
    df_test["pred_code"] = pred_codes

    df_test.to_json(file_path, orient='records', lines=True)

def main():
    parser = argparse.ArgumentParser(description='Run MBPP evaluation')
    parser.add_argument('--start', type=int, required=True, help='Start index (inclusive)')
    parser.add_argument('--end', type=int, required=True, help='End index (exclusive)')
    parser.add_argument('--section', type=str, default="test", help='Optional section of the MBPP dataset (test or train, defaults to test)')
    parser.add_argument('--version', type=str, help='Optional version prefix for the output file')

    args = parser.parse_args()

    range = None if args.start is None else (args.start, args.end)
    prefix = f"{args.version}_" if args.version else ""
    file_path = f"eval_results/{prefix}mbpp_results_{range[0]}_to_{range[1]-1}.jsonl"
    run_codegen(file_path=file_path, section=args.section, range=range)

if __name__ == "__main__":
    main()
