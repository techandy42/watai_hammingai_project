import argparse
import time
import logging
import json
import concurrent.futures
from typing import Optional, Tuple
from tqdm import tqdm
from datasets import load_dataset
from o1_research.model import O1BaselineModel
from o1_research.helpers import MBPPRequestId, calc_cost
from demo.codegen import get_io_struct, get_io_struct_prompt, get_codegen_prompt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_mbpp(file_path: str, num_threads: Optional[int] = None, range: Optional[Tuple[int, int]] = None):
    start_time = time.time()

    dataset = load_dataset('google-research-datasets/mbpp')
    test_dataset = dataset["test"]
    df_test = test_dataset.to_pandas()
    if range is None:
        range = (0, len(df_test)+1)
    df_test = df_test.iloc[range[0]:range[1]]

    df_test["io_struct_prompt"] = df_test.apply(get_io_struct_prompt, axis=1)

    if not num_threads:
        num_threads = 8

    print("Running IO Struct extraction...")
    extraction_model = "gpt-4o-mini-2024-07-18"
    rows = [row for _, row in df_test.iterrows()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        io_structs = list(tqdm(executor.map(lambda row: get_io_struct(row, extraction_model), rows), total=len(rows)))
    
    df_test["io_struct"] = io_structs
    df_test["codegen_prompt"] = df_test.apply(get_codegen_prompt, axis=1)

    system_message = "Only include Python code in your output, do not include any comments or tags."
    models = [
        "gpt-4o-2024-08-06", # $2.5/1M input, $10/1M output
        "claude-3-5-sonnet-20240620", # $3/1M input, $15/1M output
        "gemini/gemini-1.5-pro", # $1.25/1M input, $5/1M output
        "command-r-plus-08-2024" # $2.5/1M input, $10/1M output
    ]
    ranking_model = "gpt-4o-2024-08-06"
    context_limit = 4096
    token_limit = 4096
    interactive = False
    total_input_token_count = 0
    total_output_token_count = 0
    price_per_mill_input = (2.5 + 3 + 1.25 + 2.5) / 4
    price_per_mill_output = (10 + 15 + 5 + 10) / 4

    results = []

    def process_row(row):
        nonlocal total_input_token_count, total_output_token_count
        request_id = MBPPRequestId.create_request_id(row['task_id'])
        initial_question = row['codegen_prompt']
        model = O1BaselineModel(
            request_id=request_id,
            models=models,
            ranking_model=ranking_model,
            context_limit=context_limit,
            token_limit=token_limit,
            initial_question=initial_question,
            system_message=system_message,
            interactive=interactive
        )
        try:
            _ = model.think_v1()
        except Exception as e:
            logging.error(f"Exception for task_id {row['task_id']}: {str(e)}")
        result = model.save_result()
        results.append(result)
        total_input_token_count += model.input_token_count
        total_output_token_count += model.output_token_count

    print("Running codegen...")
    rows = [row for _, row in df_test.iterrows()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(process_row, rows), total=len(rows)))

    print("Saving results...")
    with open(file_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    execution_time = time.time() - start_time
    total_cost = calc_cost(total_input_token_count, total_output_token_count, price_per_mill_input, price_per_mill_output)

    print("=" * 100)
    print(f"**Execution Time: {execution_time:.2f} seconds")
    print(f"**Total Cost: ${total_cost:.4f}")
    print(f"Input Token Count: {total_input_token_count}")
    print(f"Output Token Count: {total_output_token_count}")
    print("=" * 100)

def merge_and_sort_jsonl(file_paths, output_file):
    merged_data = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                merged_data.append(data)

    sorted_data = sorted(merged_data, key=lambda x: MBPPRequestId.extract_request_id(x['id']))

    with open(output_file, 'w') as out_f:
        for item in sorted_data:
            out_f.write(json.dumps(item) + '\n')

# Example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MBPP evaluation')
    parser.add_argument('--start', type=int, help='Start index (required if --end is provided)')
    parser.add_argument('--end', type=int, help='End index (required if --start is provided)')
    parser.add_argument('--version', type=str, help='Optional version prefix for the output file')
    
    args = parser.parse_args()
    
    if (args.start is None) != (args.end is None):
        parser.error("Both --start and --end must be provided together")
    
    range = None if args.start is None else (args.start, args.end)
    prefix = f"{args.version}_" if args.version else ""
    file_path = f"eval_results/{prefix}mbpp_results_{range[0]}_to_{range[1]-1}.jsonl"
    run_mbpp(file_path=file_path, range=range)
