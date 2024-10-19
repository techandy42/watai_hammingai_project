import logging
import json
import concurrent.futures
from typing import Optional
from tqdm import tqdm
from datasets import load_dataset
from o1_research.model import O1BaselineModel
from o1_research.helpers import MBPPRequestId
from demo.codegen import get_io_struct, get_io_struct_prompt, get_codegen_prompt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_mbpp(file_path: str, num_threads: Optional[int] = None):
    dataset = load_dataset('google-research-datasets/mbpp')
    test_dataset = dataset["test"]
    df_test = test_dataset.to_pandas()
    # For now, just running the first two questions
    df_test = df_test[:2]

    df_test["io_struct_prompt"] = df_test.apply(get_io_struct_prompt, axis=1)

    print("Running IO Struct extraction...")
    rows = [row for _, row in df_test.iterrows()]
    if not num_threads:
        num_threads = 32
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        io_structs = list(tqdm(executor.map(get_io_struct, rows), total=len(rows)))
    df_test["io_struct"] = io_structs
    
    df_test["codegen_prompt"] = df_test.apply(get_codegen_prompt, axis=1)

    system_message = "Only include Python code in your output, do not include any comments or tags."
    base_model = "gpt-4o-mini-2024-07-18"  
    context_limit = 8192
    token_limit = 8192
    interactive = False 
    
    results = []
    
    for idx, row in tqdm(df_test.iterrows()): 
        request_id = MBPPRequestId.create_request_id(row['task_id'])
        initial_question = row['codegen_prompt']
        model = O1BaselineModel(
            request_id=request_id,
            base_model=base_model,
            context_limit=context_limit,
            token_limit=token_limit,
            initial_question=initial_question,
            system_message=system_message,
            interactive=interactive
        )
        try:
            _ = model.think_v1()
        except Exception as e:
            logging.error(f"Exception for index {idx}: {str(e)}")
        result = model.save_result()
        results.append(result)
    
    with open(file_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    run_mbpp("mbpp_results.jsonl")
