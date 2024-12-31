from datasets import load_dataset
from tqdm import tqdm
import random
import json
import re

tqdm.pandas()

# Load the MBPP dataset
dataset = load_dataset('google-research-datasets/mbpp')
dataset_functions = [example['code'] for example in dataset['train']]


def get_function_name(function_str):
    match = re.match(r"def (\w+)\(", function_str)
    if match:
        return match.group(1)
    return None

def printFuncNames(all_error_funcs):
    func_names = []
    for func in all_error_funcs:
        func_names.append(get_function_name(func))
    print(func_names)

def convert_codestack_to_string(codestack):
    str_codestack = ""
    for function in codestack:
        str_codestack += function + "\n"
    return str_codestack.strip()

def generate_code_stack(context_size, random_error_func):
    random_error_func_tokens = len(random_error_func.split())
    codestack = []
    token_count = 0
    
    for function in dataset_functions:
        function_token_count = len(function.split())
        
        # If adding this function exceeds the context size, stop
        if token_count + function_token_count + random_error_func_tokens > context_size:
            break
        
        # Otherwise, add the function to the codestack
        codestack.append(function)
        token_count += function_token_count
    
    return codestack

def insert_buggy_function(codestack, error_function, depth_size):
    # Calculate the insertion index based on the depth
    num_functions = len(codestack)
    insertion_index = int((depth_size / 100) * num_functions)
    
    # Insert the error function at the calculated index
    codestack.insert(insertion_index, error_function)
    
    return codestack


def run_tests(all_error_funcs, context_sizes, depth_sizes, results_file):
    with open(results_file, 'a') as f:
        for _ in tqdm(range(20), desc="Processing testcases"):
            for context_length in tqdm(context_sizes, desc="Processing context sizes", leave=False):
                depth_bar = tqdm(depth_sizes, leave=False)
                for depth_percentage in depth_bar:
                    random_error_func = random.choice(all_error_funcs)
                    error_func_name = get_function_name(random_error_func)

                    depth_bar.set_description(
                        f"Buggy: {error_func_name}, Context: {context_length}, Depth: {depth_percentage}%"
                    )

                    codestack = generate_code_stack(context_length, random_error_func)
                    codestack = insert_buggy_function(codestack, random_error_func, depth_percentage)
                    str_codestack = convert_codestack_to_string(codestack)

                    entry = {
                        "code": str_codestack,
                        "func_error": error_func_name,
                        "context_length": context_length,
                        "depth_percentage": depth_percentage
                    }

                    # Write the entry to the results file
                    f.write(json.dumps(entry) + '\n')    


def main():
    context_sizes = [500, 1000, 2000, 4000, 8000, 16000]
    depth_sizes = [0, 25, 50, 75, 100]  # Percentages as integers

    with open('all_error_funcs.json', 'r') as f:
        all_error_funcs = json.load(f)  
        # printFuncNames(all_error_funcs)

        results_file = "bug_in_codestack_dataset.jsonl"
        run_tests(all_error_funcs, context_sizes, depth_sizes, results_file)

if __name__ == "__main__":
    main()