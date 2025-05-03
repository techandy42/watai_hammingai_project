import math
import concurrent.futures
import pandas as pd
from datasets import load_dataset
from litellm import completion
import subprocess
import tempfile
import sys
import os
from typing import List, Tuple
from tqdm import tqdm
import ast
import os
from dotenv import load_dotenv
from demo.prompts import CodegenPrompts, CodevalTemplates

# Load API keys from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

# Ensure litellm can use the key directly
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def run_python_code(code: str) -> Tuple[List[str], int]:
    """Runs the code and returns (output_lines, status)
    status: 1 if successful, 0 if error"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(code)
        temp_file.flush()
    try:
        result = subprocess.run(
            [sys.executable, temp_file_name],
            capture_output=True,
            text=True,
            timeout=30
        )
        stdout_lines = result.stdout.splitlines()
        stderr_lines = result.stderr.splitlines()
        if result.returncode != 0:
            return stderr_lines, 0  # Returns 0 if code fails
        return stdout_lines, 1  # Returns 1 if code runs successfully
    except:
        return ["Error"], 0

def run_codeval(file_path: str) -> Tuple[pd.DataFrame, float]:
    df_test = pd.read_json(file_path, orient='records', lines=True)
    def run_code(row):
        code = row["pred_code"]
        stdout_lines, status_code = run_python_code(code)
        return stdout_lines, status_code

    df_test[['stdout_lines', 'status_code']] = df_test.apply(run_code, axis=1, result_type="expand")
    num_success = df_test[df_test['status_code'] == 1].shape[0]
    total_num = df_test.shape[0]
    accuracy = num_success / total_num
    return df_test, accuracy

def calculate_aggregated_probability(tokens_info: List[dict]) -> float:
    """Calculate aggregated probability from logprobs."""
    if not tokens_info:
        return float('-inf')
    
    # Filter out non-code tokens and whitespace
    code_tokens = [
        token for token in tokens_info 
        if not (token['token'].strip() == '' or  # Skip whitespace
               token['token'] in ['{', '}', '"', ':', '[', ']', ','] or  # Skip JSON/punctuation
               token['token'].startswith('"') or  # Skip JSON strings
               token['token'].startswith('{') or  # Skip JSON structure
               token['token'].startswith('}'))
    ]
    
    # Convert logprobs to probabilities and multiply
    probabilities = [math.exp(token['logprob']) for token in code_tokens]
    aggregated_prob = math.prod(probabilities)
    
    # Print debug info
    print("\nToken probabilities:")
    for token, prob in zip(code_tokens, probabilities):
        print(f"{token['token']}: {prob:.4f}")
    print(f"Aggregated probability: {aggregated_prob}")
    
    return aggregated_prob

def generate_code_with_probabilities(row, model: str = 'gpt-4o-mini-2024-07-18', num_samples: int = 5) -> Tuple[str, float]:
    """Generate multiple code samples and return the one with highest probability."""
    prompt = row['codegen_prompt']
    messages = [{"role": "user", "content": prompt}]
    
    best_code = None
    highest_probability = float('-inf')
    
    for i in range(num_samples):
        try:
            print(f"\n\nGenerating sample {i+1} with model {model}")
            
            response = completion(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=512,
                logprobs=True,
                top_logprobs=1
            )
            
            # Extract the generated code
            if not response.choices[0].message.content:
                print("Warning: Empty response received")
                continue
                
            generated_code = response.choices[0].message.content
            print(f"\nGenerated Code:\n{generated_code}")
            
            # Calculate probability
            if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                tokens_info = response.choices[0].logprobs.get('content', [])
                aggregated_prob = calculate_aggregated_probability(tokens_info)
                
                if aggregated_prob > highest_probability:
                    highest_probability = aggregated_prob
                    best_code = generated_code
                    print(f"\nNew best sample found! Probability: {aggregated_prob}")
            
        except Exception as e:
            print(f"\nError during generation: {str(e)}")
            continue
    
    if best_code is None:
        return "", float('-inf')
    
    return best_code, highest_probability

def find_best_prompt_with_probabilities(row):
    """Use gpt-4o to find the most probable solution."""
    best_code, highest_prob = generate_code_with_probabilities(row)
    return best_code, highest_prob

def run_entropix_evaluation(file_path: str, num_samples: int = 10):
    # Load and prepare dataset
    dataset = load_dataset('google-research-datasets/mbpp')
    test_dataset = dataset['test']
    df_test = test_dataset.to_pandas()
    
    # Select only 50 problems
    df_test = df_test.head(50)
    
    # Convert numpy arrays to lists
    def convert_test_list(test_array):
        if isinstance(test_array, (list, str)):
            return test_array if isinstance(test_array, list) else [test_array]
        return test_array.tolist()
    
    # Format test lists properly
    df_test['test_list'] = df_test['test_list'].apply(convert_test_list)
    df_test['challenge_test_list'] = df_test['challenge_test_list'].apply(convert_test_list)
    
    # Generate proper prompts
    df_test['codegen_prompt'] = df_test.apply(
        lambda row: CodegenPrompts.get_codegen_prompt(
            text=row['text'],
            io_struct={},
            test_list=row['test_list']
        ), 
        axis=1
    )

    print("Generating solutions with gpt-4o...")
    all_results = []
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        problem_results = []
        for _ in range(num_samples):
            code, prob = generate_code_with_probabilities(row, model='gpt-4o', num_samples=1)
            problem_results.append((code, prob))
        all_results.append(problem_results)
    
    # Store all samples and their probabilities
    df_test['all_samples'] = all_results
    
    # Evaluate each sample
    def evaluate_samples(row):
        samples = row['all_samples']
        results = []
        for code, prob in samples:
            template = CodevalTemplates.get_codeval_template(
                code,
                row["test_list"],
                row["challenge_test_list"]
            )
            _, status = run_python_code(template)
            results.append((prob, status == 1))
        return results
    
    print("\nEvaluating all samples...")
    df_test['evaluation_results'] = df_test.apply(evaluate_samples, axis=1)
    
    # Analyze results
    print("\nAnalyzing results per problem:")
    all_correct_probs = []
    all_wrong_probs = []
    
    print("\n=== Results Summary ===")
    print("Format: [Problem ID]: Success Rate | Avg Probability (Correct) vs Avg Probability (Wrong)")
    print("-" * 80)
    
    for idx, row in df_test.iterrows():
        results = row['evaluation_results']
        correct_results = [(prob, is_correct) for prob, is_correct in results if is_correct]
        wrong_results = [(prob, is_correct) for prob, is_correct in results if not is_correct]
        
        num_correct = len(correct_results)
        num_wrong = len(wrong_results)
        success_rate = (num_correct / num_samples) * 100
        
        correct_probs = [prob for prob, _ in correct_results]
        wrong_probs = [prob for prob, _ in wrong_results]
        
        correct_avg = sum(correct_probs) / len(correct_probs) if correct_probs else 0
        wrong_avg = sum(wrong_probs) / len(wrong_probs) if wrong_probs else 0
        
        all_correct_probs.extend(correct_probs)
        all_wrong_probs.extend(wrong_probs)
        
        print(f"Problem {row['task_id']}: {success_rate:>6.1f}% success | "
              f"Correct: {correct_avg:.4f} vs Wrong: {wrong_avg:.4f} "
              f"({num_correct} correct, {num_wrong} wrong)")
    
    # Overall statistics
    total_correct = len(all_correct_probs)
    total_wrong = len(all_wrong_probs)
    total_samples = total_correct + total_wrong
    overall_success_rate = (total_correct / total_samples) * 100
    
    overall_correct_avg = sum(all_correct_probs) / len(all_correct_probs) if all_correct_probs else 0
    overall_wrong_avg = sum(all_wrong_probs) / len(all_wrong_probs) if all_wrong_probs else 0
    
    print("\n" + "=" * 80)
    print(f"OVERALL RESULTS:")
    print(f"Total Success Rate: {overall_success_rate:.1f}% ({total_correct} correct, {total_wrong} wrong)")
    print(f"Average Probability:")
    print(f"  - Correct solutions: {overall_correct_avg:.4f}")
    print(f"  - Wrong solutions:   {overall_wrong_avg:.4f}")
    print("=" * 80)
    
    # Save results
    df_test.to_json(file_path, orient='records', lines=True)
    return df_test

if __name__ == "__main__":
    print("\nRunning evaluation...")
    df_test = run_entropix_evaluation(
        file_path='entropix_results_test.jsonl',
        num_samples=10  # Generate 10 samples for each problem
    )
