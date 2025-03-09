import math
from datasets import load_dataset
import pandas as pd
from litellm import completion
import os
from dotenv import load_dotenv
from demo.prompts import CodegenPrompts, CodevalTemplates
from test import run_python_code
from typing import List, Dict, Tuple
from tqdm import tqdm

# Load API keys from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def get_code_with_probabilities(prompt: str) -> Tuple[str, List[Dict]]:
    """Generate code and get token probabilities."""
    messages = [{"role": "user", "content": prompt}]
    
    response = completion(
        model='gpt-4o',
        messages=messages,
        temperature=0.2,
        max_tokens=512,
        logprobs=True,
        top_logprobs=1
    )
    
    generated_code = response.choices[0].message.content
    tokens_info = response.choices[0].logprobs.get('content', [])
    
    return generated_code, tokens_info

def format_token_probabilities(tokens_info: List[Dict]) -> str:
    """Format token probabilities like the example."""
    result = ""
    current_line = ""
    
    for token in tokens_info:
        prob = math.exp(token['logprob'])
        token_text = token['token']
        
        if token_text.strip() == '':
            if '\n' in token_text:
                result += current_line + "\n"
                current_line = ""
            continue
            
        if token_text.strip():
            current_line += f"{token_text} ({prob:.2f}) "
    
    if current_line:
        result += current_line
    
    return result

def identify_uncertain_parts(code: str, tokens_info: List[Dict], threshold: float = 0.8) -> List[Tuple[int, str, float]]:
    """Identify tokens with probability below threshold and their line numbers."""
    uncertain_parts = []
    current_line = 1
    
    for token in tokens_info:
        if '\n' in token['token']:
            current_line += token['token'].count('\n')
        
        prob = math.exp(token['logprob'])
        if prob < threshold and token['token'].strip():
            uncertain_parts.append((current_line, token['token'], prob))
    
    return uncertain_parts

def create_debug_prompt(problem_text: str, code: str, uncertain_parts: List[Tuple[int, str, float]]) -> str:
    """Create a prompt for debugging following the example format."""
    # Format code with line numbers
    code_lines = code.split('\n')
    numbered_code = '\n'.join(f"Line {i+1}] {line}" for i, line in enumerate(code_lines, 1))
    
    # Format uncertain parts
    uncertain_lines = []
    for line_num, token, prob in uncertain_parts:
        uncertain_lines.append(f"Line {line_num}] {token} ({prob:.2f})")
    uncertain_parts_str = '\n'.join(uncertain_lines)
    
    prompt = f"""
Initial Problem Statement:
{problem_text}

Incorrect Code:
{numbered_code}

Uncertain Parts:
{uncertain_parts_str}

Instruction:
Given the above problem statement, the incorrect solution to the problem, and parts of the code that the LLM was less certain about, can you please find and correct issues with the code?
Output only the corrected code as your final output.
"""
    return prompt

def run_debugging_evaluation():
    """Run the debugging evaluation on MBPP test questions."""
    # Load dataset
    dataset = load_dataset('google-research-datasets/mbpp')
    test_dataset = dataset['test']
    df_test = test_dataset.to_pandas().head(50)  # Use first 50 problems
    
    results = []
    
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        print(f"\nProcessing problem {row['task_id']}...")
        
        # Generate initial solution
        prompt = CodegenPrompts.get_codegen_prompt(
            text=row['text'],
            io_struct={},
            test_list=row['test_list']
        )
        
        # Get initial solution with probabilities
        initial_code, tokens_info = get_code_with_probabilities(prompt)
        
        # Test initial solution
        test_template = CodevalTemplates.get_codeval_template(
            initial_code,
            row["test_list"],
            row["challenge_test_list"]
        )
        _, initial_status = run_python_code(test_template)
        
        result = {
            'task_id': row['task_id'],
            'problem_text': row['text'],
            'initial_code': initial_code,
            'initial_status': initial_status,
            'token_probabilities': format_token_probabilities(tokens_info)
        }
        
        # If solution is incorrect, try debugging
        if initial_status == 0:
            print("\nSolution incorrect, analyzing uncertain parts...")
            print("\nToken probabilities:")
            print(result['token_probabilities'])
            
            uncertain_parts = identify_uncertain_parts(initial_code, tokens_info)
            debug_prompt = create_debug_prompt(row['text'], initial_code, uncertain_parts)
            
            print("\nAttempting to debug with uncertain parts...")
            debugged_code, _ = get_code_with_probabilities(debug_prompt)
            
            # Test debugged solution
            debug_template = CodevalTemplates.get_codeval_template(
                debugged_code,
                row["test_list"],
                row["challenge_test_list"]
            )
            _, debug_status = run_python_code(debug_template)
            
            result.update({
                'uncertain_parts': [(line, token, prob) for line, token, prob in uncertain_parts],
                'debugged_code': debugged_code,
                'debug_status': debug_status
            })
        
        results.append(result)
        
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_json('debug_results.jsonl', orient='records', lines=True)
    
    # Print summary
    total = len(results)
    initially_correct = sum(1 for r in results if r['initial_status'] == 1)
    needed_debug = total - initially_correct
    debug_successful = sum(
        1 for r in results 
        if r['initial_status'] == 0 and 
        r.get('debug_status', 0) == 1
    )
    
    print("\n=== Debug Evaluation Results ===")
    print(f"Total problems: {total}")
    print(f"Initially correct: {initially_correct} ({initially_correct/total*100:.1f}%)")
    print(f"Needed debugging: {needed_debug}")
    if needed_debug > 0:
        print(f"Successfully debugged: {debug_successful} ({debug_successful/needed_debug*100:.1f}% of incorrect solutions)")
    print(f"Final success rate: {(initially_correct + debug_successful)/total*100:.1f}%")

if __name__ == "__main__":
    print("Running debugging evaluation...")
    run_debugging_evaluation()
