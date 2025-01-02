import json
from openai import OpenAI
import os
from tqdm import tqdm

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def construct_prompt(code):
    return f"""
<instruction_header>
The following is a code sample (e.g., source_code).
One of the functions in this code contains a bug.
Identify the function with the bug.
<instruction_header>

<source_code>
{code}
<source_code>

<output_format>
Please return the name of the function containing the bug, nothing else. Do not alter the name of the function in any way. If there is no bug, return 'none'.
<output_format>
"""

# Test LLM on the jsonl data
def test_llm_on_jsonl(bics_results_file, jsonl_file, model="gpt-4o", temperature=0.0):
    data = load_jsonl(jsonl_file)
    correct_predictions = 0
    total_predictions = 0

    with open(bics_results_file, 'a') as f:
        # Wrap the iteration with tqdm for a progress bar
        for item in tqdm(data, desc="Testing LLM", unit="sample"):
            code = item['code']
            correct_func_name = item['func_error']
            
            # Construct the prompt
            prompt = construct_prompt(code)
            
            # Call the LLM API using the new API method (ChatCompletion)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=16000,
            )

            # Get the model's response
            predicted_func_name = response.choices[0].message.content.strip()
            item['guess'] = predicted_func_name

            # Check if the predicted function name matches the correct one
            if correct_func_name and correct_func_name in predicted_func_name:
                correct_predictions += 1
                item['is_correct'] = 1
            else:
                item['is_correct'] = 0

            total_predictions += 1
            accuracy = round(correct_predictions / total_predictions * 100, 2)
            item['accuracy'] = accuracy

            f.write(json.dumps(item) + '\n')    


def main():
    bics_results_file = 'bics_results_file.jsonl'
    jsonl_file = 'bug_in_codestack_dataset.jsonl'
    test_llm_on_jsonl(bics_results_file, jsonl_file)

if __name__ == "__main__":
    main()