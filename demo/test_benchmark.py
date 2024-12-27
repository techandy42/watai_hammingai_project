import json
import openai  # Replace with the LLM provider you're using, e.g., OpenAI, Hugging Face, etc.

# Load the jsonl file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Prompt construction for testing LLMs
def construct_prompt(code):
    return f"""
The following is a code sample. One of the functions in this code contains a bug. Identify the function with the bug:

{code}

Please return the name of the function containing the bug.
"""

# Test LLM on the jsonl data
def test_llm_on_jsonl(jsonl_file, model="gpt-4", temperature=0.0):
    data = load_jsonl(jsonl_file)
    correct_predictions = 0
    total_predictions = len(data)

    for item in data:
        code = item['code']
        correct_func_name = item['func_error']
        
        # Construct the prompt
        prompt = construct_prompt(code)
        
        # Call the LLM API (this example uses OpenAI's GPT-4 API)
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=100,
            temperature=temperature,
            stop=["\n"]
        )

        # Get the model's response
        predicted_func_name = response.choices[0].text.strip()

        # Check if the predicted function name matches the correct one
        if predicted_func_name == correct_func_name:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# Example usage:
# Test the model with the generated results from `results.jsonl`
jsonl_file = 'results.jsonl'  # Make sure to specify your jsonl file here
openai.api_key = 'your-api-key-here'  # Replace with your OpenAI API key

accuracy = test_llm_on_jsonl(jsonl_file)
