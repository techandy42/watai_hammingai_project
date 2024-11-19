import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load
reward_model_name = './reward_model'
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
reward_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reward_model.to(device)

# sample
context = "Write a function to calculate the factorial of a number."
candidate_a = "What is the base case for the factorial function?"
candidate_b = "Provide the Python code for the factorial function."

def score_candidate(context, candidate):
    input_text = context + " " + candidate
    inputs = reward_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = reward_model(**inputs)
        score = outputs.logits.item()
    return score

# score the candidates
score_a = score_candidate(context, candidate_a)
score_b = score_candidate(context, candidate_b)

print(f"Score for Candidate A: {score_a}")
print(f"Score for Candidate B: {score_b}")

# determine which candidate is better
if score_a > score_b:
    print("Candidate A is preferred.")
else:
    print("Candidate B is preferred.")

test_cases = [
    {
        'context': "Explain the concept of inheritance in object-oriented programming.",
        'candidate_a': "What is inheritance in OOP?",
        'candidate_b': "Provide an example of polymorphism.",
        'expected_preference': 'a'
    },
    {
        'context': "Describe how to implement a linked list in Python.",
        'candidate_a': "What are the basic operations of a linked list?",
        'candidate_b': "What is a linked list?",
        'expected_preference': 'a'
    },
]

for idx, case in enumerate(test_cases):
    score_a = score_candidate(case['context'], case['candidate_a'])
    score_b = score_candidate(case['context'], case['candidate_b'])
    predicted_preference = 'a' if score_a > score_b else 'b'
    print(f"Test Case {idx+1}:")
    print(f"Score A: {score_a}, Score B: {score_b}")
    print(f"Predicted Preference: Candidate {predicted_preference}")
    print(f"Expected Preference: Candidate {case['expected_preference']}")
    print("Result:", "Pass" if predicted_preference == case['expected_preference'] else "Fail")
    print("-" * 50)
