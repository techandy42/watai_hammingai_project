import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import numpy as np
import os

# dataset
dataset_path = 'hybrid_test_mbpp_results_0_to_49_successful.jsonl'

# load
reward_model_name = './reward_model'  
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
reward_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reward_model.to(device)

# using the reward model
def score_candidate(context, candidate):
    input_text = context + " " + candidate
    inputs = reward_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = reward_model(**inputs)
        score = outputs.logits.item()
    return score

# count
total_thoughts = 0
correct_chosen_answers = 0

# dataset file
with open(dataset_path, 'r') as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line)
        initial_question = data['initial_question']
        thoughts = data['thoughts']
        entry_id = data.get('id', f'Entry {line_num}')

        for thought_idx, thought in enumerate(thoughts):
            total_thoughts += 1
            chosen_answer_idx = thought['chosen_answer_idx']
            chosen_answer = thought['chosen_answer']
            candidate_answers = thought['answers']
            context = initial_question + " " + thought['chosen_question']

            scores = []
            for idx, answer in enumerate(candidate_answers):
                # assign low score to parsing error
                if answer == "Parsing Error after retries":
                    score = float('-inf')  # if invalid
                else:
                    score = score_candidate(context, answer)
                scores.append(score)
                # uncomment the following line to print each score
                print(f"Entry {entry_id}, Thought {thought_idx+1}, Answer {idx}: Score = {score}")

            # find the index of the answer with the highest score
            best_answer_idx = int(np.argmax(scores))

            # check if the chosen answer is the one with the highest reward
            if chosen_answer_idx == best_answer_idx:
                correct_chosen_answers += 1
                # uncomment to print when the chosen answer matches
                print(f"Entry {entry_id}, Thought {thought_idx+1}: Chosen answer is the one with the highest reward.")
            else:
                # uncomment to print when the chosen answer does not match
                print(f"Entry {entry_id}, Thought {thought_idx+1}: Chosen answer is NOT the one with the highest reward.")
                pass

accuracy = (correct_chosen_answers / total_thoughts) * 100 if total_thoughts > 0 else 0
print(f"Total thoughts processed: {total_thoughts}")
print(f"Number of times the chosen answer had the highest reward: {correct_chosen_answers}")
print(f"Accuracy: {accuracy:.2f}%")
