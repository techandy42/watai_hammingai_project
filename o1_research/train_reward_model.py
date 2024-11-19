import json
from itertools import combinations

def generate_pairwise_data(file_path):
    pairwise_data = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            initial_question = data['initial_question']
            for thought in data['thoughts']:
                questions = thought['questions']
                chosen_question_idx = thought['chosen_question_idx']
                answers = thought['answers']
                chosen_answer_idx = thought['chosen_answer_idx']

                # get pairwise comparisons for questions
                for i, j in combinations(range(len(questions)), 2):
                    label = 1 if i == chosen_question_idx else 0
                    pairwise_data.append({
                        'context': initial_question,
                        'query_a': questions[i],
                        'query_b': questions[j],
                        'label': label  # 1 if query_a is preferred over query_b
                    })

                # get pairwise comparisons for answers
                for i, j in combinations(range(len(answers)), 2):
                    label = 1 if i == chosen_answer_idx else 0
                    pairwise_data.append({
                        'context': initial_question + " " + thought['chosen_question'],
                        'response_a': answers[i],
                        'response_b': answers[j],
                        'label': label  # 1 if response_a is preferred over response_b
                    })
    return pairwise_data

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from datasets import Dataset
import numpy as np

def train_reward_model(pairwise_data, model_name='bert-base-uncased'):
    # prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = []
    labels = []
    for item in pairwise_data:
        # combine context and response for both options
        text_a = item['context'] + " " + item.get('query_a', '') + " " + item.get('response_a', '')
        text_b = item['context'] + " " + item.get('query_b', '') + " " + item.get('response_b', '')
        texts.append((text_a, text_b))
        labels.append(float(item['label']))  # float

    dataset = Dataset.from_dict({
        'text_a': [t[0] for t in texts],
        'text_b': [t[1] for t in texts],
        'label': labels
    })

    def tokenize_function(examples):
        inputs = tokenizer(examples['text_a'], examples['text_b'], padding='max_length', truncation=True, max_length=512)
        # float32
        inputs['labels'] = np.array(examples['label'], dtype=np.float32)
        return inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    config = AutoConfig.from_pretrained(model_name, num_labels=1, problem_type="regression")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # args
    training_args = TrainingArguments(
        output_dir='./reward_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    # train
    trainer.train()
    # save
    model.save_pretrained('./reward_model')
    tokenizer.save_pretrained('./reward_model')

pairwise_data = generate_pairwise_data('hybrid_test_mbpp_results_0_to_49_successful.jsonl')

train_reward_model(pairwise_data, model_name='bert-base-uncased')
