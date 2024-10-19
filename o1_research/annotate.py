import logging
import json
from tqdm import tqdm
from datasets import load_dataset
from o1_research.model import O1BaselineModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

if __name__ == "__main__":
    dataset = load_dataset('google-research-datasets/mbpp')
    test_dataset = dataset["test"]
    df_test = test_dataset.to_pandas()
    
    system_message = "Only include Python code in your output, do not include any comments or tags."
    base_model = "gpt-4o-mini-2024-07-18"  
    context_limit = 8192
    token_limit = 8192
    interactive = False 
    
    results = []
    
    for index, row in tqdm(df_test.head(2).iterrows()): 
        request_id = f"mbpp_{row['task_id']}"
        initial_question = row['text']
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
            logging.error(f"Exception for index {index}: {str(e)}")
        result = model.save_result()
        results.append(result)
    
    with open('mbpp_results.jsonl', 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

### REFERENCE CODE TO SAVE RESULT AS CSV
# save to csv file (whichever is preferred)

# with open('results.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     fieldnames = ['index', 'initial_question', 'response', 'execution_time', 'token_count',
#                   'thought_idx', 'questions', 'roles', 'chosen_question_idx', 'chosen_question',
#                   'answers', 'chosen_answer_idx', 'chosen_answer']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()

#     for result in results:
#         index = result['index']
#         initial_question = result['initial_question']
#         response = result['response']
#         execution_time = result['execution_time']
#         token_count = result['token_count']
#         thoughts = result['thoughts']
#         for thought_idx, thought in enumerate(thoughts):
#             row = {
#                 'index': index,
#                 'initial_question': initial_question,
#                 'response': response,
#                 'execution_time': execution_time,
#                 'token_count': token_count,
#                 'thought_idx': thought_idx,
#                 'questions': ' | '.join(thought['questions']) if thought['questions'] else '',
#                 'roles': ' | '.join(thought['roles']) if thought.get('roles') else '',
#                 'chosen_question_idx': thought['chosen_question_idx'],
#                 'chosen_question': thought['chosen_question'],
#                 'answers': ' | '.join(thought['answers']) if thought['answers'] else '',
#                 'chosen_answer_idx': thought['chosen_answer_idx'],
#                 'chosen_answer': thought['chosen_answer'],
#             }
#             writer.writerow(row)
### REFERENCE CODE ENDS
