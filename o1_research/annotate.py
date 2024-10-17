import logging
import time
import csv
import json
from typing import Optional
from datasets import load_dataset
from request import make_request_structured_output
from prompts import O1BaselinePrompts
from thought_chain import ThoughtChain, Thought
from response_format import BaselineQuestion, BaselineAnswer, BaselineRank

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class O1BaselineModel:
    exceed_token_limit = "exceeded_token_limit"
    other_exception = "other_exception"

    def __init__(self, base_model: str, context_limit: int, token_limit: int, initial_question: str, system_message: Optional[str] = None, interactive: bool = False):
        self.thought_chain = ThoughtChain(initial_question=initial_question, system_message=system_message)
        self.base_model = base_model
        self.context_limit = context_limit
        self.token_limit = token_limit
        self.interactive = interactive

    # structured output
    def think_v1(self) -> str:
        if self.interactive:
            logging.info("=" * 100)
            logging.info("Initial Problem:")
            logging.info("=" * 100)
            logging.info(f"{self.thought_chain.initial_question}")
            logging.info("=" * 100)

        thought_idx = 0

        while not self.thought_chain.is_thinking_done():
            if self.thought_chain.total_path_token_count() > self.token_limit:
                return O1BaselineModel.exceed_token_limit
            try:
                if self.interactive:
                    logging.info("\n")
                    logging.info("=" * 100)
                    logging.info(f"Reasoning Step No.{thought_idx+1}")
                    logging.info("=" * 100)

                thought = Thought()
                self.thought_chain.add_thought(thought)

                # generate questions
                question_prompt = None
                if self.thought_chain.is_empty():
                    question_prompt = O1BaselinePrompts.get_initial_question_prompt(self.thought_chain)
                else:
                    question_prompt = O1BaselinePrompts.get_followup_question_prompt(self.thought_chain)
                for i in range(4):
                    question_dict = make_request_structured_output(
                        model=self.base_model,
                        messages=[{"role": "user", "content": question_prompt}],
                        response_format=BaselineQuestion,
                        max_tokens=self.context_limit
                    )
                    thought.add_question(question_dict['question'])
                    thought.add_role(question_dict['role'])

                    if self.interactive:
                        logging.info(f"Question No.{i+1}: {question_dict['question']} (role: {question_dict['role']})")
                
                # rank questions
                rank_question_prompt = O1BaselinePrompts.get_rank_question_prompt(self.thought_chain)
                question_rank_dict = make_request_structured_output(
                    model=self.base_model,
                    messages=[{"role": "user", "content": rank_question_prompt}],
                    response_format=BaselineRank,
                    max_tokens=self.context_limit
                )
                chosen_question_idx = BaselineRank(**question_rank_dict).map_choice_to_number()
                thought.choose_question(chosen_question_idx)

                if self.interactive:
                    logging.info(f"Best Question: {thought.get_question()} (Question No.{chosen_question_idx+1})")
                
                # generate answers
                answer_prompt = None
                if thought.get_role() == Thought.internal:
                    answer_prompt = O1BaselinePrompts.get_internal_answer_prompt(self.thought_chain)
                else:
                    if self.thought_chain.system_message is not None:
                        answer_prompt = O1BaselinePrompts.get_external_answer_system_message_prompt(self.thought_chain)
                    else:
                        answer_prompt = O1BaselinePrompts.get_external_answer_prompt(self.thought_chain)
                for i in range(4):
                    answer_dict = make_request_structured_output(
                        model=self.base_model,
                        messages=[{"role": "user", "content": answer_prompt}],
                        response_format=BaselineAnswer,
                        max_tokens=self.context_limit
                    )
                    thought.add_answer(answer_dict["answer"])

                    if self.interactive:
                        logging.info(f"Answer No.{i+1}: {answer_dict['answer']}")

                # rank the answers
                rank_answer_prompt = O1BaselinePrompts.get_rank_answer_prompt(self.thought_chain)
                answer_rank_dict = make_request_structured_output(
                    model=self.base_model,
                    messages=[{"role": "user", "content": rank_answer_prompt}],
                    response_format=BaselineRank,
                    max_tokens=self.context_limit
                )
                chosen_answer_idx = BaselineRank(**answer_rank_dict).map_choice_to_number()
                thought.choose_answer(chosen_answer_idx)

                if self.interactive:
                    logging.info(f"Best Answer: {thought.get_answer()} (Answer No.{chosen_answer_idx+1})")
                    logging.info("=" * 100)

                thought_idx += 1

            except Exception as e:
                logging.error(str(e))
                return O1BaselineModel.other_exception

        if self.interactive:
            logging.info("\n")
            logging.info("=" * 100)
            logging.info("Final Solution")
            logging.info("=" * 100)
            logging.info(f"{self.thought_chain.get_final_answer()}")
            logging.info("=" * 100)

        return self.thought_chain.get_final_answer()

if __name__ == "__main__":
    # load dataset for mbpp (change to desired dataset with correct column here)
    dataset = load_dataset('google-research-datasets/mbpp')
    test_dataset = dataset["test"]
    df_test = test_dataset.to_pandas()
    
    # parameters copied from model.py
    system_message = "Only include Python code in your output, do not include any comments or tags."
    base_model = "gpt-4o-mini-2024-07-18"  
    context_limit = 8192
    token_limit = 8192
    interactive = False 
    
    results = []
    
    for index, row in df_test.head(1).iterrows(): 
        # only did one row 
        # change the number here for the desired number of rows to iterate through
        initial_question = row['text']
        # initialize
        model = O1BaselineModel(
            base_model=base_model,
            context_limit=context_limit,
            token_limit=token_limit,
            initial_question=initial_question,
            system_message=system_message,
            interactive=interactive
        )
        # run think_v1
        start_time = time.time()
        try:
            response = model.think_v1()
        except Exception as e:
            logging.error(f"Exception for index {index}: {str(e)}")
            response = None
        execution_time = time.time() - start_time
        token_count = model.thought_chain.total_generated_token_count()
        
        # get thoughts
        thoughts_data = []
        for thought in model.thought_chain.chain: 
            thought_data = {
                'questions': thought.questions,
                'role': thought.roles,
                'chosen_question_idx': thought.chosen_question_idx,
                'chosen_question': thought.get_question(),  # chosen question?
                'answers': thought.answers,
                'chosen_answer_idx': thought.chosen_answer_idx,
                'chosen_answer': thought.get_answer(),  # chosen answer
            }
            thoughts_data.append(thought_data)
        
        result = {
            'index': index,
            'initial_question': initial_question,
            'response': response,
            'execution_time': execution_time,
            'token_count': token_count,
            'thoughts': thoughts_data
        }
        results.append(result)
    
    # save to json file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)



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