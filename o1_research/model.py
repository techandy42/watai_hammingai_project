import logging
import json
from typing import List
import time
from typing import Optional
import concurrent.futures
from demo.request import make_request, make_request_structured_output
from o1_research.prompts import O1BaselinePrompts
from o1_research.thought_chain import ThoughtChain, Thought
from o1_research.response_format import BaselineQuestion, BaselineAnswer, BaselineRank
from o1_research.helpers import count_tokens, calc_cost

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
class O1BaselineModel:
    exceed_token_limit = "exceeded_token_limit"
    other_exception = "other_exception"

    def __init__(
        self, 
        request_id: str, 
        models: List[str],  # List of model names
        context_limit: int, 
        token_limit: int, 
        initial_question: str, 
        system_message: Optional[str] = None, 
        interactive: bool = False,
        validation_retries: int = 3  # Added parameter for retries
    ):
        self.request_id = request_id
        self.thought_chain = ThoughtChain(
            initial_question=initial_question, 
            system_message=system_message
        )
        self.models = models  # List of model names
        self.context_limit = context_limit
        self.token_limit = token_limit
        self.interactive = interactive
        self.token_limit_buffer = 1000
        self.input_token_count = 0
        self.output_token_count = 0
        self.validation_retries = validation_retries  # Store the retries parameter

    def think_v1(self) -> str:
        if self.interactive:
            logging.info("=" * 100)
            logging.info("Initial Problem:")
            logging.info("=" * 100)
            logging.info(f"{self.thought_chain.initial_question}")
            logging.info("=" * 100)

        thought_idx = 0

        while not self.thought_chain.is_thinking_done():
            if self.thought_chain.total_path_token_count() > self.token_limit - self.token_limit_buffer:
                return O1BaselineModel.exceed_token_limit
            try:
                if self.interactive:
                    logging.info("\n")
                    logging.info("=" * 100)
                    logging.info(f"Reasoning Step No.{thought_idx+1}")
                    logging.info("=" * 100)

                thought = Thought()
                self.thought_chain.add_thought(thought)

                # Determine the question prompt
                if self.thought_chain.is_empty():
                    question_prompt = O1BaselinePrompts.get_initial_question_prompt(self.thought_chain)
                else:
                    question_prompt = O1BaselinePrompts.get_followup_question_prompt(self.thought_chain)

                # Generate questions from each model
                def generate_question(model):
                    if "gpt-4o" in model:
                        question_dict = make_request_structured_output(
                            model=model,
                            messages=[{"role": "user", "content": question_prompt}],
                            response_format=BaselineQuestion,
                            max_tokens=self.context_limit
                        )
                        return question_dict
                    else:
                        # Retry logic for models using make_request
                        attempt = 0
                        while attempt < self.validation_retries:
                            response = make_request(
                                model=model,
                                messages=[{"role": "user", "content": question_prompt}],
                                max_tokens=self.context_limit,
                                json_mode=True
                            )
                            try:
                                validated_response = BaselineQuestion.parse_raw(response)
                                question_dict = validated_response.dict()
                                return question_dict  # Successful parsing
                            except Exception as e:
                                logging.error(
                                    f"Attempt {attempt+1}/{self.validation_retries}: Failed to parse response from model {model}. "
                                    f"Response: {response}. Error: {e}"
                                )
                                attempt += 1
                        # If all attempts fail, return a default error dict
                        return {"question": "Parsing Error after retries", "role": "invalid"}

                # Generate questions concurrently
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = list(executor.map(generate_question, self.models))
                for i, question_dict in enumerate(futures):
                    thought.add_question(question_dict['question'])
                    thought.add_role(question_dict['role'])
                    self.input_token_count += count_tokens(question_prompt)
                    self.output_token_count += count_tokens(json.dumps(question_dict))
                    if self.interactive:
                        logging.info(
                            f"Question No.{i+1} from model '{self.models[i]}': {question_dict['question']} "
                            f"(role: {question_dict['role']})"
                        )

                # Rank the generated questions
                rank_question_prompt = O1BaselinePrompts.get_rank_question_prompt(self.thought_chain)
                question_rank_dict = make_request_structured_output(
                    model=self.models[0],  # Using the first model for ranking
                    messages=[{"role": "user", "content": rank_question_prompt}],
                    response_format=BaselineRank,
                    max_tokens=self.context_limit
                )
                chosen_question_idx = BaselineRank(**question_rank_dict).map_choice_to_number()
                thought.choose_question(chosen_question_idx)
                self.input_token_count += count_tokens(rank_question_prompt)
                self.output_token_count += count_tokens(json.dumps(question_rank_dict))
                if self.interactive:
                    logging.info(
                        f"Best Question: {thought.get_question()} (Question No.{chosen_question_idx+1} from model "
                        f"'{self.models[chosen_question_idx]}')"
                    )

                # Determine the answer prompt
                if thought.get_role() == Thought.internal:
                    answer_prompt = O1BaselinePrompts.get_internal_answer_prompt(self.thought_chain)
                else:
                    if self.thought_chain.system_message is not None:
                        answer_prompt = O1BaselinePrompts.get_external_answer_system_message_prompt(self.thought_chain)
                    else:
                        answer_prompt = O1BaselinePrompts.get_external_answer_prompt(self.thought_chain)

                # Generate answers from each model
                def generate_answer(model):
                    if "gpt-4o" in model:
                        answer_dict = make_request_structured_output(
                            model=model,
                            messages=[{"role": "user", "content": answer_prompt}],
                            response_format=BaselineAnswer,
                            max_tokens=self.context_limit
                        )
                        return answer_dict
                    else:
                        # Retry logic for models using make_request
                        attempt = 0
                        while attempt < self.validation_retries:
                            response = make_request(
                                model=model,
                                messages=[{"role": "user", "content": answer_prompt}],
                                max_tokens=self.context_limit,
                                json_mode=True
                            )
                            try:
                                validated_response = BaselineAnswer.parse_raw(response)
                                answer_dict = validated_response.dict()
                                return answer_dict  # Successful parsing
                            except Exception as e:
                                logging.error(
                                    f"Attempt {attempt+1}/{self.validation_retries}: Failed to parse response from model {model}. "
                                    f"Response: {response}. Error: {e}"
                                )
                                attempt += 1
                        # If all attempts fail, return a default error dict
                        return {"answer": "Parsing Error after retries"}

                # Generate answers concurrently
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = list(executor.map(generate_answer, self.models))
                for i, answer_dict in enumerate(futures):
                    thought.add_answer(answer_dict["answer"])
                    self.input_token_count += count_tokens(answer_prompt)
                    self.output_token_count += count_tokens(json.dumps(answer_dict))
                    if self.interactive:
                        logging.info(
                            f"Answer No.{i+1} from model '{self.models[i]}': {answer_dict['answer']}"
                        )

                # Rank the generated answers
                rank_answer_prompt = O1BaselinePrompts.get_rank_answer_prompt(self.thought_chain)
                answer_rank_dict = make_request_structured_output(
                    model=self.models[0],  # Using the first model for ranking
                    messages=[{"role": "user", "content": rank_answer_prompt}],
                    response_format=BaselineRank,
                    max_tokens=self.context_limit
                )
                chosen_answer_idx = BaselineRank(**answer_rank_dict).map_choice_to_number()
                thought.choose_answer(chosen_answer_idx)
                self.input_token_count += count_tokens(rank_answer_prompt)
                self.output_token_count += count_tokens(json.dumps(answer_rank_dict))
                if self.interactive:
                    logging.info(
                        f"Best Answer: {thought.get_answer()} (Answer No.{chosen_answer_idx+1} from model "
                        f"'{self.models[chosen_answer_idx]}')"
                    )
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

    def save_result(self):
        thoughts_data = []
        for thought in self.thought_chain.chain: 
            thought_data = {
                'questions': thought.questions,
                'role': thought.roles,
                'chosen_question_idx': thought.chosen_question_idx,
                'chosen_question': thought.get_question(),
                'answers': thought.answers,
                'chosen_answer_idx': thought.chosen_answer_idx,
                'chosen_answer': thought.get_answer(),
            }
            thoughts_data.append(thought_data)
        
        initial_question = self.thought_chain.initial_question
        system_message = self.thought_chain.system_message
        response = self.thought_chain.get_final_answer()
        base_model = self.models
        context_limit = self.context_limit
        token_limit = self.token_limit
        interactive = self.interactive
        result = {
            'id': self.request_id,
            'initial_question': initial_question,
            'system_message': system_message,
            'response': response,
            'thoughts': thoughts_data,
            'base_model': base_model,
            'context_limit': context_limit,
            'token_limit': token_limit,
            'interactive': interactive,
        }

        return result

def initialize_models_from_jsonl(file_path: str) -> List[O1BaselineModel]:
    models = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            thought_chain = ThoughtChain(initial_question=data['initial_question'], system_message=None)
            for thought_data in data['thoughts']:
                thought = Thought()
                for question, role, answer in zip(thought_data['questions'], thought_data['role'], thought_data['answers']):
                    thought.add_question(question)
                    thought.add_role(role)
                    thought.add_answer(answer)
                thought.choose_question(thought_data['chosen_question_idx'])
                thought.choose_answer(thought_data['chosen_answer_idx'])
                thought_chain.add_thought(thought)
            model = O1BaselineModel(
                request_id=data['id'],
                models=data['base_model'],
                context_limit=data['context_limit'],
                token_limit=data['token_limit'],
                initial_question=data['initial_question'],
                system_message=data['system_message'],
                interactive=data['interactive']
            )
            model.thought_chain = thought_chain
            models.append(model)
    return models

if __name__ == "__main__":
    request_id = "test_1"
    initial_question = "Write a python function to find the first repeated character in a given string."
    system_message = "Only include Python code in your output, do not include any comments or tags."
    models = [
        "gpt-4o-2024-08-06",
        "claude-3-5-sonnet-20240620",
        "gemini/gemini-1.5-pro",
        "gpt-4o-mini-2024-07-18"
    ]  # List of 4 models
    context_limit = 8192
    token_limit = 8192
    interactive = True
    price_per_mill_input = 0.15
    price_per_mill_output = 0.6
    validation_retries = 3  # Set the number of retries

    model = O1BaselineModel(
        request_id=request_id,
        models=models,  # Passing the list of models
        context_limit=context_limit,
        token_limit=token_limit,
        initial_question=initial_question,
        system_message=system_message,
        interactive=interactive,
        validation_retries=validation_retries  # Pass the retries parameter
    )

    start_time = time.time()
    response = model.think_v1()
    execution_time = time.time() - start_time

    input_token_count = model.input_token_count
    output_token_count = model.output_token_count
    total_cost = calc_cost(input_token_count, output_token_count, price_per_mill_input, price_per_mill_output)

    print()
    print("=" * 100)
    print(f"**Execution Time: {execution_time:.2f} seconds")
    print(f"**Total Cost: ${total_cost:.4f}")
    print(f"Input Token Count: {input_token_count}")
    print(f"Output Token Count: {output_token_count}")
    print("=" * 100)
    print("Final Response:")
    print(response)
    print("=" * 100)
