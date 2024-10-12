import logging
import time
from typing import Optional
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

    # Model must support structured output
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

                # Generating Questions:
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
                
                # Ranking Questions:
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
                
                # Generate Answers:
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

                # Ranking Answer:
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
    # Define the parameters for the O1BaselineModel
    initial_question = "Write a python function to find the first repeated character in a given string."
    system_message = "Only include Python code in your output, do not include any comments or tags."
    base_model = "gpt-4o-mini-2024-07-18"  # Replace this with your actual base model
    context_limit = 8192
    token_limit = 8192
    interactive = True

    # Initialize the O1BaselineModel
    model = O1BaselineModel(
        base_model=base_model,
        context_limit=context_limit,
        token_limit=token_limit,
        initial_question=initial_question,
        system_message=system_message,
        interactive=interactive
    )

    # Run the think_v1 method and get the result
    start_time = time.time()
    response = model.think_v1()
    execution_time = time.time() - start_time
    token_count = model.thought_chain.total_generated_token_count()

    # Print final response
    print()
    print("=" * 100)
    print("Final Response")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Total Token Count: {token_count}")
    print("=" * 100)
    print(response)
    print("=" * 100)
