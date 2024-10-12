from request import make_request_structured_output
from prompts import O1BaselinePrompts
from thought_chain import ThoughtChain, Thought
from response_format import BaselineQuestion, BaselineAnswer, BaselineRank

class O1BaselineModel:
    exceed_token_limit = "exceeded_token_limit"
    openai_error = "openai_error"

    def __init__(self, initial_question: str, base_model: str, token_limit: int, interactive: bool = False):
        self.thought_chain = ThoughtChain(initial_question=initial_question)
        self.base_model = base_model
        self.token_limit = token_limit
        self.interactive = interactive

    # Model must support structured output
    def think_v1(self) -> str:
        while not self.thought_chain.is_thinking_done():
            if self.thought_chain.total_token_count() > self.token_limit:
                return O1BaselineModel.exceed_token_limit
            try:
                thought = Thought()
                question_prompt = None
                if self.thought_chain.is_empty():
                    question_prompt = O1BaselinePrompts.get_initial_question_prompt(self.thought_chain)
                else:
                    question_prompt = O1BaselinePrompts.get_followup_question_prompt(self.thought_chain)
                question_dict = make_request_structured_output(
                    model=self.base_model,
                    messages=[{"role": "system", "content": question_prompt}],
                    response_format=BaselineQuestion,
                    max_tokens=self.token_limit
                )
                # Continue from here
            except Exception as e:
                print(str(e))
                return O1BaselineModel.openai_error
            self.thought_chain.add_thought(thought)
        return self.thought_chain.get_final_answer()
