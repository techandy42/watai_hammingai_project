from typing import Literal, List, Optional
from helpers import count_tokens

class Thought:
    internal = "internal"
    external = "external"

    def __init__(self):
        self.questions = []
        self.answers = []
        self.chosen_question_idx = None
        self.chosen_answer_idx = None
        self.chosen_question_token_count = None
        self.chosen_answer_token_count = None
        self.role = None # Thought.internal or Thought.external

    def add_question(self, question: str) -> None:
        self.questions.append(question)

    def add_answer(self, answer: str) -> None:
        self.answers.append(answer)

    def choose_question(self, idx: int) -> bool:
        if idx >= len(self.questions):
            return False
        self.chosen_question_idx = idx
        self.chosen_question_token_count = count_tokens(self.questions[idx])

    def choose_answer(self, idx: int) -> bool:
        if idx >= len(self.answers):
            return False
        self.chosen_answer_idx = idx
        self.chosen_answer_token_count = count_tokens(self.answers[idx])

    def set_role(self, role: Literal["internal", "external"]) -> None:
        self.role = role

    def get_question(self) -> Optional[str]:
        if self.chosen_question_idx is None:
            return None
        if self.chosen_question_idx >= len(self.questions):
            return None
        return self.questions[self.chosen_question_idx]
    
    def get_answer(self) -> Optional[str]:
        if self.chosen_answer_idx is None:
            return None
        if self.chosen_answer_idx >= len(self.answers):
            return None
        return self.answers[self.chosen_answer_idx]

class ThoughtChain:
    def __init__(self, initial_question: str):
        self.initial_question = initial_question
        self.chain: List[Thought] = []

    def add_thought(self, thought: Thought) -> None:
        self.chain.append(thought)

    def is_empty(self) -> bool:
        return len(self.chain) == 0

    def is_thinking_done(self) -> bool:
        last_thought = self.chain[-1]
        if last_thought.role == Thought.external and last_thought.get_answer() is not None:
            return True
        return False
    
    def get_final_answer(self) -> Optional[str]:
        if self.is_thinking_done():
            return self.chain[-1].get_answer()
        return None
    
    def total_token_count(self) -> int:
        token_count = 0
        for thought in self.chain:
            chosen_question_token_count = thought.chosen_question_token_count
            if chosen_question_token_count is not None:
                token_count += chosen_question_token_count
            chosen_answer_token_count = thought.chosen_answer_token_count
            if chosen_answer_token_count is not None:
                token_count += chosen_answer_token_count
        return token_count
