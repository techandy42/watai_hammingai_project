from enum import Enum
from typing import Any, Dict, ClassVar
from pydantic import BaseModel, ValidationError

class RoleEnum(str, Enum):
    internal = "internal"
    external = "external"

class ChoiceEnum(str, Enum):
    a = "a"
    b = "b"
    c = "c"
    d = "d"

class BaselineQuestion(BaseModel):
    question: str
    role: RoleEnum

    @classmethod
    def is_valid(cls, obj: Dict[str, Any]) -> bool:
        try:
            cls(**obj)
            return True
        except ValidationError:
            return False

class BaselineAnswer(BaseModel):
    answer: str

    @classmethod
    def is_valid(cls, obj: Dict[str, Any]) -> bool:
        try:
            cls(**obj)
            return True
        except ValidationError:
            return False

class BaselineRank(BaseModel):
    choice: ChoiceEnum
    choice_mapping: ClassVar[Dict[str, int]] = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3
    }

    @classmethod
    def is_valid(cls, obj: Dict[str, Any]) -> bool:
        try:
            cls(**obj)
            return True
        except ValidationError:
            return False

    def map_choice_to_number(self) -> int:
        return self.choice_mapping[self.choice]

# Example Usage:
if __name__ == "__main__":
    # BaselineQuestion
    question_data = {
        "question": "What is your favorite color?",
        "role": "internal"
    }
    print(BaselineQuestion.is_valid(question_data))  # Output: True

    # BaselineAnswer
    answer_data = {
        "answer": "Blue"
    }
    print(BaselineAnswer.is_valid(answer_data))  # Output: True

    # BaselineRank
    rank_data = {
        "choice": "b"
    }
    if BaselineRank.is_valid(rank_data):
        rank = BaselineRank(**rank_data)
        print(rank.map_choice_to_number())  # Output: 1
