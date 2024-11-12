from enum import Enum
from typing import Any, Dict, ClassVar, List
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
    best: ChoiceEnum
    second: ChoiceEnum
    third: ChoiceEnum
    worst: ChoiceEnum
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

    def map_rankings_to_numbers(self) -> List[int]:
        return [self.choice_mapping[self.best], self.choice_mapping[self.second], self.choice_mapping[self.third], self.choice_mapping[self.worst]]

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
        "best": "c",
        "second": "a",
        "third": "b",
        "worst": "d"
    }
    print(BaselineRank.is_valid(rank_data))  # Output: True
    print(BaselineRank(**rank_data).map_rankings_to_numbers())  # Output: [0, 1, 2, 3]
