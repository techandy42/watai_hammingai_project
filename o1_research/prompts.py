from o1_research.helpers import format_prompt
from o1_research.thought_chain import Thought, ThoughtChain

class O1BaselinePrompts:
    @staticmethod
    def get_initial_question_prompt(thought_chain: ThoughtChain, include_json_format: bool) -> str:
        json_format = """
        \"\"\"
        Output Format (JSON):
        {
            "question": str,
            "role": Literal["internal", "external"]
        }
        \"\"\"
        """
        
        prompt = f"""
        Instruction:
        \"\"\"
        - Your job is to break the given problem into multiple steps of thoughts.
        - If the problem is very simple and does not require complex reasoning, output a question that would ask to directly solve the problem.
        - Otherwise, output the first step of reasoning that you would take to solve the problem, in form of a question.
        - Only return one question, and make sure to not answer your own question in your output.
        - "question": Your question.
        - "role": "internal" (if you are asking for the first step of reasoning) or "external" (if you are asking to directly solve the problem).
        \"\"\"
        {json_format if include_json_format else ""}
        Example:
        \"\"\"
        - Problem: ... some complex coding question ...
        - Output: What is the expected behaviour of the code?
        \"\"\"

        Problem:
        \"\"\"
        {thought_chain.initial_question}
        \"\"\"
        """

        return format_prompt(prompt)
    
    @staticmethod
    def get_followup_question_prompt(thought_chain: ThoughtChain, include_json_format: bool) -> str:
        json_format = """
        \"\"\"
        Output Format (JSON):
        {
            "question": str,
            "role": Literal["internal", "external"]
        }
        \"\"\"
        """
        
        prompt = f"""
        Instruction:
        \"\"\"
        - Your job is to break the given problem into multiple steps of thoughts.
        - The following is the initial problem, and the previous steps of reasoning.
        - If the previous steps of reasoning is sufficient to answer the question, output a question that would ask to finally solve the problem.
        - Otherwise, output the next step of reasoning that you would take to solve the problem, in form of a question.
        - Only return one question, and make sure to not answer your own question in your output.
        - "question": Your question.
        - "role": "internal" (if you are asking for the next step of reasoning) or "external" (if you are asking to finally solve the problem).
        \"\"\"
        {json_format if include_json_format else ""}
        Example No.1:
        \"\"\"
        - Problem: ... some complex coding question ...
        - Previous Steps: ... sufficient reasoning to answer the problem ...
        - Output: Given the above reasoning, what is the final solution to the problem?
        \"\"\"
        
        Example No.2:
        \"\"\"
        - Problem: ... some complex coding question ...
        - Previous Steps: ... insufficient reasoning to answer the problem ...
        - Output: How would you implement ABC algorithm to solve XYZ?
        \"\"\"

        Problem:
        \"\"\"
        {thought_chain.initial_question}
        \"\"\"
        """

        prompt = format_prompt(prompt)

        for i, thought in enumerate(thought_chain.chain):
            previous_step = f"""
            Previous Step No.{i+1}:
            \"\"\"
            - Question: {thought.get_question()}
            - Answer: {thought.get_answer()}
            \"\"\"
            """
            previous_step = format_prompt(previous_step)
            prompt += ("\n\n" + previous_step)

        return prompt
    
    @staticmethod
    def get_internal_answer_prompt(thought_chain: ThoughtChain, include_json_format: bool) -> str:
        json_format = """
        \"\"\"
        Output Format (JSON):
        {
            "answer": str
        }
        \"\"\"
        """

        prompt = f"""
        Instruction:
        \"\"\"
        - Your job is to answer the current question to the best of your ability.
        - Reference initial problem statement and the previous steps of reasoning to answer the current question.
        - Only return one answer, and make sure to not include anything else in your output.
        - If you are unsure about the answer, indicate in your response that you are unsure.
        - "answer": Your answer.
        \"\"\"
        {json_format if include_json_format else ""}
        Initial Problem Statement:
        \"\"\"
        {thought_chain.initial_question}
        \"\"\"
        """

        prompt = format_prompt(prompt)

        for i, thought in enumerate(thought_chain.chain):
            previous_step = f"""
            Previous Step No.{i+1}:
            \"\"\"
            - Question: {thought.get_question()}
            - Answer: {thought.get_answer()}
            \"\"\"
            """
            previous_step = format_prompt(previous_step)
            prompt += ("\n\n" + previous_step)

        question = f"""
        Current Question:
        \"\"\"
        - {thought_chain.chain[-1].get_question()}
        \"\"\"
        """

        question = format_prompt(question)
        prompt += ("\n\n" + question)

        return prompt
    
    @staticmethod
    def get_external_answer_prompt(thought_chain: ThoughtChain, include_json_format: bool) -> str:
        json_format = """
        \"\"\"
        Output Format (JSON):
        {
            "answer": str
        }
        \"\"\"
        """

        prompt = f"""
        Instruction:
        \"\"\"
        - Your job is to answer the current question that would solve the initial problem statement.
        - Reference initial problem statement and the previous steps of reasoning to answer the current question.
        - Only return one answer, and make sure to not include anything else in your output.
        - "answer": Your answer.
        \"\"\"
        {json_format if include_json_format else ""}
        Initial Problem Statement:
        \"\"\"
        {thought_chain.initial_question}
        \"\"\"
        """

        prompt = format_prompt(prompt)

        for i, thought in enumerate(thought_chain.chain):
            previous_step = f"""
            Previous Step No.{i+1}:
            \"\"\"
            - Question: {thought.get_question()}
            - Answer: {thought.get_answer()}
            \"\"\"
            """
            previous_step = format_prompt(previous_step)
            prompt += ("\n\n" + previous_step)

        question = f"""
        Current Question:
        \"\"\"
        - {thought_chain.chain[-1].get_question()}
        \"\"\"
        """

        question = format_prompt(question)
        prompt += ("\n\n" + question)

        return prompt
    
    @staticmethod
    def get_external_answer_system_message_prompt(thought_chain: ThoughtChain, include_json_format: bool) -> str:
        json_format = """
        \"\"\"
        Output Format (JSON):
        {
            "answer": str
        }
        \"\"\"
        """

        prompt = f"""
        Instruction:
        \"\"\"
        - Your job is to answer the current question that would solve the initial problem statement.
        - Reference initial problem statement and the previous steps of reasoning to answer the current question.
        - Only return one answer, and make sure to not include anything else in your output.
        - Reference the system message for the specification for your answer for the current question.
        - "answer": Your answer.
        \"\"\"
        {json_format if include_json_format else ""}
        Initial Problem Statement:
        \"\"\"
        {thought_chain.initial_question}
        \"\"\"
        """

        prompt = format_prompt(prompt)

        for i, thought in enumerate(thought_chain.chain):
            previous_step = f"""
            Previous Step No.{i+1}:
            \"\"\"
            - Question: {thought.get_question()}
            - Answer: {thought.get_answer()}
            \"\"\"
            """
            previous_step = format_prompt(previous_step)
            prompt += ("\n\n" + previous_step)

        question = f"""
        System Message:
        \"\"\"
        - {thought_chain.system_message}
        \"\"\"

        Current Question:
        \"\"\"
        - {thought_chain.chain[-1].get_question()}
        \"\"\"
        """

        question = format_prompt(question)
        prompt += ("\n\n" + question)

        return prompt

    @staticmethod
    def get_rank_question_prompt(thought_chain: Thought, include_json_format: bool) -> str:
        json_format = """
        \"\"\"
        Output Format (JSON):
        {
            "best": Literal["a", "b", "c", "d"],
            "second": Literal["a", "b", "c", "d"],
            "third": Literal["a", "b", "c", "d"],
            "worst": Literal["a", "b", "c", "d"]
        }
        \"\"\"
        """

        prompt = f"""
        Instruction:
        \"\"\"
        - The following are the initial problem statement, the previous steps of reasoning, and the current questions.
        - The current questions that ask to solve the initial problem statement using the previous steps of reasoning are listed below with labels: a, b, c, d
        - Return the rank of the labels of the current questions in order of best to worst, where "best" is the best question, "second" is the second best question, "third" is the third best question, and "worst" is the worst question.
        - During your judgement, consider the role of each question.
            - You should favor a question with (role: internal) if you think there needs to be additional steps of reasoning to the solve the initial problem statement.
            - Otherwise, you should favor a question with (role: external) if you think the problem is very simple, or the previous steps of reasoning provide enough information to solve the initial problem statement. 
        - "best": "a", "b", "c", or "d" (only one of the following labels)
        - "second": "a", "b", "c", or "d" (only one of the following labels)
        - "third": "a", "b", "c", or "d" (only one of the following labels)
        - "worst": "a", "b", "c", or "d" (only one of the following labels)
        \"\"\"
        {json_format if include_json_format else ""}
        Initial Problem Statement:
        \"\"\"
        {thought_chain.initial_question}
        \"\"\"
        """

        prompt = format_prompt(prompt)

        for i, thought in enumerate(thought_chain.chain):
            previous_step = f"""
            Previous Step No.{i+1}:
            \"\"\"
            - Question: {thought.get_question()}
            - Answer: {thought.get_answer()}
            \"\"\"
            """
            previous_step = format_prompt(previous_step)
            prompt += ("\n\n" + previous_step)

        questions = f"""
        Current Questions:
        \"\"\"
        - a: {thought_chain.chain[-1].questions[0]} (role: {thought_chain.chain[-1].roles[0]})
        - b: {thought_chain.chain[-1].questions[1]} (role: {thought_chain.chain[-1].roles[1]})
        - c: {thought_chain.chain[-1].questions[2]} (role: {thought_chain.chain[-1].roles[2]})
        - d: {thought_chain.chain[-1].questions[3]} (role: {thought_chain.chain[-1].roles[3]})
        \"\"\"
        """

        questions = format_prompt(questions)
        prompt += ("\n\n" + questions)

        return prompt

    @staticmethod
    def get_rank_answer_prompt(thought_chain: Thought, include_json_format: bool) -> str:
        json_format = """
        \"\"\"
        Output Format (JSON):
        {
            "best": Literal["a", "b", "c", "d"],
            "second": Literal["a", "b", "c", "d"],
            "third": Literal["a", "b", "c", "d"],
            "worst": Literal["a", "b", "c", "d"]
        }
        \"\"\"
        """

        prompt = f"""
        Instruction:
        \"\"\"
        - The following are the initial problem statement, the previous steps of reasoning, the current question, and the current answers to the current question.
        - The current answers that aim to answer the current question are listed below with labels: a, b, c, d
        - Return the rank of the labels of the current answers in order of best to worst, where "best" is the best answer, "second" is the second best answer, "third" is the third best answer, and "worst" is the worst answer.
        - "best": "a", "b", "c", or "d" (only one of the following labels)
        - "second": "a", "b", "c", or "d" (only one of the following labels)
        - "third": "a", "b", "c", or "d" (only one of the following labels)
        - "worst": "a", "b", "c", or "d" (only one of the following labels)
        \"\"\"
        {json_format if include_json_format else ""}
        Initial Problem Statement:
        \"\"\"
        {thought_chain.initial_question}
        \"\"\"
        """

        prompt = format_prompt(prompt)

        for i, thought in enumerate(thought_chain.chain):
            previous_step = f"""
            Previous Step No.{i+1}:
            \"\"\"
            - Question: {thought.get_question()}
            - Answer: {thought.get_answer()}
            \"\"\"
            """
            previous_step = format_prompt(previous_step)
            prompt += ("\n\n" + previous_step)

        question = f"""
        Current Question:
        \"\"\"
        - {thought_chain.chain[-1].get_question()}
        \"\"\"
        """

        question = format_prompt(question)
        prompt += ("\n\n" + question)

        answers = f"""
        Current Answers:
        \"\"\"
        - a: {thought_chain.chain[-1].answers[0]}
        - b: {thought_chain.chain[-1].answers[1]}
        - c: {thought_chain.chain[-1].answers[2]}
        - d: {thought_chain.chain[-1].answers[3]}
        \"\"\"
        """

        answers = format_prompt(answers)
        prompt += ("\n\n" + answers)

        return prompt


if __name__ == "__main__":
    # Test 1: get_initial_question_prompt
    thought_chain_1 = ThoughtChain(initial_question="What is 2 + 2?")
    initial_prompt_1 = O1BaselinePrompts.get_initial_question_prompt(thought_chain_1, include_json_format=True)
    print("Test 1: Initial Question Prompt")
    print(initial_prompt_1)

    # Test 2: get_followup_question_prompt with one thought
    thought_chain_2 = ThoughtChain(initial_question="How can I reverse a linked list?")
    thought_1 = Thought()
    thought_1.add_question("What data structure is the linked list?")
    thought_1.add_answer("The linked list is singly linked.")
    thought_1.add_role(Thought.internal)  # Set role
    thought_1.choose_question(0)
    thought_1.choose_answer(0)
    thought_chain_2.add_thought(thought_1)
    question_prompt_2 = O1BaselinePrompts.get_followup_question_prompt(thought_chain_2, include_json_format=True)
    print("\nTest 2: Follow-up Question Prompt with One Thought")
    print(question_prompt_2)

    # Test 3: get_followup_question_prompt with two thoughts
    thought_chain_3 = ThoughtChain(initial_question="How can I optimize matrix multiplication?", system_message="Answer in LaTeX format.")
    
    thought_2 = Thought()
    thought_2.add_question("What is the size of the matrices?")
    thought_2.add_answer("The matrices are 3x3.")
    thought_2.add_role(Thought.internal)
    thought_2.choose_question(0)
    thought_2.choose_answer(0)
    
    thought_3 = Thought()
    thought_3.add_question("Can I use Strassen's algorithm?")
    thought_3.add_answer("Yes, Strassen's algorithm can be applied.")
    thought_3.add_role(Thought.external)
    thought_3.choose_question(0)
    thought_3.choose_answer(0)
    
    thought_chain_3.add_thought(thought_2)
    thought_chain_3.add_thought(thought_3)
    
    question_prompt_3 = O1BaselinePrompts.get_followup_question_prompt(thought_chain_3, include_json_format=True)
    print("\nTest 3: Follow-up Question Prompt with Two Thoughts")
    print(question_prompt_3)

    # Test 4: get_internal_answer_prompt
    internal_answer_prompt = O1BaselinePrompts.get_internal_answer_prompt(thought_chain_3, include_json_format=True)
    print("\nTest 4: Internal Answer Prompt")
    print(internal_answer_prompt)

    # Test 5: get_external_answer_system_message_prompt
    external_answer_prompt = O1BaselinePrompts.get_external_answer_system_message_prompt(thought_chain_3, include_json_format=True)
    print("\nTest 5: External Answer Prompt (System Message)")
    print(external_answer_prompt)

    # Test 6: get_rank_question_prompt
    thought_chain_3.chain[-1].add_question("How would you multiply two matrices?")
    thought_chain_3.chain[-1].add_role(Thought.internal)
    
    thought_chain_3.chain[-1].add_question("What is Strassen's algorithm?")
    thought_chain_3.chain[-1].add_role(Thought.internal)
    
    thought_chain_3.chain[-1].add_question("What is matrix multiplication?")
    thought_chain_3.chain[-1].add_role(Thought.external)
    
    thought_chain_3.chain[-1].add_question("Can Strassen's algorithm be applied?")
    thought_chain_3.chain[-1].add_role(Thought.external)
    
    rank_question_prompt = O1BaselinePrompts.get_rank_question_prompt(thought_chain_3, include_json_format=True)
    print("\nTest 6: Rank Question Prompt")
    print(rank_question_prompt)

    # Test 7: get_rank_answer_prompt
    thought_chain_3.chain[-1].add_answer("Answer A")
    thought_chain_3.chain[-1].add_answer("Answer B")
    thought_chain_3.chain[-1].add_answer("Answer C")
    thought_chain_3.chain[-1].add_answer("Answer D")
    
    rank_answer_prompt = O1BaselinePrompts.get_rank_answer_prompt(thought_chain_3, include_json_format=True)
    print("\nTest 7: Rank Answer Prompt")
    print(rank_answer_prompt)
