import re
import textwrap
from typing import List, Dict, Any


class CodegenPrompts:
    # XML format is more suitable for Llama and Claude models
    @staticmethod
    def get_codegen_prompt(text: str, io_struct: Dict[str, Any]):
        nl = "\n"
        nltab = "\n\t"
        
        prompt = f"""
        Question:
        {text}

        Additional Instructions:
        - The output should be a valid Python code that wouldn't crash when ran.
        - The name of the function of your program that serves as the entry point should be named: {io_struct["function_name"]}
        - The parameters should have the following structure: {str(io_struct["input"])} ({len(io_struct["input"])} parameters in total)
        - The output should have the following structure: {str(io_struct["output"])}
        {f"- The output should be one of the following values: {str(io_struct['specific_output_values'])}{nl}" if io_struct["specific_output"] else ""}
        Warnings:
        - Do not include any type annotations in the input parameters.
        - Do not include any unit tests or example usage.
        - Do not include any uncommented non-code text.
        - Do not include Python tags (e.g. ```python ...some code ... ```) in any part of the code.
        """
        
        return textwrap.dedent(prompt)

    @staticmethod
    def get_io_struct_extraction_prompt_markdown(test_list: List[str]) -> Dict[str, str]:
        prompt_instruction = f"""
        Instruction:
        - Given the following list of assert statement, extract the following information for the function that is being tested.
        - "function_name": Name of the function being tested (e.g. "merge_sort")
        - "inputs": Valid Python data structures in list of literal string format (e.g. ["List[str]", "int"])
        - "output": Valid Python data structure in literal string format (e.g. "str")
        - "specific_output": True or False (True only if the function outputs only a specific set of string values that act as flags, such as "Passed" or "Not Passed". Set of string values that is directly computed from the input values, and do not act as flags, should not be considered as True. For example, if the function outputs "hll" for given input "hello", the answer should be False)
        - "specific_output_values": List of specific output values, return empty list if "specific_output" is False (e.g. ["Passed", "Not Passed"])
        - Follow the provided structured output JSON format style for your output.
        """

        assert_list = ["Assert Statements:", *test_list]

        assert_lines = "\n".join(assert_list)

        prompt = textwrap.dedent(prompt_instruction) + "\n" + assert_lines

        return prompt

class CodevalTemplates:
    @staticmethod
    def extract_function_call(assert_statement: str) -> str:
        """
        Extracts the function call from an assert statement and wraps it with print().
        
        Example:
            Input: "assert max_chain_length([Pair(5, 24), Pair(15, 25)], 4) == 3"
            Output: "print(max_chain_length([Pair(5, 24), Pair(15, 25)], 4))"
        """
        # Define a regex pattern to capture the function call part
        pattern = r'assert\s+(.+?)\s*==\s*.+'
        match = re.match(pattern, assert_statement)
        
        if match:
            function_call = match.group(1).strip()
            return f"print({function_call})"
        else:
            # Handle cases where the pattern does not match
            # You can choose to raise an error, return None, or handle differently
            raise ValueError(f"Invalid assert statement format: '{assert_statement}'")
    
    @staticmethod
    def get_codeval_template(code: str, test_list_1: List[str], test_list_2: List[str]) -> str:
        """
        Generates a code template by combining the user-provided code with test cases.
        
        Args:
            code (str): The user's original code.
            test_list_1 (List[str]): The first list of assert statements.
            test_list_2 (List[str]): The second list of assert statements.
        
        Returns:
            str: The combined code template ready for execution.
        """

        nl = "\n"
        nltab = "\n    "

        test_list = test_list_1 + test_list_2

        only_test_list = []

        for test in test_list:
            try:
                only_test_list.append(CodevalTemplates.extract_function_call(test))
            except ValueError as e:
                print(e)
        
        template_print_list = []
        
        for test in only_test_list:
            formatted_test = f'try:{nltab}{test}{nl}except Exception as e:{nltab}print(e)'
            template_print_list.append(formatted_test)

        template_print_lines = f'{nl.join(template_print_list)}'
        
        template_assert_lines = f'{nl.join(test_list)}'
        
        template_parts = [code, template_print_lines, template_assert_lines]
        
        template = "\n\n".join(template_parts)
        
        return template
