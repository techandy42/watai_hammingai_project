import textwrap

class CodegenPrompts:
    @staticmethod
    def get_codegen_prompt(text: str, function_name: str):
        prompt = f"""
        <question>
        {text}
        </question>

        <additional_instructions>
            <instruction>The name of the function of your program that serves as the entry point should be named: {function_name}</instruction>
            <instruction>The output format should be a valid Python code. </instruction>
            <instruction>Do not include any unit tests or example usage.</instruction>
            <instruction>Do not include any uncommented non-code text that will crash the program when it's ran.</instruction>
        </additional_instructions>
        """
        
        return textwrap.dedent(prompt)
