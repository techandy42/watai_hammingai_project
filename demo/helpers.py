import re

def get_tested_function(assert_statements):
    """
    Extracts the name of the function being tested from a list of assert statements.

    Parameters:
    assert_statements (list of str): List containing assert statements as strings.

    Returns:
    str or None: The name of the tested function if found, otherwise None.
    """
    # Regular expression pattern to capture the function name after 'assert'
    pattern = re.compile(r'assert\s+(\w+)\s*\(')
    
    for stmt in assert_statements:
        match = pattern.search(stmt)
        if match:
            return match.group(1)
    
    return None

def remove_python_code_tags(input_text):
    """
    Removes leading ```python and trailing ``` tags from a code block,
    along with any surrounding spaces and newlines.

    Parameters:
    input_text (str): The input string containing the code block.

    Returns:
    str: The code without the surrounding ```python and ``` tags,
         stripped of extra whitespace.
    """
    # Strip leading and trailing whitespace first
    stripped_text = input_text.strip()

    # Remove the leading ```python along with any following spaces/newlines
    stripped_text = re.sub(r'^```python\s*\n?', '', stripped_text, flags=re.IGNORECASE)

    # Remove the trailing ``` along with any preceding spaces/newlines
    stripped_text = re.sub(r'\n?```\s*$', '', stripped_text)

    return stripped_text.strip()
