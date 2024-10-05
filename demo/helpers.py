import re
from typing import List

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

def split_assert_statements(test_list: List[str]) -> List[str]:
    fixed_test_list = []
    
    for test in test_list:
        fixed_test_list.extend(['assert ' + t.strip() for t in test.split('assert') if t.strip()])

    return fixed_test_list
