import subprocess
import tempfile
import sys
from typing import List, Tuple

def run_python_code(code: str) -> Tuple[List[str], int]:
    """
    Executes the provided Python source code and captures its output.

    Parameters:
    - code (str): Multiline string containing Python source code.

    Returns:
    - Tuple[List[str], int]: A tuple containing:
        1. A list of strings, each representing a line of standard output.
        2. An integer status code where:
            - 1 indicates the program ran successfully.
            - 0 indicates an error occurred (the program crashed).
    """
    # Create a temporary file to store the Python code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(code)
    
    try:
        # Execute the Python script using subprocess
        # sys.executable ensures the same Python interpreter is used
        result = subprocess.run(
            [sys.executable, temp_file_name],
            capture_output=True,
            text=True,
            timeout=10  # Optional: prevent hanging indefinitely
        )
        
        # Split the standard output into a list of lines
        stdout_lines = result.stdout.splitlines()
        
        # Determine the status code based on the return code
        # According to the requirement:
        # 1 if the program ran successfully (returncode == 0)
        # 0 if there was an error (returncode != 0)
        status_code = 1 if result.returncode == 0 else 0
        
        return stdout_lines, status_code
    
    except subprocess.TimeoutExpired:
        # Handle cases where the subprocess times out
        return [f"Error: Execution timed out."], 0
    
    except Exception as e:
        # Handle other unforeseen exceptions
        return [f"Error: {str(e)}"], 0
    
    finally:
        # Clean up the temporary file
        try:
            import os
            os.remove(temp_file_name)
        except OSError:
            pass
