import subprocess
import tempfile
import sys
import os
from typing import List, Tuple

def run_python_code(code: str) -> Tuple[List[str], int]:
    """
    Executes the provided Python source code and captures its output.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(code)
        temp_file.flush()  # Make sure to flush the buffer
    
    try:
        result = subprocess.run(
            [sys.executable, temp_file_name],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Print debug info
        print("\nProcess output:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        stdout_lines = result.stdout.splitlines()
        stderr_lines = result.stderr.splitlines()
        
        # Include error output if there was an error
        if result.returncode != 0:
            return stderr_lines, 0
            
        return stdout_lines, 1
    
    except subprocess.TimeoutExpired:
        return ["Error: Execution timed out."], 0
    except Exception as e:
        return [f"Error: {str(e)}"], 0
    finally:
        try:
            os.remove(temp_file_name)
        except OSError:
            pass
