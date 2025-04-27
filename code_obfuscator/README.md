# Python Code Obfuscator

A tool for obfuscating Python code by replacing identifiers with random strings.

## Purpose

This tool replaces identifiers (variable, function, and class names) in Python code with random strings. This is useful for testing if there is pre-trained bias with LLMs when evaluating code benchmarks.

## Features

- Replaces class/function/variable names defined within the repo itself
- Preserves Python standard keywords (e.g., `import`, `for`, `def`, etc.)
- Preserves imported packages and their classes/functions/variables
- Preserves comments for debugging purposes
- Maintains functionality of the obfuscated code
- Two methods for handling cross-file imports:
  1. Automatically updating import statements to match obfuscated names
  2. Preserving exported function names while obfuscating their implementations

## Installation

```bash
git clone https://github.com/your-username/code-obfuscator.git
cd code-obfuscator
pip install -e .
```

## Usage

### Obfuscating a Single File

```bash
python -m code_obfuscator.obfuscate source_file.py destination_directory/
```

### Obfuscating a Directory

```bash
python -m code_obfuscator.obfuscate source_directory/ destination_directory/ --keep-structure
```

### Preserving Exported Function Names

To preserve the names of functions that are imported by other files (recommended approach):

```bash
python -m code_obfuscator.obfuscate source_directory/ destination_directory/ --preserve-exports
```

### Updating Import Statements

Alternatively, you can update the import statements to match the obfuscated names (this happens by default if `--preserve-exports` is not specified):

```bash
python -m code_obfuscator.obfuscate source_directory/ destination_directory/
```

### Running Tests After Obfuscation

```bash
python -m code_obfuscator.obfuscate source_directory/ destination_directory/ --test
```

## Example

Original code:

```python
import requests
from bs4 import BeautifulSoup

def get_all_links(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        for a_tag in soup.find_all('a', href=True):
            links.append(a_tag['href'])
        return links
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []
```

Obfuscated code:

```python
import requests
from bs4 import BeautifulSoup

def abdhf(zhfjdy):
    try:
        rfjgs = requests.get(zhfjdy)
        rfjgs.raise_for_status()  # Raise an error for bad status codes
        qwxpf = BeautifulSoup(rfjgs.text, 'html.parser')
        gkhptod = []
        for bmngj in qwxpf.find_all('a', href=True):
            gkhptod.append(bmngj['href'])
        return gkhptod
    except requests.RequestException as hvieo:
        print(f"Error fetching {zhfjdy}: {hvieo}")
        return []
```

## Testing with Multi-File Projects

When working with multi-file projects where files import from each other, you have two options:

1. **Preserve Exported Function Names** (recommended):
   ```bash
   python -m code_obfuscator.obfuscate source_directory/ destination_directory/ --preserve-exports
   ```
   This preserves the names of functions that are imported by other files, ensuring that import statements continue to work.

2. **Update Import Statements**:
   ```bash
   python -m code_obfuscator.obfuscate source_directory/ destination_directory/
   ```
   This updates import statements in all files to use the obfuscated function names.

Both approaches maintain the functionality of the original code while obfuscating the implementations.

## Testing

The obfuscator can automatically run tests to ensure that the obfuscated code maintains the same functionality as the original code:

```bash
python -m code_obfuscator.obfuscate source_directory/ destination_directory/ --test
```

This will look for test files (files starting with `test_`) in the source directory and run them on both the original and obfuscated code, comparing the results.

## Limitations

- The obfuscator may not handle dynamic code evaluation correctly (e.g., `eval()`, `exec()`).
- The obfuscator might not handle all special Python constructs.
- The obfuscator is limited by the capabilities of Python's AST module.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 