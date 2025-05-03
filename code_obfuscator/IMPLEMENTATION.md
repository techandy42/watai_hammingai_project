# Implementation Details

This document describes the implementation details of the Python Code Obfuscator.

## Architecture

The code obfuscator consists of three main components:

1. **Identifier Analyzer**: Uses Python's AST module to scan the code and identify all custom identifiers that need to be replaced.
2. **Random String Generator**: Generates random strings to replace the identified custom identifiers.
3. **Code Transformer**: Transforms the code by replacing the identified custom identifiers with random strings.

## Key Features

- **Custom Identifier Detection**: The obfuscator detects all custom identifiers (variables, functions, classes) in the code that should be replaced.
- **Preservation of Imports**: The obfuscator does not replace imported modules and their identifiers.
- **Preservation of Python Keywords**: The obfuscator does not replace Python keywords (e.g., `import`, `for`, `def`).
- **Preservation of Comments**: The obfuscator keeps all comments intact, which is useful for debugging purposes.
- **Maintenance of Code Functionality**: The obfuscated code has the same functionality as the original code.
- **Consistent Replacement**: The same identifier is always replaced with the same random string throughout a project.
- **Cross-file Reference Handling**: The obfuscator provides two methods for handling imports between files in the same project.

## Implementation Details

### Identifier Analyzer

The `IdentifierAnalyzer` class is an AST visitor that analyzes Python code to identify custom identifiers that should be replaced. It visits all nodes in the AST and collects names of variables, functions, classes, and other user-defined identifiers.

```python
class IdentifierAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.custom_identifiers = set()
        self.imported_identifiers = set()
        self.defined_identifiers = set()
        self.used_identifiers = set()
        self.builtin_names = set(dir(builtins))
        self.python_keywords = set(keyword.kwlist)
        
        # Skip these names
        self.skip_names = self.builtin_names | self.python_keywords | {'self', 'cls'}
```

### Random String Generator

The `random_string_generator` module provides functions to generate random strings for replacing identifiers.

```python
def generate_random_string(length=6, existing_strings=None):
    if existing_strings is None:
        existing_strings = set()
    
    # Characters to use for random string generation
    chars = string.ascii_letters + string.digits
    
    # Generate random strings until we find one that's not in existing_strings
    while True:
        random_str = ''.join(random.choice(chars) for _ in range(length))
        
        # Make sure it doesn't start with a digit (invalid Python identifier)
        if random_str[0] not in string.digits and random_str not in existing_strings:
            return random_str
```

### Code Transformer

The `CodeTransformer` class is an AST transformer that replaces identifiers with random strings. It visits all nodes in the AST and replaces names of variables, functions, classes, and other user-defined identifiers with random strings.

```python
class CodeTransformer(ast.NodeTransformer):
    def __init__(self, identifier_map):
        self.identifier_map = identifier_map
        self.builtin_names = set(dir(builtins))
        self.python_keywords = set(keyword.kwlist)
        
        # Skip these names
        self.skip_names = self.builtin_names | self.python_keywords | {'self', 'cls'}
```

## Workflow

1. The obfuscator starts by parsing the Python source code into an AST.
2. The `IdentifierAnalyzer` visits all nodes in the AST and identifies all custom identifiers that need to be replaced.
3. The obfuscator generates a random string for each identified custom identifier.
4. The `CodeTransformer` transforms the code by replacing the identified custom identifiers with random strings.
5. The transformed AST is then converted back to Python code.

## Limitations

- **Dynamic Code Execution**: The obfuscator does not handle dynamic code execution (e.g., `eval()`, `exec()`) correctly.
- **Special Python Constructs**: The obfuscator might not handle all special Python constructs correctly.
- **AST Limitations**: The obfuscator is limited by the capabilities of Python's AST module.

## Cross-file Reference Handling

One of the key challenges in code obfuscation is handling references between multiple files. When functions or classes defined in one file are imported and used in another file, simply obfuscating the names can break the code. We've implemented two solutions to address this:

### Solution 1: Update Import Statements

This approach modifies all import statements to match the obfuscated names:

```python
def update_imports(dest_dir, identifier_map):
    """
    Update import statements in all Python files in the destination directory to use obfuscated names.
    """
    for root, _, files in os.walk(dest_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Read the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the AST
                try:
                    tree = ast.parse(content)
                    
                    # Find all import statements
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom):
                            # Handle "from x import y" statements
                            for name in node.names:
                                if name.name in identifier_map and not name.asname:
                                    # Replace import name with obfuscated name
                                    name.name = identifier_map[name.name]
                                elif name.asname in identifier_map:
                                    # Replace import alias with obfuscated name
                                    name.asname = identifier_map[name.asname]
                    
                    # Write the modified file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(ast.unparse(tree))
                except SyntaxError:
                    pass
```

This function is automatically called after obfuscation when the `--preserve-exports` flag is not used, ensuring all import statements match the obfuscated names.

### Solution 2: Preserve Exported Function Names

This approach preserves the names of functions and classes that are imported by other files:

```python
if preserve_exports:
    # First pass: analyze all files to find exported identifiers
    exported_identifiers = set()
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.py'):
                source_path = os.path.join(root, file)
                
                # Read source file
                with open(source_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # Find all imported identifiers
                tree = ast.parse(source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        # Add all imported names to the set of exported identifiers
                        for name in node.names:
                            exported_identifiers.add(name.name)
    
    # Map exported identifiers to themselves (preserve their names)
    for identifier in exported_identifiers:
        identifier_map[identifier] = identifier
```

This code performs a first pass over all files to identify functions and classes that are imported by other files. It then preserves their original names in the identifier map, effectively exempting them from obfuscation.

### Choosing Between Solutions

Both solutions have their advantages:

1. **Update Import Statements**: Provides more thorough obfuscation by changing all custom identifiers, but requires modifying import statements.
2. **Preserve Exported Function Names**: Maintains the original API of modules, making it easier to use the obfuscated code as part of a larger project, but leaves some identifiers unchanged.

The choice between these approaches depends on the specific requirements of the project. The obfuscator provides a command-line option `--preserve-exports` to choose the second approach.

## Future Improvements

- **Dynamic Code Execution**: Add support for dynamic code execution.
- **Special Python Constructs**: Add support for more special Python constructs.
- **Better Random String Generation**: Improve the random string generation algorithm to generate more readable random strings.
- **More Obfuscation Techniques**: Add support for more obfuscation techniques, such as control flow obfuscation, dead code insertion, etc.
- **Performance Optimization**: Improve performance for large codebases by optimizing the AST traversal and transformation process.
- **Enhanced Cross-file Analysis**: Further refine the cross-file reference detection to handle more complex import patterns.
- **IDE Integration**: Create plugins for popular IDEs to facilitate the obfuscation process directly from the development environment. 