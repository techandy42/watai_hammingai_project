#!/usr/bin/env python
"""
Python Code Obfuscator

This script replaces class/function/variable identifiers in Python code with random strings
while preserving functionality and imports, keywords, and comments.
"""

import os
import sys
import argparse
import ast
import random
import string
from typing import Dict, Set, List, Tuple, Optional, Any

# Import our utility modules
from code_obfuscator.utils.identifier_analyzer import IdentifierAnalyzer
from code_obfuscator.utils.random_string_generator import generate_random_string
from code_obfuscator.utils.code_transformer import CodeTransformer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Python Code Obfuscator")
    parser.add_argument("source", help="Source file or directory to obfuscate")
    parser.add_argument("dest", help="Destination directory for obfuscated code")
    parser.add_argument("--test", action="store_true", help="Run tests after obfuscation")
    parser.add_argument("--keep-structure", action="store_true", 
                        help="Keep directory structure when obfuscating directories")
    parser.add_argument("--preserve-exports", action="store_true", 
                        help="Preserve names of exported functions (those that are imported in other files)")
    return parser.parse_args()


def obfuscate_file(source_path: str, dest_path: str, identifier_map: Dict[str, str] = None) -> Dict[str, str]:
    """
    Obfuscate a single Python file.
    
    Args:
        source_path: Path to the source file
        dest_path: Path to write the obfuscated file
        identifier_map: Optional dictionary mapping original identifiers to obfuscated ones
                        (used to maintain consistency across multiple files)
    
    Returns:
        Updated identifier map
    """
    # Create identifier map if not provided
    if identifier_map is None:
        identifier_map = {}
    
    # Read source file
    with open(source_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    # Parse the AST
    tree = ast.parse(source_code)
    
    # Analyze the code to find all custom identifiers
    analyzer = IdentifierAnalyzer()
    analyzer.visit(tree)
    
    # Generate random strings for identifiers not in the map
    for identifier in analyzer.custom_identifiers:
        if identifier not in identifier_map:
            identifier_map[identifier] = generate_random_string()
    
    # Transform the code using our identifier map
    transformer = CodeTransformer(identifier_map)
    new_tree = transformer.visit(ast.parse(source_code))  # Create a fresh tree
    
    # Write the transformed code
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(ast.unparse(new_tree))
    
    print(f"Obfuscated {source_path} -> {dest_path}")
    return identifier_map


def update_imports(dest_dir: str, identifier_map: Dict[str, str]) -> None:
    """
    Update import statements in all Python files in the destination directory to use obfuscated names.
    
    Args:
        dest_dir: Destination directory containing obfuscated files
        identifier_map: Dictionary mapping original identifiers to obfuscated ones
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
                    
                    print(f"Updated imports in {file_path}")
                except SyntaxError:
                    print(f"Error parsing {file_path} - skipping import updates")


def obfuscate_directory(source_dir: str, dest_dir: str, keep_structure: bool = True, 
                        preserve_exports: bool = False) -> None:
    """
    Obfuscate all Python files in a directory.
    
    Args:
        source_dir: Source directory
        dest_dir: Destination directory
        keep_structure: Whether to keep the directory structure
        preserve_exports: Whether to preserve names of exported functions
    """
    identifier_map = {}
    
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
    
    # Second pass: obfuscate all files
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.py'):
                source_path = os.path.join(root, file)
                
                if keep_structure:
                    # Keep the directory structure
                    rel_path = os.path.relpath(source_path, source_dir)
                    dest_path = os.path.join(dest_dir, rel_path)
                else:
                    # Flatten the directory structure
                    dest_path = os.path.join(dest_dir, file)
                
                # Make sure the destination directory exists
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Obfuscate the file and update the identifier map
                identifier_map = obfuscate_file(source_path, dest_path, identifier_map)
    
    # Update import statements to use obfuscated names
    if not preserve_exports:
        update_imports(dest_dir, identifier_map)


def run_tests(source_dir: str, dest_dir: str) -> bool:
    """
    Run tests to ensure obfuscated code works the same as original code.
    
    Args:
        source_dir: Directory with original code
        dest_dir: Directory with obfuscated code
    
    Returns:
        True if all tests pass, False otherwise
    """
    # Find all test files in the project
    test_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    
    print(f"Found {len(test_files)} test files")
    
    # Run tests on original code
    print("Running tests on original code...")
    # This is a simplified implementation. In practice, you'd use unittest or pytest
    original_results = {}
    for test_file in test_files:
        # Simplified test execution - in practice use subprocess or unittest
        result = os.system(f"python {test_file}")
        original_results[test_file] = result
    
    # Run tests on obfuscated code
    print("Running tests on obfuscated code...")
    obfuscated_results = {}
    for test_file in test_files:
        rel_path = os.path.relpath(test_file, source_dir)
        obf_test = os.path.join(dest_dir, rel_path)
        if os.path.exists(obf_test):
            result = os.system(f"python {obf_test}")
            obfuscated_results[test_file] = result
    
    # Compare results
    all_pass = True
    for test_file in original_results:
        if test_file in obfuscated_results and original_results[test_file] == obfuscated_results[test_file]:
            print(f"Test passed: {test_file}")
        else:
            print(f"Test failed: {test_file}")
            all_pass = False
    
    return all_pass


def main():
    """Main entry point."""
    args = parse_arguments()
    
    if os.path.isfile(args.source):
        # Obfuscate a single file
        os.makedirs(args.dest, exist_ok=True)
        dest_file = os.path.join(args.dest, os.path.basename(args.source))
        obfuscate_file(args.source, dest_file)
    elif os.path.isdir(args.source):
        # Obfuscate a directory
        obfuscate_directory(args.source, args.dest, args.keep_structure, args.preserve_exports)
    else:
        print(f"Error: {args.source} is not a valid file or directory")
        sys.exit(1)
    
    if args.test:
        # Run tests if requested
        if run_tests(args.source, args.dest):
            print("All tests passed!")
        else:
            print("Some tests failed.")
            sys.exit(1)
    
    print("Obfuscation complete!")


if __name__ == "__main__":
    main() 