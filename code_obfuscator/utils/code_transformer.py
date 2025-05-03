"""
Code Transformer for obfuscation.

This module provides an AST transformer that replaces identifiers with random strings.
"""

import ast
import builtins
import keyword
from typing import Dict, Set, Any


class CodeTransformer(ast.NodeTransformer):
    """
    AST transformer that replaces identifiers with random strings.
    
    This class visits all nodes in the AST and replaces names of variables,
    functions, classes, and other user-defined identifiers with random strings.
    """
    
    def __init__(self, identifier_map: Dict[str, str]):
        """
        Initialize the transformer with an identifier map.
        
        Args:
            identifier_map: Dictionary mapping original identifiers to obfuscated ones
        """
        self.identifier_map = identifier_map
        self.builtin_names = set(dir(builtins))
        self.python_keywords = set(keyword.kwlist)
        
        # Skip these names
        self.skip_names = self.builtin_names | self.python_keywords | {'self', 'cls'}
    
    def should_replace_identifier(self, name: str) -> bool:
        """
        Determine if an identifier should be replaced.
        
        Args:
            name: The identifier name
            
        Returns:
            True if the identifier should be replaced, False otherwise
        """
        # Skip Python builtins, keywords, and special names
        if name in self.skip_names:
            return False
        
        # Skip names with double underscore (dunder methods)
        if name.startswith('__') and name.endswith('__'):
            return False
        
        # Skip protected and private attributes/methods
        if name.startswith('_'):
            return False
            
        # Only replace if in our map
        return name in self.identifier_map
    
    def visit_Name(self, node):
        """Visit a Name node and replace its identifier if necessary."""
        # Replace the identifier if it's in our map
        if self.should_replace_identifier(node.id):
            node.id = self.identifier_map[node.id]
        
        return self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Visit a function definition and replace its name if necessary."""
        # Replace the function name if it's in our map
        if self.should_replace_identifier(node.name):
            node.name = self.identifier_map[node.name]
        
        # Process function arguments
        node.args = self.visit(node.args)
        
        # Process function body
        for i, item in enumerate(node.body):
            node.body[i] = self.visit(item)
        
        # Process decorators
        for i, decorator in enumerate(node.decorator_list):
            node.decorator_list[i] = self.visit(decorator)
        
        return node
    
    def visit_ClassDef(self, node):
        """Visit a class definition and replace its name if necessary."""
        # Replace the class name if it's in our map
        if self.should_replace_identifier(node.name):
            node.name = self.identifier_map[node.name]
        
        # Process class body
        for i, item in enumerate(node.body):
            node.body[i] = self.visit(item)
        
        # Process bases
        for i, base in enumerate(node.bases):
            node.bases[i] = self.visit(base)
        
        # Process decorators
        for i, decorator in enumerate(node.decorator_list):
            node.decorator_list[i] = self.visit(decorator)
        
        return node
    
    def visit_arg(self, node):
        """Visit a function argument and replace its name if necessary."""
        # Replace the argument name if it's in our map
        if self.should_replace_identifier(node.arg):
            node.arg = self.identifier_map[node.arg]
        
        # Process annotation if present
        if node.annotation:
            node.annotation = self.visit(node.annotation)
        
        return node
    
    def visit_Attribute(self, node):
        """Visit an attribute and replace its name if necessary."""
        # We don't replace attributes because they might be from imported modules
        # Just process the value part (e.g., in `obj.attr`, process `obj`)
        node.value = self.visit(node.value)
        
        return node
    
    def visit_Import(self, node):
        """Visit an import statement and keep it unchanged."""
        # We don't replace imported module names
        return node
    
    def visit_ImportFrom(self, node):
        """Visit an import-from statement and keep it unchanged."""
        # We don't replace imported module or function names
        return node 