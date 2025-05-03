"""
Identifier Analyzer for code obfuscation.

This module provides an AST visitor that identifies custom identifiers in Python code.
"""

import ast
import builtins
import keyword
from typing import Set, List, Dict, Any


class IdentifierAnalyzer(ast.NodeVisitor):
    """
    AST visitor that identifies all custom identifiers in Python code.
    
    This class visits all nodes in the AST and collects names of variables,
    functions, classes, and other user-defined identifiers.
    """
    
    def __init__(self):
        self.custom_identifiers: Set[str] = set()
        self.imported_identifiers: Set[str] = set()
        self.defined_identifiers: Set[str] = set()
        self.used_identifiers: Set[str] = set()
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
            
        return True
    
    def visit_Name(self, node):
        """Visit a Name node and collect identifier information."""
        if isinstance(node.ctx, ast.Store):
            # This is a variable being defined
            self.defined_identifiers.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            # This is a variable being used
            self.used_identifiers.add(node.id)
        
        # If this is a custom identifier, add it to our set
        if self.should_replace_identifier(node.id):
            self.custom_identifiers.add(node.id)
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Visit a function definition and collect the function name."""
        if self.should_replace_identifier(node.name):
            self.custom_identifiers.add(node.name)
        
        # Process function arguments
        for arg in node.args.args:
            if self.should_replace_identifier(arg.arg):
                self.custom_identifiers.add(arg.arg)
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit a class definition and collect the class name."""
        if self.should_replace_identifier(node.name):
            self.custom_identifiers.add(node.name)
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Visit an import statement and collect imported names."""
        for name in node.names:
            if name.asname:
                # Handle "import x as y"
                self.imported_identifiers.add(name.asname)
            else:
                # Handle "import x"
                self.imported_identifiers.add(name.name.split('.')[0])  # Just get the top-level module
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit an import-from statement and collect imported names."""
        if node.module is not None:
            for name in node.names:
                if name.asname:
                    # Handle "from x import y as z"
                    self.imported_identifiers.add(name.asname)
                else:
                    # Handle "from x import y"
                    self.imported_identifiers.add(name.name)
        
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """Visit an assignment and collect assigned names."""
        # Process the left-hand side (targets)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if self.should_replace_identifier(target.id):
                    self.custom_identifiers.add(target.id)
            elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                for elt in target.elts:
                    if isinstance(elt, ast.Name) and self.should_replace_identifier(elt.id):
                        self.custom_identifiers.add(elt.id)
        
        self.generic_visit(node)
    
    def visit_arg(self, node):
        """Visit a function argument and collect its name."""
        if self.should_replace_identifier(node.arg):
            self.custom_identifiers.add(node.arg)
        
        self.generic_visit(node)
        
    def analyze(self, code: str) -> Dict[str, Set[str]]:
        """
        Analyze Python code and return sets of identifiers.
        
        Args:
            code: Python source code
            
        Returns:
            Dictionary with sets of custom, imported, defined, and used identifiers
        """
        tree = ast.parse(code)
        self.visit(tree)
        
        # Remove imported identifiers from custom identifiers
        self.custom_identifiers -= self.imported_identifiers
        
        return {
            'custom': self.custom_identifiers,
            'imported': self.imported_identifiers,
            'defined': self.defined_identifiers,
            'used': self.used_identifiers
        } 