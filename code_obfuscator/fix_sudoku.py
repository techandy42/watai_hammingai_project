#!/usr/bin/env python
"""
Script to demonstrate how to fix the Sudoku project using the code obfuscator.

This script shows two different approaches to fixing the import issues:
1. Preserving exported function names
2. Updating import statements

Both approaches should maintain the functionality of the original code.
"""

import os
import sys
import shutil
import ast
from pathlib import Path

# Helper function to update imports manually for a more direct approach
def update_imports_direct_fix(source_file, dest_file, import_map):
    """
    Update imports directly without using the AST parser.
    This is a simpler approach for quick fixes.
    
    Args:
        source_file: Path to the source file
        dest_file: Path to write the fixed file
        import_map: Dictionary mapping original import names to obfuscated names
    """
    with open(source_file, 'r') as f:
        content = f.read()
    
    # Replace import statements
    import_line = "from sudokutools import "
    original_imports = content.split(import_line)[1].split("\n")[0]
    
    # Create the new import statement with obfuscated names
    new_imports = []
    for original in original_imports.split(", "):
        if original in import_map:
            new_imports.append(import_map[original])
        else:
            new_imports.append(original)
    
    new_import_line = import_line + ", ".join(new_imports)
    content = content.replace(import_line + original_imports, new_import_line)
    
    # Fix common references
    content = content.replace("self.solvedBoard = ZBIFzA(self.board)", "self.solvedBoard = deepcopy(self.board)")
    content = content.replace("F3LZuP.randint", "random.randint")
    content = content.replace("Vsg6oz", "pygame")
    
    # Also fix references to imported functions within the code
    for original, obfuscated in import_map.items():
        # Replace function calls (making sure it's not part of a larger word)
        content = content.replace(f"{original}(", f"{obfuscated}(")
    
    # Fix method references
    content = content.replace("self.draw_board()", "self.bWm7xs()")
    content = content.replace("self.visualSolve(", "self.kDSmnZ(")
    content = content.replace("self.redraw(", "self.eBP5i9(")
    content = content.replace(".draw(", ".YLDkIb(")
    content = content.replace(".display(", ".qh1vJM(")
    content = content.replace(".clicked(", ".YTeHJP(")
    
    with open(dest_file, 'w') as f:
        f.write(content)

def main():
    # Paths
    root_dir = Path('/Users/sanskritiakhoury/Documents/watai_hammingai_project')
    sudoku_orig = root_dir / 'test_repos/Sudoku-Solver'
    sudoku_obfs = root_dir / 'test_repos/Sudoku-Solver-Obfuscated'
    
    print("Fixing the Sudoku-Solver-Obfuscated project...")
    
    # Approach 1: Manually fix imports in SudokuGUI.py
    print("\nApproach 1: Manually fixing imports")
    
    # Create a directory for our fixed solution
    fixed_dir = root_dir / 'test_repos/Sudoku-Solver-Fixed-Manual'
    if fixed_dir.exists():
        shutil.rmtree(fixed_dir)
    fixed_dir.mkdir(parents=True)
    
    # Copy the obfuscated files
    shutil.copy(sudoku_obfs / 'sudokutools.py', fixed_dir / 'sudokutools.py')
    
    # Map of original import names to obfuscated names (based on our analysis)
    import_map = {
        'valid': 'W5Z8uL',
        'solve': 'tsA87i',
        'find_empty': 'soidWX',
        'generate_board': 'wPYqII'
    }
    
    # Fix the imports in SudokuGUI.py
    update_imports_direct_fix(
        sudoku_obfs / 'SudokuGUI.py',
        fixed_dir / 'SudokuGUI.py',
        import_map
    )
    
    print(f"Files created in {fixed_dir}")
    print(f"To test: cd {fixed_dir} && python SudokuGUI.py")
    
    # Approach 2: Re-obfuscate with preserved exports
    print("\nApproach 2: Re-obfuscating with preserved exports")
    print("Note: This would require running the obfuscator with --preserve-exports")
    print("Example command: python -m code_obfuscator.obfuscate test_repos/Sudoku-Solver test_repos/Sudoku-Solver-Fixed --preserve-exports")
    
    print("\nDone! Two approaches have been demonstrated for fixing the import issues.")
    print("The first approach is a quick direct fix for this specific case.")
    print("The second approach is a more general solution using the obfuscator with preserved exports.")
    print("Both approaches should maintain the functionality of the original code.")

if __name__ == "__main__":
    main() 