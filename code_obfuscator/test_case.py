#!/usr/bin/env python
"""
Test case for the code obfuscator.

This is a simple script that demonstrates the functionality of the code obfuscator.
"""

import sys
import os
import tempfile
import shutil

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from obfuscate import obfuscate_file


def main():
    """Run a test case for the code obfuscator."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create a source file
        source_file = os.path.join(temp_dir, 'source.py')
        with open(source_file, 'w') as f:
            f.write("""
# Example code to obfuscate
import math
from datetime import datetime

def calculate_area(radius):
    '''Calculate the area of a circle.'''
    return math.pi * radius ** 2

class Circle:
    def __init__(self, radius):
        self.radius = radius
        self.created_at = datetime.now()
    
    def area(self):
        return calculate_area(self.radius)
    
    def perimeter(self):
        return 2 * math.pi * self.radius
    
    def __str__(self):
        return f"Circle(radius={self.radius}, created_at={self.created_at})"

# Test the code
radius = 5
circle = Circle(radius)
print(f"Circle with radius {radius}:")
print(f"Area: {circle.area()}")
print(f"Perimeter: {circle.perimeter()}")
print(f"Object: {circle}")
""")
        
        # Create a destination file
        dest_file = os.path.join(temp_dir, 'obfuscated.py')
        
        # Obfuscate the file
        obfuscate_file(source_file, dest_file)
        
        print(f"Source file: {source_file}")
        print(f"Obfuscated file: {dest_file}")
        
        # Print the original file
        print("\nOriginal file content:")
        print("=" * 80)
        with open(source_file, 'r') as f:
            print(f.read())
        
        # Print the obfuscated file
        print("\nObfuscated file content:")
        print("=" * 80)
        with open(dest_file, 'r') as f:
            print(f.read())
        
        # Run both files and compare output
        print("\nRunning original file:")
        print("=" * 80)
        original_output = os.popen(f"{sys.executable} {source_file}").read()
        print(original_output)
        
        print("\nRunning obfuscated file:")
        print("=" * 80)
        obfuscated_output = os.popen(f"{sys.executable} {dest_file}").read()
        print(obfuscated_output)
        
        # Check if the outputs are the same
        if original_output == obfuscated_output:
            print("\nTest passed! The outputs are the same.")
        else:
            print("\nTest failed! The outputs are different.")
            print("\nOriginal output:")
            print(original_output)
            print("\nObfuscated output:")
            print(obfuscated_output)
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main() 