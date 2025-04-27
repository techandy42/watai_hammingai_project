#!/usr/bin/env python
"""Example script for the code obfuscator."""

import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from code_obfuscator import obfuscate_file

# Create a simple Python script to obfuscate
source_code = """
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
"""

# Write the source code to a temporary file
with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
    temp_file.write(source_code.encode('utf-8'))
    source_path = temp_file.name

# Create a destination file
dest_path = os.path.join(tempfile.gettempdir(), 'obfuscated_example.py')

# Obfuscate the file
obfuscate_file(source_path, dest_path)

print(f"Original file: {source_path}")
print(f"Obfuscated file: {dest_path}")

# Print the original and obfuscated code
print("\nOriginal code:")
print("="*80)
print(source_code)

print("\nObfuscated code:")
print("="*80)
with open(dest_path, 'r') as f:
    print(f.read())

# Clean up
os.unlink(source_path)
print(f"\nRun the obfuscated code with: python {dest_path}") 