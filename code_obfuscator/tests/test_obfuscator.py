"""Test the code obfuscator."""

import os
import tempfile
import unittest
import sys
import subprocess
import shutil

# Add parent directory to path so we can import the code_obfuscator package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from code_obfuscator import obfuscate_file, obfuscate_directory


class TestCodeObfuscator(unittest.TestCase):
    """Test the code obfuscator."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directories for source and destination
        self.temp_dir = tempfile.mkdtemp()
        self.source_dir = os.path.join(self.temp_dir, 'source')
        self.dest_dir = os.path.join(self.temp_dir, 'dest')
        os.mkdir(self.source_dir)
        os.mkdir(self.dest_dir)
        
        # Create a simple Python file for testing
        self.test_file = os.path.join(self.source_dir, 'test.py')
        with open(self.test_file, 'w') as f:
            f.write("""
# Simple test file
import math

def calculate_area(radius):
    '''Calculate the area of a circle.'''
    return math.pi * radius ** 2

class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return calculate_area(self.radius)
    
    def perimeter(self):
        return 2 * math.pi * self.radius

# Test the code
radius = 5
circle = Circle(radius)
print(f"Circle with radius {radius}:")
print(f"Area: {circle.area()}")
print(f"Perimeter: {circle.perimeter()}")
""")
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_obfuscate_file(self):
        """Test obfuscating a single file."""
        dest_file = os.path.join(self.dest_dir, 'test.py')
        
        # Obfuscate the file
        obfuscate_file(self.test_file, dest_file)
        
        # Check that the destination file exists
        self.assertTrue(os.path.exists(dest_file))
        
        # Check that the obfuscated file is different from the original
        with open(self.test_file, 'r') as f:
            original_content = f.read()
        with open(dest_file, 'r') as f:
            obfuscated_content = f.read()
        self.assertNotEqual(original_content, obfuscated_content)
        
        # Run both files and compare the output
        original_output = subprocess.check_output([sys.executable, self.test_file]).decode('utf-8')
        obfuscated_output = subprocess.check_output([sys.executable, dest_file]).decode('utf-8')
        self.assertEqual(original_output, obfuscated_output)
    
    def test_obfuscate_directory(self):
        """Test obfuscating an entire directory."""
        # Create another file in a subdirectory
        subdir = os.path.join(self.source_dir, 'subdir')
        os.mkdir(subdir)
        subdir_file = os.path.join(subdir, 'subdir_test.py')
        with open(subdir_file, 'w') as f:
            f.write("""
# Another test file
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(f"Factorial of 5 is {result}")
""")
        
        # Obfuscate the directory
        obfuscate_directory(self.source_dir, self.dest_dir, keep_structure=True)
        
        # Check that the destination files exist
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, 'test.py')))
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, 'subdir', 'subdir_test.py')))
        
        # Run the original and obfuscated files and compare the output
        original_output = subprocess.check_output([sys.executable, subdir_file]).decode('utf-8')
        obfuscated_output = subprocess.check_output([
            sys.executable, os.path.join(self.dest_dir, 'subdir', 'subdir_test.py')
        ]).decode('utf-8')
        self.assertEqual(original_output, obfuscated_output)


if __name__ == '__main__':
    unittest.main() 