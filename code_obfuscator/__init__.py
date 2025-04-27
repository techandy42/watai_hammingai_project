"""Python Code Obfuscator.

This package provides tools to obfuscate Python code by replacing identifiers with random strings.
"""

from .obfuscate import obfuscate_file, obfuscate_directory, run_tests

__version__ = '0.1.0'
__author__ = 'Your Name'

__all__ = [
    'obfuscate_file',
    'obfuscate_directory',
    'run_tests',
] 