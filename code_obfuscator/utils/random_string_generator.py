"""
Random string generator for code obfuscation.

This module provides functions to generate random strings for replacing identifiers.
"""

import random
import string
from typing import Set


def generate_random_string(length: int = 6, existing_strings: Set[str] = None) -> str:
    """
    Generate a random string of specified length using uppercase letters and digits.
    
    Args:
        length: The length of the random string
        existing_strings: A set of strings that are already used (to avoid duplicates)
    
    Returns:
        A random string
    """
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


def generate_random_string_sequence(num_strings: int, length: int = 6) -> Set[str]:
    """
    Generate a set of unique random strings.
    
    Args:
        num_strings: Number of strings to generate
        length: Length of each string
    
    Returns:
        A set of unique random strings
    """
    result = set()
    
    while len(result) < num_strings:
        result.add(generate_random_string(length, result))
    
    return result 