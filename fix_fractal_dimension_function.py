#!/usr/bin/env python
"""
This script fixes the duplicate implementation of the generate_fractal_dimension_vs_recursion
function in generate_paper_graphs.py file.

It reads the original file, removes duplicate implementations,
and writes the fixed content back to the original file.
"""

import re

def fix_generate_paper_graphs():
    """
    Fix the generate_paper_graphs.py file to ensure there's only one implementation
    of the generate_fractal_dimension_vs_recursion function.
    """
    # Read the original file content
    with open('generate_paper_graphs.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and remove any duplicate implementations
    # This pattern matches any function definition for generate_fractal_dimension_vs_recursion
    pattern = r'def generate_fractal_dimension_vs_recursion\(output_dir\):\s+""".*?""".*?plt\.close\(\)'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if len(matches) > 1:
        print(f"Found {len(matches)} implementations of generate_fractal_dimension_vs_recursion function")
        # Keep only the last (most complete) implementation
        keep_match = matches[-1]
        
        # Remove other implementations
        for match in matches[:-1]:
            start, end = match.span()
            content = content[:start] + content[end:]
            print(f"Removed duplicate implementation from positions {start}-{end}")
    else:
        print("No duplicate implementations found")
    
    # Write the fixed content back to the original file
    with open('generate_paper_graphs.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed generate_paper_graphs.py successfully")

if __name__ == "__main__":
    fix_generate_paper_graphs()
