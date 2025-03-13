#!/usr/bin/env python
"""
Script to fix the syntax error in the generate_paper_graphs.py file.
It corrects the plt.savefig line in the generate_fractal_dimension_vs_recursion function
by adding missing commas between arguments.
"""

import re

def fix_syntax_error():
    # Read the generate_paper_graphs.py file
    with open('generate_paper_graphs.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and fix the syntax error pattern
    pattern = r'plt\.savefig\(output_dir / "fractal_dim_vs_recursion\.png" dpi=300 bbox_inches=\'tight\'\)'
    replacement = r'plt.savefig(output_dir / "fractal_dim_vs_recursion.png", dpi=300, bbox_inches=\'tight\')'
    
    # Replace the pattern
    fixed_content = re.sub(pattern, replacement, content)
    
    # Write the fixed content back to the file
    with open('generate_paper_graphs.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Fixed the syntax error in generate_paper_graphs.py")

if __name__ == "__main__":
    fix_syntax_error()
