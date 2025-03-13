#!/usr/bin/env python
"""Script to fix line 524 in generate_paper_graphs.py"""

import re

def fix_line_524():
    # Read the entire file
    with open('generate_paper_graphs.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Line 524 is index 523 (zero-based indexing)
    line_idx = 523
    if line_idx < len(lines):
        original_line = lines[line_idx]
        print(f"Original line 524: {original_line.strip()}")
        
        # Create a fixed line with proper commas
        if 'plt.savefig' in original_line and '.png"' in original_line and 'dpi=' in original_line:
            # Replace missing commas - precise patching
            fixed_line = original_line.replace('png"', 'png",').replace('dpi=300 ', 'dpi=300, ')
            
            # Update the line
            lines[line_idx] = fixed_line
            print(f"Fixed line 524: {fixed_line.strip()}")
            
            # Write the modified file
            with open('generate_paper_graphs.py', 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print("File updated successfully")
        else:
            print("Line 524 doesn't contain the expected pattern")
    else:
        print(f"Line 524 not found - file only has {len(lines)} lines")

if __name__ == "__main__":
    fix_line_524()
