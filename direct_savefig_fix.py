#!/usr/bin/env python
"""
Direct fix for the savefig syntax errors in generate_paper_graphs.py
"""

def fix_line_524():
    with open('generate_paper_graphs.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the problematic line
    problematic_pattern = 'plt.savefig(output_dir / "fractal_dim_vs_recursion.png" dpi=300 bbox_inches=\'tight\')'
    
    # Create the fixed line with proper commas
    fixed_line = 'plt.savefig(output_dir / "fractal_dim_vs_recursion.png", dpi=300, bbox_inches=\'tight\')'
    
    # Replace the line
    if problematic_pattern in content:
        print(f"Found problematic line: {problematic_pattern}")
        content = content.replace(problematic_pattern, fixed_line)
        print(f"Replaced with: {fixed_line}")
        
        # Write the fixed content back
        with open('generate_paper_graphs.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("File has been updated")
    else:
        print("Problematic pattern not found in file")

if __name__ == "__main__":
    fix_line_524()
