!/usr/bin/env python
"""
Script to fix the syntax error in the generate_paper_graphs.py file at line 524.
"""

def fix_syntax_error():
    # Read the generate_paper_graphs.py file
    with open('generate_paper_graphs.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and fix line 524 (savefig with missing commas)
    for i, line in enumerate(lines):
        if "plt.savefig(output_dir / \"fractal_dim_vs_recursion.png\" dpi=300 bbox_inches='tight')" in line:
            print(f"Found error on line {i+1}: {line.strip()}")
            lines[i] = line.replace(
                "plt.savefig(output_dir / \"fractal_dim_vs_recursion.png\" dpi=300 bbox_inches='tight')",
                "plt.savefig(output_dir / \"fractal_dim_vs_recursion.png\", dpi=300, bbox_inches='tight')"
            )
            print(f"Fixed to: {lines[i].strip()}")
    
    # Write the fixed content back to the file
    with open('generate_paper_graphs.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Fixed the syntax error in generate_paper_graphs.py")

if __name__ == "__main__":
    fix_syntax_error()
