!/usr/bin/env python
"""
Direct line-by-line fix for syntax errors in the generate_paper_graphs.py file.
"""

def fix_specific_lines():
    # Read the file line by line
    with open('generate_paper_graphs.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fixing issues line-by-line with specific solutions
    changes_made = False
    
    # Fix pattern 1: Extra commas such as ",,,," in savefig lines
    for i, line in enumerate(lines):
        if 'plt.savefig' in line and ',,,' in line:
            print(f"Line {i+1}: Found extra commas: {line.strip()}")
            # Replace multiple commas with a single comma
            new_line = line.replace(',,,,', ',')
            lines[i] = new_line
            print(f"Fixed to: {new_line.strip()}")
            changes_made = True
    
    # Fix pattern 2: Missing commas between "png" dpi=300 bbox_inches
    for i, line in enumerate(lines):
        if 'plt.savefig' in line and '.png"' in line and ' dpi=' in line:
            print(f"Line {i+1}: Missing comma after filename: {line.strip()}")
            # Add comma after the filename - before dpi
            new_line = line.replace('.png"', '.png",')
            # Also fix missing comma between dpi and bbox_inches if needed
            if ' bbox_inches=' in new_line:
                new_line = new_line.replace('dpi=300 ', 'dpi=300, ')
            lines[i] = new_line
            print(f"Fixed to: {new_line.strip()}")
            changes_made = True

    # Write back modified content
    if changes_made:
        with open('generate_paper_graphs.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("File updated successfully")
    else:
        print("No issues found")

if __name__ == "__main__":
    fix_specific_lines()
