!/usr/bin/env python
"""
Comprehensive script to fix ALL missing commas in plt.savefig() calls
in the generate_paper_graphs.py file.
"""

def fix_all_savefig_calls():
    # Read the entire file
    with open('generate_paper_graphs.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fixed version of the problematic lines
    fixed_lines = []
    changes_made = False
    
    for line in lines:
        original_line = line
        # Check if line contains plt.savefig and lacks commas between arguments
        if 'plt.savefig(' in line and ('dpi=' in line or 'bbox_inches=' in line):
            # Does it have missing commas?
            if '"' in line and ' dpi=' in line:
                print(f"Found potential issue: {line.strip()}")
                
                # Fix specific patterns - replace spaces with commas where needed
                # Replace ".png" dpi= with ".png", dpi=
                line = line.replace('.png"', '.png",')
                
                # Replace dpi=300 bbox_inches= with dpi=300, bbox_inches=
                if 'dpi=' in line and ' bbox_inches=' in line:
                    line = line.replace('dpi=300 ', 'dpi=300, ')
                
                print(f"Fixed to: {line.strip()}")
                
                if line != original_line:
                    changes_made = True
        
        fixed_lines.append(line)
    
    # Only write back if changes were made
    if changes_made:
        # Write the fixed content back
        with open('generate_paper_graphs.py', 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        print("Fixes applied to plt.savefig() calls")
    else:
        print("No changes were made")

if __name__ == "__main__":
    fix_all_savefig_calls()
