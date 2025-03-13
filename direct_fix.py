!/usr/bin/env python
"""
Direct fix for the syntax error in the generate_paper_graphs.py file.
"""

def direct_fix():
    # Read the file line by line
    with open('generate_paper_graphs.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Loop through lines looking for the problematic savefig calls
    for i, line in enumerate(lines):
        # Look for plt.savefig with missing commas
        if 'plt.savefig(' in line and '.png"' in line and ' dpi=' in line:
            print(f"Line {i+1}: {line.strip()}")
            
            # Direct replacement of the "png" dpi with "png", dpi
            lines[i] = line.replace('.png"', '.png",')
            
            # Direct replacement of dpi=300 bbox_inches with dpi=300, bbox_inches
            lines[i] = lines[i].replace('dpi=300 ', 'dpi=300, ')
            
            print(f"Fixed: {lines[i].strip()}")
    
    # Write back modified content
    with open('generate_paper_graphs.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Direct fix completed")

if __name__ == "__main__":
    direct_fix()
