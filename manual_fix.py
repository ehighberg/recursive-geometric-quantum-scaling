#!/usr/bin/env python
"""
Manual fix for the specific syntax error in generate_paper_graphs.py line 237.
"""

from pathlib import Path
import re

def fix_line_237():
    file_path = Path('generate_paper_graphs.py')
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Target specifically line 237
    old_line = lines[236]  # 0-indexed
    print(f"Old line 237: {old_line.strip()}")
    
    # Fix 1: Replace any sequence of multiple commas with a single comma
    new_line = re.sub(r',{2,}', ',', old_line)
    
    # Fix 2: Add comma after filename if missing
    pattern = r'\.png"\s+dpi='
    if re.search(pattern, new_line):
        new_line = re.sub(pattern, '.png", dpi=', new_line)
    
    # Fix 3: Add comma between dpi and bbox_inches if missing
    pattern = r'dpi=300\s+bbox_inches'
    if re.search(pattern, new_line):
        new_line = re.sub(pattern, 'dpi=300, bbox_inches', new_line)
    
    # Update line
    lines[236] = new_line
    print(f"New line 237: {new_line.strip()}")
    
    # Apply same fixes to all similar savefig lines
    for i, line in enumerate(lines):
        if "plt.savefig" in line and ".png" in line:
            # Skip already fixed line
            if i == 236:
                continue
                
            old_line = lines[i]
            
            # Apply all fixes
            new_line = re.sub(r',{2,}', ',', old_line)
            
            pattern = r'\.png"\s+dpi='
            if re.search(pattern, new_line):
                new_line = re.sub(pattern, '.png", dpi=', new_line)
            
            pattern = r'dpi=300\s+bbox_inches'
            if re.search(pattern, new_line):
                new_line = re.sub(pattern, 'dpi=300, bbox_inches', new_line)
                
            if new_line != old_line:
                print(f"Fixed line {i+1}: {old_line.strip()} -> {new_line.strip()}")
                lines[i] = new_line
    
    # Write changes back to file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print("File updated successfully")

if __name__ == "__main__":
    fix_line_237()
