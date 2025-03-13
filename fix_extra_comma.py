

#!/usr/bin/env python
"""
Fix the extra comma in line 524 of generate_paper_graphs.py
"""

def fix_extra_comma():
    with open('generate_paper_graphs.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix line 524 with the extra comma
    line_524 = lines[523]  # 0-indexed
    print(f"Original line: {line_524.strip()}")
    
    # Replace double comma with single comma
    fixed_line = line_524.replace('png",,', 'png",')
    
    # Update the line
    lines[523] = fixed_line
    print(f"Fixed line: {fixed_line.strip()}")
    
    # Write back to file
    with open('generate_paper_graphs.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("File updated successfully")

if __name__ == "__main__":
    fix_extra_comma()
