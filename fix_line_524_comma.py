#!/usr/bin/env python
"""
Fix the extra comma in line 524 of generate_paper_graphs.py
"""

def fix_comma_error():
    # Read the file
    with open('generate_paper_graphs.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix line 524 (index 523)
    if len(lines) > 523:
        line = lines[523]
        if ',,' in line:
            fixed_line = line.replace(',,', ',')
            lines[523] = fixed_line
            print(f"Fixed line 524 from:\n{line}")
            print(f"To:\n{fixed_line}")
        else:
            print("No double comma found in line 524")
    else:
        print(f"File has only {len(lines)} lines, cannot access line 524")
    
    # Write the fixed content back
    with open('generate_paper_graphs.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("File updated successfully")

if __name__ == "__main__":
    fix_comma_error()
