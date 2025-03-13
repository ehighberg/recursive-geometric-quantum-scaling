#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Check file for merge conflicts and other issues
import sys

def check_file(filename):
    print(f"Checking {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check for merge conflict markers
    for i, line in enumerate(lines, 1):
        if line.startswith('<<<<<<<') or line.startswith('=======') or line.startswith('>>>>>>>'):
            print(f"Line {i}: Merge conflict marker found: {line.strip()}")
    
    # Print the last 5 lines
    print("\nLast 5 lines:")
    for i, line in enumerate(lines[-5:], len(lines) - 4):
        print(f"Line {i}: {line.strip()}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_file(sys.argv[1])
    else:
        check_file("analyses/scaling/analyze_fractal_topology_relation.py")
