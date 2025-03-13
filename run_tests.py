#!/usr/bin/env python
"""
Simple script to run the topological invariants tests and display results.
"""

import pytest
import sys
import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.join(script_dir, 'tests', 'test_topological_invariants_extended.py')

print(f"Running tests from: {test_path}")
print("=" * 80)

# Run the tests
result = pytest.main(['-v', test_path])

print("=" * 80)
print(f"Test run completed with exit code: {result}")
sys.exit(result)
