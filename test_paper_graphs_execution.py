#!/usr/bin/env python
"""
Test to diagnose issues in generate_paper_graphs.py
"""

import sys
import traceback
import importlib.util
from pathlib import Path

def test_module_functions():
    """Import generate_paper_graphs and test each function separately"""
    # Import the generate_paper_graphs module
    spec = importlib.util.spec_from_file_location(
        "generate_paper_graphs", 
        Path("generate_paper_graphs.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Create output directory
    output_dir = module.create_output_directory(Path("test_output"))
    print(f"Created test output directory: {output_dir}")
    
    # Test each function
    functions_to_test = [
        "generate_fractal_energy_spectrum",
        "generate_wavefunction_profile",
        "generate_fractal_dimension_vs_recursion",
        "generate_topological_invariants_graph",
        "generate_robustness_under_perturbations",
        "generate_scale_factor_dependence",
        "generate_wavepacket_evolution",
        "generate_entanglement_entropy"
    ]
    
    # Dictionary to hold results
    results = {}
    
    for func_name in functions_to_test:
        print(f"\n{'='*50}")
        print(f"Testing function: {func_name}")
        print(f"{'='*50}")
        
        try:
            # Get the function object
            func = getattr(module, func_name)
            
            # Execute the function
            func(output_dir)
            
            # If we get here, it worked
            print(f"SUCCESS: {func_name} executed without errors")
            results[func_name] = "SUCCESS"
        except Exception as e:
            # If we get here, it failed
            print(f"ERROR: {func_name} failed with exception:")
            traceback.print_exc()
            results[func_name] = f"ERROR: {str(e)}"
    
    # Print summary of all results
    print("\n\n")
    print(f"{'='*50}")
    print(f"SUMMARY OF RESULTS")
    print(f"{'='*50}")
    for func_name, result in results.items():
        status = "✅" if result == "SUCCESS" else "❌"
        print(f"{status} {func_name}")
    
    # Find all failures
    failures = [func_name for func_name, result in results.items() if result != "SUCCESS"]
    print(f"\nTotal functions: {len(functions_to_test)}")
    print(f"Successful: {len(functions_to_test) - len(failures)}")
    print(f"Failed: {len(failures)}")
    if failures:
        print(f"Failed functions: {', '.join(failures)}")

if __name__ == "__main__":
    test_module_functions()
