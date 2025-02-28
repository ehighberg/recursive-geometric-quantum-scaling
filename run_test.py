#!/usr/bin/env python
# run_test.py

"""
Script to run a specific test for the phi-driven quantum framework.

This script provides a simple interface to run individual tests from the
validation framework, making it easier to test specific components.
"""

import argparse
import numpy as np
from pathlib import Path

from tests.test_phi_coherence import run_coherence_comparison
from tests.test_topological_protection import test_anyon_topological_protection
from analyses.scaling.analyze_phi_significance import analyze_phi_significance
from run_phi_resonant_analysis import run_phi_analysis

def main():
    """Run a specific test for the phi-driven quantum framework."""
    parser = argparse.ArgumentParser(description="Run a specific test for the phi-driven quantum framework.")
    parser.add_argument(
        "test",
        choices=["coherence", "topological", "resonance", "phi_analysis", "all"],
        help="Test to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick version of the test with fewer parameters"
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run the specified test
    if args.test == "coherence" or args.test == "all":
        print("\n=== Testing Coherence Enhancement ===")
        coherence_dir = output_dir / "coherence"
        coherence_dir.mkdir(exist_ok=True, parents=True)
        
        if args.quick:
            # Quick version with fewer parameters
            results = run_coherence_comparison(
                qubit_counts=[1],
                n_steps=20,
                noise_levels=[0.05],
                output_dir=coherence_dir
            )
        else:
            # Full version
            results = run_coherence_comparison(
                qubit_counts=[1, 2],
                n_steps=50,
                noise_levels=[0.01, 0.05, 0.1],
                output_dir=coherence_dir
            )
        
        # Print summary
        if 'statistics' in results:
            stats = results['statistics']
            print("\nCoherence Enhancement Results:")
            print(f"Mean Improvement Factor: {stats['mean_improvement']:.2f} ± {stats['std_improvement']:.2f}")
            # Handle NaN values for t-statistic and p-value
            if np.isnan(stats['t_statistic']) or np.isnan(stats['p_value']):
                print(f"t-statistic: {stats['t_statistic']}, p-value: {stats['p_value']}")
                print("Note: Statistical tests not applicable with limited data points.")
            else:
                print(f"t-statistic: {stats['t_statistic']:.2f}, p-value: {stats['p_value']:.4f}")
            
            if stats['significant_improvement']:
                print("RESULT: Phi-scaling shows statistically significant coherence improvement.")
            else:
                print("RESULT: No statistically significant coherence improvement with phi-scaling.")
    
    if args.test == "topological" or args.test == "all":
        print("\n=== Testing Topological Protection ===")
        topological_dir = output_dir / "topological"
        topological_dir.mkdir(exist_ok=True, parents=True)
        
        if args.quick:
            # Quick version with fewer parameters
            results = test_anyon_topological_protection(
                braid_sequences=["1,2,1"],
                noise_levels=[0.05],
                output_dir=topological_dir
            )
        else:
            # Full version
            results = test_anyon_topological_protection(
                braid_sequences=["1,2,1", "1,2,1,2,1"],
                noise_levels=[0.01, 0.05],
                output_dir=topological_dir
            )
        
        # Print summary
        if 'statistics' in results:
            stats = results['statistics']
            print("\nTopological Protection Results:")
            print(f"Mean Protection Factor: {stats['mean_protection']:.2f} ± {stats['std_protection']:.2f}")
            print(f"t-statistic: {stats['t_statistic']:.2f}, p-value: {stats['p_value']:.4f}")
            
            if stats['significant_protection']:
                print("RESULT: Fibonacci anyons show statistically significant topological protection.")
            else:
                print("RESULT: No statistically significant topological protection with Fibonacci anyons.")
    
    if args.test == "resonance" or args.test == "all":
        print("\n=== Testing Phi Resonance in Quantum Properties ===")
        resonance_dir = output_dir / "resonance"
        resonance_dir.mkdir(exist_ok=True, parents=True)
        
        results = analyze_phi_significance(
            fine_resolution=not args.quick,
            save_results=True
        )
        
        # Move results to output directory
        for file in Path(".").glob("phi_significance_*.png"):
            file.rename(resonance_dir / file.name)
        
        if Path("phi_significance_results.csv").exists():
            Path("phi_significance_results.csv").rename(resonance_dir / "phi_significance_results.csv")
        
        # Print summary
        phi_idx = np.argmin(np.abs(results['fs_values'] - 1.618))
        print("\nPhi Resonance Results:")
        print(f"Band Gap at Phi: {results['band_gaps'][phi_idx]:.4f}")
        print(f"Fractal Dimension at Phi: {results['fractal_dimensions'][phi_idx]:.4f}")
        print(f"Topological Invariant at Phi: {results['topological_invariants'][phi_idx]:.4f}")
    
    if args.test == "phi_analysis" or args.test == "all":
        print("\n=== Running Phi-Resonant Analysis ===")
        phi_dir = output_dir / "phi_analysis"
        phi_dir.mkdir(exist_ok=True, parents=True)
        
        if args.quick:
            # Quick version with fewer parameters
            results = run_phi_analysis(
                output_dir=phi_dir,
                num_qubits=1,
                n_steps=20
            )
        else:
            # Full version
            results = run_phi_analysis(
                output_dir=phi_dir,
                num_qubits=1,
                n_steps=50
            )
        
        # Print summary
        phi_idx = np.argmin(np.abs(results['scaling_factors'] - 1.618))
        if phi_idx < len(results['scaling_factors']):
            phi_factor = results['scaling_factors'][phi_idx]
            print("\nPhi-Resonant Analysis Results:")
            print(f"State Overlap at Phi: {results['comparative_metrics'][phi_factor]['state_overlap']:.4f}")
            print(f"Dimension Difference at Phi: {results['comparative_metrics'][phi_factor]['dimension_difference']:.4f}")

if __name__ == "__main__":
    main()
