#!/usr/bin/env python
# validate_phi_framework.py

"""
Comprehensive validation script for the phi-driven quantum framework.

This script runs all test components and generates a validation report to determine
if the paper's claims are supported by the simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
import time
import json
from datetime import datetime

# Import test modules
from tests.test_phi_coherence import run_coherence_comparison
from tests.test_topological_protection import test_anyon_topological_protection
from analyses.scaling.analyze_phi_significance import analyze_phi_significance
from run_phi_resonant_analysis import run_phi_analysis

def validate_phi_framework(tests_to_run=None, output_dir=None):
    """
    Comprehensive validation of the phi-driven quantum framework.
    
    Parameters:
    -----------
    tests_to_run : list, optional
        List of tests to run ('coherence', 'topological', 'resonance', 'phi_analysis')
    output_dir : str or Path, optional
        Directory to save results
        
    Returns:
    --------
    dict
        Dictionary containing validation results
    """
    if tests_to_run is None:
        tests_to_run = ['coherence', 'resonance', 'phi_analysis']  # Skip 'topological' for now
    
    if output_dir is None:
        output_dir = Path("validation_results")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize results storage
    results = {}
    
    # Start time
    start_time = time.time()
    
    # 1. Test phi resonance in quantum properties
    if 'resonance' in tests_to_run:
        print("\n=== Testing Phi Resonance in Quantum Properties ===")
        resonance_dir = output_dir / "resonance"
        resonance_dir.mkdir(exist_ok=True, parents=True)
        
        results['resonance'] = analyze_phi_significance(
            fine_resolution=True,
            save_results=True
        )
        
        # Move results to validation directory
        for file in Path(".").glob("phi_significance_*.png"):
            target_file = resonance_dir / file.name
            if target_file.exists():
                target_file.unlink()  # Remove existing file
            file.rename(target_file)
        
        if Path("phi_significance_results.csv").exists():
            target_file = resonance_dir / "phi_significance_results.csv"
            if target_file.exists():
                target_file.unlink()  # Remove existing file
            Path("phi_significance_results.csv").rename(target_file)
    
    # 2. Test coherence enhancement
    if 'coherence' in tests_to_run:
        print("\n=== Testing Coherence Enhancement ===")
        coherence_dir = output_dir / "coherence"
        coherence_dir.mkdir(exist_ok=True, parents=True)
        
        results['coherence'] = run_coherence_comparison(
            qubit_counts=[1, 2],
            n_steps=50,
            noise_levels=[0.01, 0.05, 0.1],
            output_dir=coherence_dir
        )
    
    # 3. Test topological protection
    if 'topological' in tests_to_run:
        print("\n=== Testing Topological Protection ===")
        topological_dir = output_dir / "topological"
        topological_dir.mkdir(exist_ok=True, parents=True)
        
        results['topological'] = test_anyon_topological_protection(
            braid_sequences=["1,2,1", "1,2,1,2,1"],
            noise_levels=[0.01, 0.05],
            output_dir=topological_dir
        )
    
    # 4. Run phi-resonant analysis
    if 'phi_analysis' in tests_to_run:
        print("\n=== Running Phi-Resonant Analysis ===")
        phi_dir = output_dir / "phi_analysis"
        phi_dir.mkdir(exist_ok=True, parents=True)
        
        results['phi_analysis'] = run_phi_analysis(
            output_dir=phi_dir,
            num_qubits=1,
            n_steps=50
        )
    
    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Generate validation report
    report = generate_validation_report(results, elapsed_time)
    
    # Save report
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Create HTML report
    create_html_report(report, output_dir / "validation_report.html")
    
    # Check if claims are supported by evidence
    claims_validated = validate_paper_claims(results)
    
    # Print summary
    print("\n=== Validation Summary ===")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print("\nPaper Claims Validation:")
    for claim, validated in claims_validated.items():
        status = "VALIDATED" if validated else "NOT VALIDATED"
        print(f"  - {claim}: {status}")
    
    return {
        'results': results,
        'report': report,
        'claims_validated': claims_validated,
        'elapsed_time': elapsed_time
    }

def generate_validation_report(results, elapsed_time):
    """
    Generate a validation report from test results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing test results
    elapsed_time : float
        Elapsed time in seconds
        
    Returns:
    --------
    dict
        Dictionary containing validation report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_time': elapsed_time,
        'tests_run': list(results.keys()),
        'test_results': {}
    }
    
    # Process resonance results
    if 'resonance' in results:
        resonance = results['resonance']
        report['test_results']['resonance'] = {
            'phi_index': np.argmin(np.abs(resonance['fs_values'] - 1.618)),
            'band_gaps': resonance['band_gaps'].tolist() if isinstance(resonance['band_gaps'], np.ndarray) else resonance['band_gaps'],
            'fractal_dimensions': resonance['fractal_dimensions'].tolist() if isinstance(resonance['fractal_dimensions'], np.ndarray) else resonance['fractal_dimensions'],
            'topological_invariants': resonance['topological_invariants'].tolist() if isinstance(resonance['topological_invariants'], np.ndarray) else resonance['topological_invariants']
        }
    
    # Process coherence results
    if 'coherence' in results:
        coherence = results['coherence']
        if 'statistics' in coherence:
            report['test_results']['coherence'] = {
                'mean_improvement': float(coherence['statistics']['mean_improvement']),
                'std_improvement': float(coherence['statistics']['std_improvement']),
                't_statistic': float(coherence['statistics']['t_statistic']),
                'p_value': float(coherence['statistics']['p_value']),
                'significant_improvement': bool(coherence['statistics']['significant_improvement'])
            }
    
    # Process topological results
    if 'topological' in results:
        topological = results['topological']
        if 'statistics' in topological:
            report['test_results']['topological'] = {
                'mean_protection': float(topological['statistics']['mean_protection']),
                'std_protection': float(topological['statistics']['std_protection']),
                't_statistic': float(topological['statistics']['t_statistic']),
                'p_value': float(topological['statistics']['p_value']),
                'significant_protection': bool(topological['statistics']['significant_protection'])
            }
    
    # Process phi analysis results
    if 'phi_analysis' in results:
        phi_analysis = results['phi_analysis']
        # Extract key metrics from phi analysis
        phi_idx = np.argmin(np.abs(phi_analysis['scaling_factors'] - 1.618))
        if phi_idx < len(phi_analysis['scaling_factors']):
            phi_factor = phi_analysis['scaling_factors'][phi_idx]
            report['test_results']['phi_analysis'] = {
                'phi_factor': float(phi_factor),
                'state_overlap': float(phi_analysis['comparative_metrics'][phi_factor]['state_overlap']),
                'dimension_difference': float(phi_analysis['comparative_metrics'][phi_factor]['dimension_difference'])
            }
    
    return report

def validate_paper_claims(results):
    """
    Validate claims from the paper based on simulation results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing test results
        
    Returns:
    --------
    dict
        Dictionary containing validation results for each claim
    """
    validations = {
        'phi_resonance': False,
        'enhanced_coherence': False,
        'topological_protection': False
    }
    
    # 1. Phi resonance claim
    if 'resonance' in results:
        resonance = results['resonance']
        phi_idx = np.argmin(np.abs(resonance['fs_values'] - 1.618))
        
        # Check if metrics at phi are significantly different from elsewhere
        if phi_idx < len(resonance['fs_values']):
            # Calculate z-scores for metrics at phi
            metrics = ['band_gaps', 'fractal_dimensions', 'topological_invariants']
            z_scores = {}
            
            for metric in metrics:
                if metric in resonance and len(resonance[metric]) > phi_idx:
                    # Get value at phi
                    phi_value = resonance[metric][phi_idx]
                    
                    # Get values away from phi
                    far_indices = np.where(np.abs(resonance['fs_values'] - 1.618) > 0.3)[0]
                    far_values = [resonance[metric][i] for i in far_indices if i < len(resonance[metric])]
                    
                    if far_values:
                        # Calculate z-score
                        mean = np.nanmean(far_values)
                        std = np.nanstd(far_values)
                        if std > 0:
                            z_scores[metric] = (phi_value - mean) / std
            
            # Claim is validated if any metric has |z-score| > 2
            validations['phi_resonance'] = any(
                abs(z) > 2.0 for z in z_scores.values() if not np.isnan(z)
            )
    
    # 2. Enhanced coherence claim
    if 'coherence' in results:
        coherence = results['coherence']
        if 'statistics' in coherence:
            stats = coherence['statistics']
            validations['enhanced_coherence'] = (
                stats.get('significant_improvement', False) and
                stats.get('mean_improvement', 1.0) > 1.15  # At least 15% improvement
            )
    
    # 3. Topological protection claim
    if 'topological' in results:
        topological = results['topological']
        if 'statistics' in topological:
            stats = topological['statistics']
            validations['topological_protection'] = (
                stats.get('significant_protection', False) and
                stats.get('mean_protection', 1.0) > 1.2  # At least 20% improvement
            )
    
    return validations

def create_html_report(report, output_path):
    """
    Create an HTML report from validation results.
    
    Parameters:
    -----------
    report : dict
        Dictionary containing validation report
    output_path : Path
        Path to save the HTML report
    """
    # Create HTML content
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phi-Driven Framework Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .section {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
            .validated {{ color: green; font-weight: bold; }}
            .not-validated {{ color: red; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Phi-Driven Framework Validation Report</h1>
        <p><strong>Date:</strong> {report['timestamp']}</p>
        <p><strong>Elapsed Time:</strong> {report['elapsed_time']:.2f} seconds</p>
        <p><strong>Tests Run:</strong> {', '.join(report['tests_run'])}</p>
        
        <h2>Test Results</h2>
    """
    
    # Add resonance results
    if 'resonance' in report['test_results']:
        resonance = report['test_results']['resonance']
        html += f"""
        <div class="section">
            <h3>Phi Resonance</h3>
            <p>This test evaluates if quantum properties show special behavior at or near the golden ratio (phi ≈ 1.618).</p>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value at Phi</th>
                </tr>
                <tr>
                    <td>Band Gap</td>
                    <td>{resonance['band_gaps'][resonance['phi_index']] if isinstance(resonance['band_gaps'], list) and resonance['phi_index'] < len(resonance['band_gaps']) else 'N/A'}</td>
                </tr>
                <tr>
                    <td>Fractal Dimension</td>
                    <td>{resonance['fractal_dimensions'][resonance['phi_index']] if isinstance(resonance['fractal_dimensions'], list) and resonance['phi_index'] < len(resonance['fractal_dimensions']) else 'N/A'}</td>
                </tr>
                <tr>
                    <td>Topological Invariant</td>
                    <td>{resonance['topological_invariants'][resonance['phi_index']] if isinstance(resonance['topological_invariants'], list) and resonance['phi_index'] < len(resonance['topological_invariants']) else 'N/A'}</td>
                </tr>
            </table>
        </div>
        """
    
    # Add coherence results
    if 'coherence' in report['test_results']:
        coherence = report['test_results']['coherence']
        html += f"""
        <div class="section">
            <h3>Coherence Enhancement</h3>
            <p>This test evaluates if phi-scaled pulse sequences enhance quantum coherence compared to uniform sequences.</p>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Mean Improvement Factor</td>
                    <td>{coherence['mean_improvement']:.2f} ± {coherence['std_improvement']:.2f}</td>
                </tr>
                <tr>
                    <td>t-statistic</td>
                    <td>{coherence['t_statistic'] if np.isnan(coherence['t_statistic']) else f"{coherence['t_statistic']:.2f}"}</td>
                </tr>
                <tr>
                    <td>p-value</td>
                    <td>{coherence['p_value'] if np.isnan(coherence['p_value']) else f"{coherence['p_value']:.4f}"}</td>
                </tr>
                {"""<tr>
                    <td colspan="2" style="color: #666; font-style: italic;">Note: Statistical tests not applicable with limited data points.</td>
                </tr>""" if np.isnan(coherence['t_statistic']) or np.isnan(coherence['p_value']) else ""}
                <tr>
                    <td>Significant Improvement</td>
                    <td class="{'validated' if coherence['significant_improvement'] else 'not-validated'}">{coherence['significant_improvement']}</td>
                </tr>
            </table>
        </div>
        """
    
    # Add topological results
    if 'topological' in report['test_results']:
        topological = report['test_results']['topological']
        html += f"""
        <div class="section">
            <h3>Topological Protection</h3>
            <p>This test evaluates if Fibonacci anyon braiding provides topological protection against local errors.</p>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Mean Protection Factor</td>
                    <td>{topological['mean_protection']:.2f} ± {topological['std_protection']:.2f}</td>
                </tr>
                <tr>
                    <td>t-statistic</td>
                    <td>{topological['t_statistic'] if np.isnan(topological['t_statistic']) else f"{topological['t_statistic']:.2f}"}</td>
                </tr>
                <tr>
                    <td>p-value</td>
                    <td>{topological['p_value'] if np.isnan(topological['p_value']) else f"{topological['p_value']:.4f}"}</td>
                </tr>
                {"""<tr>
                    <td colspan="2" style="color: #666; font-style: italic;">Note: Statistical tests not applicable with limited data points.</td>
                </tr>""" if np.isnan(topological['t_statistic']) or np.isnan(topological['p_value']) else ""}
                <tr>
                    <td>Significant Protection</td>
                    <td class="{'validated' if topological['significant_protection'] else 'not-validated'}">{topological['significant_protection']}</td>
                </tr>
            </table>
        </div>
        """
    
    # Add phi analysis results
    if 'phi_analysis' in report['test_results']:
        phi_analysis = report['test_results']['phi_analysis']
        html += f"""
        <div class="section">
            <h3>Phi-Resonant Analysis</h3>
            <p>This test compares standard and phi-recursive quantum evolution.</p>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>State Overlap at Phi</td>
                    <td>{phi_analysis['state_overlap']:.4f}</td>
                </tr>
                <tr>
                    <td>Dimension Difference at Phi</td>
                    <td>{phi_analysis['dimension_difference']:.4f}</td>
                </tr>
            </table>
        </div>
        """
    
    # Add paper claims validation
    claims_validated = validate_paper_claims(report['test_results'])
    html += f"""
        <h2>Paper Claims Validation</h2>
        <div class="section">
            <table>
                <tr>
                    <th>Claim</th>
                    <th>Validated</th>
                </tr>
                <tr>
                    <td>Phi Resonance: Quantum properties show special behavior at or near the golden ratio (phi ≈ 1.618)</td>
                    <td class="{'validated' if claims_validated['phi_resonance'] else 'not-validated'}">{claims_validated['phi_resonance']}</td>
                </tr>
                <tr>
                    <td>Enhanced Coherence: Phi-scaled pulse sequences enhance quantum coherence compared to uniform sequences</td>
                    <td class="{'validated' if claims_validated['enhanced_coherence'] else 'not-validated'}">{claims_validated['enhanced_coherence']}</td>
                </tr>
                <tr>
                    <td>Topological Protection: Fibonacci anyon braiding provides topological protection against local errors</td>
                    <td class="{'validated' if claims_validated['topological_protection'] else 'not-validated'}">{claims_validated['topological_protection']}</td>
                </tr>
            </table>
        </div>
    """
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, "w") as f:
        f.write(html)
    
    print(f"HTML report saved to {output_path}")

if __name__ == "__main__":
    # Run validation with default parameters
    validate_phi_framework(
        tests_to_run=['coherence', 'resonance', 'phi_analysis'],  # Skip 'topological' for now
        output_dir="validation_results"
    )
