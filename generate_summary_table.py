#!/usr/bin/env python
"""
Generate a comprehensive summary table for the paper.
This script compiles results from all analyses into a publication-ready format.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import PHI

def generate_summary_table():
    """
    Generate a comprehensive summary table for the paper.
    """
    print("Generating summary table for the paper...")
    
    # Load results from phi significance analysis
    try:
        phi_results = pd.read_csv("phi_significance_results.csv")
        print("Loaded phi significance results.")
    except FileNotFoundError:
        phi_results = None
        print("Warning: phi_significance_results.csv not found.")
    
    # Load results from fractal-topology relation analysis
    try:
        fractal_topo_results = pd.read_csv("fractal_topology_relation.csv")
        print("Loaded fractal-topology relation results.")
    except FileNotFoundError:
        fractal_topo_results = None
        print("Warning: fractal_topology_relation.csv not found.")
    
    # Load results from f_s scaling analysis
    try:
        fs_results = pd.read_csv("fs_scaling_results.csv")
        print("Loaded f_s scaling results.")
    except FileNotFoundError:
        fs_results = None
        print("Warning: fs_scaling_results.csv not found.")
    
    # Create a comprehensive summary table
    summary_data = []
    
    # Special scaling factors to highlight
    special_fs = [0.5, 1.0, PHI, 2.0, 3.0]
    special_fs_labels = ["0.5", "1.0", f"φ ({PHI:.4f})", "2.0", "3.0"]
    
    # Extract data for special scaling factors
    for fs, label in zip(special_fs, special_fs_labels):
        row = {"Scaling Factor": label}
        
        # Extract data from phi significance results
        if phi_results is not None:
            # Find the closest f_s value
            closest_idx = np.argmin(np.abs(phi_results["f_s"].values - fs))
            closest_fs = phi_results["f_s"].iloc[closest_idx]
            
            if np.isclose(closest_fs, fs, rtol=1e-2):
                row["Band Gap"] = phi_results["Band Gap"].iloc[closest_idx]
                row["Fractal Dimension"] = phi_results["Fractal Dimension"].iloc[closest_idx]
                row["Correlation Length"] = phi_results["Correlation Length"].iloc[closest_idx]
        
        # Extract data from fractal-topology relation results
        if fractal_topo_results is not None:
            # Find the closest f_s value
            closest_idx = np.argmin(np.abs(fractal_topo_results["f_s"].values - fs))
            closest_fs = fractal_topo_results["f_s"].iloc[closest_idx]
            
            if np.isclose(closest_fs, fs, rtol=1e-2):
                row["Topological Invariant"] = fractal_topo_results["Topological Invariant"].iloc[closest_idx]
                row["Z2 Index"] = fractal_topo_results["Z2 Index"].iloc[closest_idx]
                row["Self-Similarity"] = fractal_topo_results["Self-Similarity"].iloc[closest_idx]
        
        summary_data.append(row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create plots directory if it doesn't exist
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Save summary table to CSV
    summary_df.to_csv(data_dir / "summary_table.csv", index=False)
    print("Summary table saved to data/summary_table.csv")
    
    # Create a publication-ready table visualization
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    ax.axis('off')
    ax.axis('tight')
    
    # Create table
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2'] * len(summary_df.columns)
    )
    
    # Highlight phi row
    phi_row_idx = special_fs_labels.index(f"φ ({PHI:.4f})") + 1  # +1 for header
    for j in range(len(summary_df.columns)):
        cell = table[(phi_row_idx, j)]
        cell.set_facecolor('#e6f2ff')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add title
    plt.title("Quantum Properties at Different Scaling Factors", fontsize=14, pad=20)
    
    # Save table as image
    plt.tight_layout()
    plt.savefig(plots_dir / "summary_table.png", dpi=300, bbox_inches='tight')
    print("Summary table visualization saved to plots/summary_table.png")
    
    # Generate LaTeX table for the paper
    latex_table = summary_df.to_latex(index=False, float_format="%.4f")
    
    with open(data_dir / "summary_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
    print("LaTeX table saved to data/summary_table.tex")
    
    # Generate a more comprehensive comparison table
    if phi_results is not None and fs_results is not None:
        # Find phi row in results
        phi_idx = np.argmin(np.abs(phi_results["f_s"].values - PHI))
        
        # Create comparison data
        comparison_data = []
        
        # Properties to compare
        properties = ["Band Gap", "Fractal Dimension", "Topological Invariant", "Correlation Length"]
        
        for prop in properties:
            if prop in phi_results.columns:
                phi_value = phi_results[prop].iloc[phi_idx]
                
                # Calculate statistics
                mean_value = phi_results[prop].mean()
                min_value = phi_results[prop].min()
                max_value = phi_results[prop].max()
                
                # Calculate relative difference
                if mean_value != 0:
                    rel_diff = (phi_value - mean_value) / mean_value * 100
                else:
                    rel_diff = 0
                
                comparison_data.append({
                    "Property": prop,
                    f"Value at φ={PHI:.4f}": phi_value,
                    "Mean Value": mean_value,
                    "Min Value": min_value,
                    "Max Value": max_value,
                    "% Difference from Mean": rel_diff
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table to CSV
        comparison_df.to_csv(data_dir / "phi_comparison_table.csv", index=False)
        print("Phi comparison table saved to data/phi_comparison_table.csv")
        
        # Create a publication-ready comparison table visualization
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.axis('off')
        ax.axis('tight')
        
        # Create table
        table = ax.table(
            cellText=comparison_df.values,
            colLabels=comparison_df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f2f2f2'] * len(comparison_df.columns)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add title
        plt.title(f"Comparison of Quantum Properties at φ={PHI:.4f} vs. Other Scaling Factors", fontsize=14, pad=20)
        
        # Save table as image
        plt.tight_layout()
        plt.savefig(plots_dir / "phi_comparison_table.png", dpi=300, bbox_inches='tight')
        print("Phi comparison table visualization saved to plots/phi_comparison_table.png")
        
        # Generate LaTeX table for the paper
        latex_comparison = comparison_df.to_latex(index=False, float_format="%.4f")
        
        with open(data_dir / "phi_comparison_table.tex", "w", encoding="utf-8") as f:
            f.write(latex_comparison)
        print("LaTeX comparison table saved to data/phi_comparison_table.tex")

if __name__ == "__main__":
    generate_summary_table()
