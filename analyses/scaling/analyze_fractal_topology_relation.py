#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for analyzing the relationship between fractal dimensions and topological invariants.

This module implements analyses of how fractal dimensions relate to topological properties
in quantum systems, with a focus on identifying phase transitions and invariant structures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.stats as stats
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

def analyze_fractal_topology_relation(
    fractal_dimensions: Dict[float, np.ndarray],
    topological_invariants: Dict[float, Dict[str, np.ndarray]],
    scaling_factors: List[float],
    invariant_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    plot: bool = True,
    smoothing: float = 0.5
) -> Dict[str, Any]:
    """
    Analyze the relationship between fractal dimensions and topological invariants.
    
    Parameters:
        fractal_dimensions (Dict[float, np.ndarray]): Dictionary mapping scaling factors
                                                     to arrays of fractal dimensions
        topological_invariants (Dict[float, Dict[str, np.ndarray]]): Dictionary mapping scaling factors
                                                                    to dictionaries that map invariant names
                                                                    to arrays of invariant values
        scaling_factors (List[float]): List of scaling factors used
        invariant_names (Optional[List[str]]): List of invariant names to analyze
                                               (default: all available invariants)
        output_dir (Optional[str]): Directory to save results
        plot (bool): Whether to generate plots
        smoothing (float): Smoothing factor for phase diagram (0 for no smoothing)
    
    Returns:
        Dict[str, Any]: Dictionary containing analysis results
    """
    results = {}
    
    # If invariant names are not provided, use all available invariants
    if invariant_names is None:
        # Find first scaling factor with topological invariants
        for factor in scaling_factors:
            if factor in topological_invariants:
                invariant_names = list(topological_invariants[factor].keys())
                break
        
        if invariant_names is None:
            raise ValueError("No topological invariants found")
    
    # Compute correlations between fractal dimensions and topological invariants
    correlations = {}
    for inv_name in invariant_names:
        # Collect paired data for correlation analysis
        paired_data = []
        for factor in scaling_factors:
            if (factor in fractal_dimensions and 
                factor in topological_invariants and 
                inv_name in topological_invariants[factor]):
                
                fd = fractal_dimensions[factor]
                inv = topological_invariants[factor][inv_name]
                
                # Ensure arrays have same length (if not, use the minimum length)
                min_len = min(len(fd), len(inv))
                paired_data.append((factor, fd[:min_len], inv[:min_len]))
        
        if not paired_data:
            continue
            
        # Compute correlation for each scaling factor
        factor_correlations = {}
        overall_fd = []
        overall_inv = []
        
        for factor, fd, inv in paired_data:
            corr, p_value = stats.pearsonr(fd, inv)
            factor_correlations[factor] = {
                'correlation': corr,
                'p_value': p_value,
                'num_samples': len(fd)
            }
            
            # Append data for overall correlation
            overall_fd.extend(fd)
            overall_inv.extend(inv)
        
        # Compute overall correlation
        overall_corr, overall_p = stats.pearsonr(overall_fd, overall_inv)
        
        correlations[inv_name] = {
            'factor_correlations': factor_correlations,
            'overall_correlation': overall_corr,
            'overall_p_value': overall_p,
            'num_samples': len(overall_fd)
        }
    
    results['correlations'] = correlations
    
    # Create phase diagram data
    phase_diagrams = {}
    for inv_name in invariant_names:
        # Collect data for phase diagram
        fd_values = []
        factor_values = []
        inv_values = []
        
        for factor in scaling_factors:
            if (factor in fractal_dimensions and 
                factor in topological_invariants and 
                inv_name in topological_invariants[factor]):
                
                fd = fractal_dimensions[factor]
                inv = topological_invariants[factor][inv_name]
                
                # Ensure arrays have same length
                min_len = min(len(fd), len(inv))
                
                fd_values.extend(fd[:min_len])
                factor_values.extend([factor] * min_len)
                inv_values.extend(inv[:min_len])
        
        if not fd_values:
            continue
            
        # Create grid for interpolation
        grid_size = 100
        fd_grid = np.linspace(min(fd_values), max(fd_values), grid_size)
        factor_grid = np.linspace(min(factor_values), max(factor_values), grid_size)
        
        # Create meshgrid
        fd_mesh, factor_mesh = np.meshgrid(fd_grid, factor_grid)
        
        # Interpolate invariant values onto grid
        points = np.column_stack((fd_values, factor_values))
        inv_grid = griddata(points, inv_values, (fd_mesh, factor_mesh), method='linear')
        
        # Apply smoothing if requested
        if smoothing > 0:
            inv_grid = gaussian_filter(inv_grid, sigma=smoothing)
        
        phase_diagrams[inv_name] = {
            'fd_mesh': fd_mesh,
            'factor_mesh': factor_mesh,
            'inv_grid': inv_grid
        }
    
    results['phase_diagrams'] = phase_diagrams
    
    # Identify phase transitions and critical points
    phase_transitions = {}
    for inv_name, phase_diagram in phase_diagrams.items():
        inv_grid = phase_diagram['inv_grid']
        fd_grid = phase_diagram['fd_mesh'][0, :]
        factor_grid = phase_diagram['factor_mesh'][:, 0]
        
        # Calculate gradient magnitude of invariant field
        grad_y, grad_x = np.gradient(inv_grid)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Identify potential phase transitions (high gradient magnitude)
        threshold = np.percentile(gradient_magnitude[~np.isnan(gradient_magnitude)], 90)
        transitions = gradient_magnitude > threshold
        
        # Find critical points (local maxima of gradient magnitude)
        # Use a simple peak finding approach for demonstration
        critical_points = []
        for i in range(1, gradient_magnitude.shape[0]-1):
            for j in range(1, gradient_magnitude.shape[1]-1):
                if not np.isnan(gradient_magnitude[i, j]):
                    neighborhood = gradient_magnitude[i-1:i+2, j-1:j+2]
                    if np.nanmax(neighborhood) == gradient_magnitude[i, j]:
                        critical_points.append({
                            'fd': fd_grid[j],
                            'scaling_factor': factor_grid[i],
                            'invariant_value': inv_grid[i, j],
                            'gradient_magnitude': gradient_magnitude[i, j]
                        })
        
        # Sort critical points by gradient magnitude
        critical_points.sort(key=lambda p: p['gradient_magnitude'], reverse=True)
        
        # Take top 5 critical points or fewer if less exist
        top_critical_points = critical_points[:min(5, len(critical_points))]
        
        phase_transitions[inv_name] = {
            'gradient_magnitude': gradient_magnitude,
            'transition_mask': transitions,
            'critical_points': top_critical_points
        }
    
    results['phase_transitions'] = phase_transitions
    
    # Generate plots if requested
    if plot:
        if not output_dir:
            output_dir = 'plots'
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot correlation matrix
        if correlations:
            plt.figure(figsize=(10, 8))
            
            # Create correlation matrix
            inv_list = list(correlations.keys())
            corr_matrix = np.zeros((len(inv_list), len(scaling_factors)))
            
            for i, inv_name in enumerate(inv_list):
                if inv_name in correlations:
                    for j, factor in enumerate(scaling_factors):
                        if (factor in correlations[inv_name]['factor_correlations']):
                            corr_matrix[i, j] = correlations[inv_name]['factor_correlations'][factor]['correlation']
            
            # Plot heatmap
            plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(label='Correlation Coefficient')
            
            plt.xlabel('Scaling Factor')
            plt.ylabel('Topological Invariant')
            plt.title('Correlation Between Fractal Dimensions and Topological Invariants')
            
            # Set tick labels
            plt.yticks(np.arange(len(inv_list)), inv_list)
            plt.xticks(np.arange(len(scaling_factors)), [f"{f:.2f}" for f in scaling_factors], rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path / "fractal_topology_correlation.png", dpi=300)
            plt.close()
            
            # Plot overall correlations as bar chart
            plt.figure(figsize=(10, 6))
            
            inv_list = list(correlations.keys())
            overall_corrs = [correlations[inv]['overall_correlation'] for inv in inv_list]
            
            bars = plt.bar(inv_list, overall_corrs)
            
            # Add significance markers
            for i, inv in enumerate(inv_list):
                if correlations[inv]['overall_p_value'] < 0.001:
                    marker = '***'
                elif correlations[inv]['overall_p_value'] < 0.01:
                    marker = '**'
                elif correlations[inv]['overall_p_value'] < 0.05:
                    marker = '*'
                else:
                    marker = ''
                
                if marker:
                    plt.text(i, overall_corrs[i] + 0.05 * np.sign(overall_corrs[i]), 
                             marker, ha='center')
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Topological Invariant')
            plt.ylabel('Correlation with Fractal Dimension')
            plt.title('Overall Correlation Between Fractal Dimensions and Topological Invariants')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(-1.1, 1.1)
            plt.tight_layout()
            
            plt.savefig(output_path / "fractal_topology_overall_correlation.png", dpi=300)
            plt.close()
        
        # Plot phase diagrams
        for inv_name, phase_diagram in phase_diagrams.items():
            plt.figure(figsize=(10, 8))
            
            fd_mesh = phase_diagram['fd_mesh']
            factor_mesh = phase_diagram['factor_mesh']
            inv_grid = phase_diagram['inv_grid']
            
            # Plot phase diagram as a contour plot
            contour = plt.contourf(fd_mesh, factor_mesh, inv_grid, 20, cmap='viridis')
            plt.colorbar(contour, label=f'{inv_name} Value')
            
            # Add phase transitions if available
            if inv_name in phase_transitions:
                # Plot transition regions as contour
                transitions = phase_transitions[inv_name]['transition_mask']
                gradient_mag = phase_transitions[inv_name]['gradient_magnitude']
                
                # Plot gradient magnitude contour
                plt.contour(fd_mesh, factor_mesh, gradient_mag, 
                           levels=np.linspace(np.nanmin(gradient_mag), np.nanmax(gradient_mag), 5),
                           colors='white', alpha=0.5, linewidths=1)
                
                # Mark critical points
                for point in phase_transitions[inv_name]['critical_points']:
                    plt.plot(point['fd'], point['scaling_factor'], 'r*', markersize=10)
            
            plt.xlabel('Fractal Dimension')
            plt.ylabel('Scaling Factor')
            plt.title(f'Phase Diagram: {inv_name} vs. Fractal Dimension and Scaling Factor')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(output_path / f"fractal_topology_phase_diagram_{inv_name}.png", dpi=300)
            plt.close()
            
            # Plot combined phase diagram showing critical points
            if inv_name in phase_transitions:
                plt.figure(figsize=(10, 8))
                
                # Plot invariant value as background
                plt.contourf(fd_mesh, factor_mesh, inv_grid, 20, cmap='viridis', alpha=0.7)
                plt.colorbar(label=f'{inv_name} Value')
                
                # Overlay gradient magnitude
                gradient_mag = phase_transitions[inv_name]['gradient_magnitude']
                gradient_norm = gradient_mag / np.nanmax(gradient_mag)
                
                # Use a different colormap for gradient
                plt.contour(fd_mesh, factor_mesh, gradient_norm, 
                           levels=np.linspace(0.5, 1.0, 5),
                           colors='white', alpha=0.8, linewidths=1)
                
                # Mark critical points
                for point in phase_transitions[inv_name]['critical_points']:
                    plt.plot(point['fd'], point['scaling_factor'], 'r*', markersize=10,
                            label=f"Critical Point: ({point['fd']:.2f}, {point['scaling_factor']:.2f})")
                
                # Add golden ratio line if in range
                phi = (1 + np.sqrt(5)) / 2
                if min(scaling_factors) <= phi <= max(scaling_factors):
                    plt.axhline(y=phi, color='yellow', linestyle='--', 
                               label=f'Golden Ratio (Φ ≈ {phi:.3f})')
                
                plt.xlabel('Fractal Dimension')
                plt.ylabel('Scaling Factor')
                plt.title(f'Phase Transitions in {inv_name} vs. Fractal Dimension and Scaling Factor')
                plt.grid(True, alpha=0.3)
                
                # Add legend for the first few critical points only
                handles, labels = plt.gca().get_legend_handles_labels()
                if len(handles) > 6:  # Limit to 5 critical points + phi line
                    handles = handles[:6]
                    labels = labels[:6]
                plt.legend(handles, labels, loc='best')
                
                plt.tight_layout()
                plt.savefig(output_path / f"fractal_topology_phase_transitions_{inv_name}.png", dpi=300)
                plt.close()
    
    # Save results to CSV
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save correlations
        correlation_data = []
        for inv_name, corr_info in correlations.items():
            for factor, factor_corr in corr_info['factor_correlations'].items():
                row = {
                    'invariant': inv_name,
                    'scaling_factor': factor,
                    'correlation': factor_corr['correlation'],
                    'p_value': factor_corr['p_value'],
                    'num_samples': factor_corr['num_samples']
                }
                correlation_data.append(row)
        
        # Add overall correlations
        for inv_name, corr_info in correlations.items():
            row = {
                'invariant': inv_name,
                'scaling_factor': 'overall',
                'correlation': corr_info['overall_correlation'],
                'p_value': corr_info['overall_p_value'],
                'num_samples': corr_info['num_samples']
            }
            correlation_data.append(row)
        
        correlation_df = pd.DataFrame(correlation_data)
        correlation_df.to_csv(output_path / "fractal_topology_correlation.csv", index=False)
        
        # Save critical points
        critical_point_data = []
        for inv_name, transition_info in phase_transitions.items():
            for point in transition_info['critical_points']:
                row = {
                    'invariant': inv_name,
                    'fractal_dimension': point['fd'],
                    'scaling_factor': point['scaling_factor'],
                    'invariant_value': point['invariant_value'],
                    'gradient_magnitude': point['gradient_magnitude']
                }
                critical_point_data.append(row)
        
        critical_point_df = pd.DataFrame(critical_point_data)
        critical_point_df.to_csv(output_path / "fractal_topology_critical_points.csv", index=False)
    
    return results

if __name__ == "__main__":
    # Simple example usage
    # Define some synthetic data
    scaling_factors = [0.5, 0.75, 1.0, 1.25, 1.5, (1 + np.sqrt(5)) / 2, 2.0, 2.5, 3.0]
    
    # Generate synthetic fractal dimensions
    np.random.seed(42)
    fractal_dimensions = {
        factor: 1.2 + 0.3 * np.sin(factor * np.pi) + 0.05 * np.random.randn(10)
        for factor in scaling_factors
    }
    
    # Generate synthetic topological invariants
    topological_invariants = {
        factor: {
            'chern_number': np.round(0.5 + 0.5 * np.tanh((factor - 1.7) * 2) + 0.2 * np.random.randn(10)),
            'winding_number': np.round(factor / 2) + 0.1 * np.random.randn(10),
            'z2_index': np.round(np.abs(np.sin(factor * np.pi))) + 0.05 * np.random.randn(10)
        }
        for factor in scaling_factors
    }
    
    # Run analysis
    results = analyze_fractal_topology_relation(
        fractal_dimensions,
        topological_invariants,
        scaling_factors,
        output_dir='data',
        plot=True
    )
    
    # Print results
    print("\nCorrelations between fractal dimensions and topological invariants:")
    for inv_name, corr_info in results['correlations'].items():
        print(f"\n{inv_name}:")
        print(f"  Overall correlation: {corr_info['overall_correlation']:.4f} "
              f"(p-value: {corr_info['overall_p_value']:.4f})")
        
        print("  By scaling factor:")
        for factor, factor_corr in sorted(corr_info['factor_correlations'].items()):
            print(f"    {factor:.3f}: {factor_corr['correlation']:.4f} "
                  f"(p-value: {factor_corr['p_value']:.4f})")
    
    print("\nIdentified phase transitions and critical points:")
    for inv_name, transition_info in results['phase_transitions'].items():
        print(f"\n{inv_name}:")
        
        if transition_info['critical_points']:
            print("  Critical points:")
            for i, point in enumerate(transition_info['critical_points']):
                print(f"    {i+1}. Fractal dimension: {point['fd']:.3f}, "
                      f"Scaling factor: {point['scaling_factor']:.3f}")
        else:
            print("  No critical points identified.")
