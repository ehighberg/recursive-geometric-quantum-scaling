#!/usr/bin/env python
"""
Run quantum state evolution with phi scaling and analyze entanglement dynamics.
This script generates data for the dynamical perspective section of the paper.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from constants import PHI

# Import simulation components
from simulations.scripts.evolve_state import run_state_evolution
from analyses.entanglement_dynamics import (
    plot_entanglement_entropy_vs_time,
    plot_entanglement_spectrum,
    plot_entanglement_growth_rate
)
from analyses.visualization.wavepacket_plots import (
    plot_wavepacket_evolution,
    plot_wavepacket_spacetime
)

def run_evolution_analysis():
    """
    Run quantum state evolution with phi scaling and analyze the results.
    """
    print(f"Running state evolution with phi={PHI} scaling...")
    
    # Run evolution with phi scaling
    result_phi = run_state_evolution(
        num_qubits=2,
        state_label="ghz",  # Use entangled state
        n_steps=100,
        scaling_factor=PHI,
        analyze_fractal=True
    )
    
    # Run evolution with unit scaling for comparison
    result_unit = run_state_evolution(
        num_qubits=2,
        state_label="ghz",  # Use entangled state
        n_steps=100,
        scaling_factor=1.0,
        analyze_fractal=True
    )
    
    # Create coordinates for wavepacket visualization
    coordinates = np.linspace(0, 1, 2**result_phi.states[0].dims[0][0])
    
    # Plot entanglement entropy evolution
    fig_entropy_phi = plot_entanglement_entropy_vs_time(
        result_phi.states,
        result_phi.times,
        title=f"Entanglement Entropy Evolution (φ={PHI:.4f})"
    )
    # Create plots directory if it doesn't exist
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    fig_entropy_phi.savefig(plots_dir / "entanglement_entropy_phi.png", dpi=300, bbox_inches='tight')
    
    fig_entropy_unit = plot_entanglement_entropy_vs_time(
        result_unit.states,
        result_unit.times,
        title="Entanglement Entropy Evolution (Unit Scaling)"
    )
    fig_entropy_unit.savefig(plots_dir / "entanglement_entropy_unit.png", dpi=300, bbox_inches='tight')
    
    # Plot entanglement spectrum
    fig_spectrum_phi = plot_entanglement_spectrum(
        result_phi.states,
        result_phi.times,
        title=f"Entanglement Spectrum (φ={PHI:.4f})"
    )
    fig_spectrum_phi.savefig(plots_dir / "entanglement_spectrum_phi.png", dpi=300, bbox_inches='tight')
    
    # Plot entanglement growth rate
    fig_growth_phi = plot_entanglement_growth_rate(
        result_phi.states,
        result_phi.times,
        title=f"Entanglement Growth Rate (φ={PHI:.4f})"
    )
    fig_growth_phi.savefig(plots_dir / "entanglement_growth_phi.png", dpi=300, bbox_inches='tight')
    
    # Plot wavepacket evolution
    fig_wavepacket_phi = plot_wavepacket_evolution(
        result_phi.states,
        result_phi.times,
        coordinates=coordinates,
        title=f"Wavepacket Evolution (φ={PHI:.4f})"
    )
    fig_wavepacket_phi.savefig(plots_dir / "wavepacket_evolution_phi.png", dpi=300, bbox_inches='tight')
    
    fig_wavepacket_unit = plot_wavepacket_evolution(
        result_unit.states,
        result_unit.times,
        coordinates=coordinates,
        title="Wavepacket Evolution (Unit Scaling)"
    )
    fig_wavepacket_unit.savefig(plots_dir / "wavepacket_evolution_unit.png", dpi=300, bbox_inches='tight')
    
    # Plot wavepacket spacetime diagram
    fig_spacetime_phi = plot_wavepacket_spacetime(
        result_phi.states,
        result_phi.times,
        coordinates=coordinates,
        title=f"Wavepacket Spacetime (φ={PHI:.4f})"
    )
    fig_spacetime_phi.savefig(plots_dir / "wavepacket_spacetime_phi.png", dpi=300, bbox_inches='tight')
    
    # Create comparison plot for phi vs unit scaling
    plt.figure(figsize=(12, 8))
    
    # Extract fractal dimensions
    if hasattr(result_phi, 'fractal_dimensions') and hasattr(result_unit, 'fractal_dimensions'):
        plt.subplot(2, 2, 1)
        plt.plot(result_phi.recursion_depths, result_phi.fractal_dimensions, 'o-', label=f'φ={PHI:.4f}')
        plt.plot(result_unit.recursion_depths, result_unit.fractal_dimensions, 's--', label='Unit Scaling')
        plt.xlabel('Recursion Depth')
        plt.ylabel('Fractal Dimension')
        plt.title('Fractal Dimension vs Recursion Depth')
        plt.legend()
        plt.grid(alpha=0.3)
    
    # Compare final states
    plt.subplot(2, 2, 2)
    phi_final = np.abs(result_phi.states[-1].full().flatten())**2
    unit_final = np.abs(result_unit.states[-1].full().flatten())**2
    x = np.arange(len(phi_final))
    plt.bar(x - 0.2, phi_final, width=0.4, label=f'φ={PHI:.4f}')
    plt.bar(x + 0.2, unit_final, width=0.4, label='Unit Scaling')
    plt.xlabel('State Index')
    plt.ylabel('Probability')
    plt.title('Final State Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Compare energy spectra if available
    if hasattr(result_phi, 'energies') and hasattr(result_unit, 'energies'):
        plt.subplot(2, 2, 3)
        plt.plot(result_phi.parameter_values, result_phi.energies[:, 0], 'o-', label=f'φ={PHI:.4f} (Ground)')
        plt.plot(result_unit.parameter_values, result_unit.energies[:, 0], 's--', label='Unit (Ground)')
        plt.xlabel('Parameter Value')
        plt.ylabel('Energy')
        plt.title('Ground State Energy Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
    
    # Compare fractal analysis metrics
    plt.subplot(2, 2, 4)
    metrics_phi = [
        np.mean(result_phi.fractal_dimensions) if hasattr(result_phi, 'fractal_dimensions') else 0,
        np.max(np.abs(result_phi.states[-1].full())) if hasattr(result_phi, 'states') else 0,
        np.mean(np.abs(result_phi.states[-1].full())) if hasattr(result_phi, 'states') else 0
    ]
    metrics_unit = [
        np.mean(result_unit.fractal_dimensions) if hasattr(result_unit, 'fractal_dimensions') else 0,
        np.max(np.abs(result_unit.states[-1].full())) if hasattr(result_unit, 'states') else 0,
        np.mean(np.abs(result_unit.states[-1].full())) if hasattr(result_unit, 'states') else 0
    ]
    metrics_labels = ['Avg Fractal Dim', 'Max Amplitude', 'Avg Amplitude']
    x = np.arange(len(metrics_labels))
    plt.bar(x - 0.2, metrics_phi, width=0.4, label=f'φ={PHI:.4f}')
    plt.bar(x + 0.2, metrics_unit, width=0.4, label='Unit Scaling')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Fractal Analysis Metrics')
    plt.xticks(x, metrics_labels)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "phi_vs_unit_comparison.png", dpi=300, bbox_inches='tight')
    
    print("Evolution analysis complete. Results saved to PNG files.")

if __name__ == "__main__":
    run_evolution_analysis()
