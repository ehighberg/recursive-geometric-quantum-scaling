"""
Simplified visualization framework for quantum physics simulations.

This module provides a unified interface for creating high-quality visualizations
of quantum phenomena, with particular focus on phi-based scaling effects.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from qutip import Qobj
import logging
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# Set up logging
logger = logging.getLogger(__name__)

# Import visualization components
from analyses.visualization.metric_plots import (
    calculate_metrics, plot_metric_evolution, 
    plot_comparative_metrics, plot_quantum_metrics
)
from analyses.visualization.wavepacket_plots import (
    compute_wavepacket_probability, plot_wavepacket_evolution,
    plot_comparative_wavepacket_evolution, plot_wavepacket_spacetime
)
from analyses.visualization.style_config import (
    set_style, configure_axis, get_color_cycle, COLORS
)

# Constants
PHI = 1.618033988749895  # Golden ratio

class QuantumVisualizer:
    """
    Unified visualizer for quantum simulations with standardized styles and formats.
    
    This class provides a unified interface for creating visualizations of quantum
    phenomena, using actual quantum mechanics calculations rather than predetermined
    or biased data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the quantum visualizer.
        
        Parameters:
        -----------
        config : Dict[str, Any], optional
            Configuration dictionary with visualization options
        """
        # Default configuration
        self.config = {
            'output_dir': 'figures',
            'dpi': 300,
            'style': 'default',
            'color_scheme': 'default',
            'figsize_standard': (10, 6),
            'figsize_large': (12, 8),
            'figsize_grid': (15, 10),
            'show_plots': False
        }
        
        # Update with user configuration
        if config:
            self.config.update(config)
        
        # Create output directory
        if self.config['output_dir']:
            Path(self.config['output_dir']).mkdir(exist_ok=True, parents=True)
        
        # Set style
        set_style(self.config['style'])
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> str:
        """
        Save figure to disk.
        
        Parameters:
        -----------
        fig : plt.Figure
            Matplotlib figure to save
        filename : str
            Output filename
        
        Returns:
        --------
        str
            Path to saved figure
        """
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            filename += '.png'
        
        output_path = Path(self.config['output_dir']) / filename
        
        try:
            fig.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Figure saved to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error saving figure: {e}")
            return ""
    
    def visualize_wavefunction(
        self,
        state: Qobj,
        coordinates: np.ndarray,
        scaling_factor: Optional[float] = None,
        title: Optional[str] = None,
        output_filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize a quantum wavefunction probability distribution.
        
        Parameters:
        -----------
        state : Qobj
            Quantum state to visualize
        coordinates : np.ndarray
            Spatial coordinates for probability distribution
        scaling_factor : Optional[float]
            Scaling factor used to generate this state (for reference)
        title : Optional[str]
            Plot title
        output_filename : Optional[str]
            Filename for saving the figure
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.config['figsize_standard'])
        
        # Compute probability distribution
        probabilities = compute_wavepacket_probability(state, coordinates)
        
        # Plot wavefunction
        ax.plot(coordinates, probabilities, linewidth=2)
        
        # Add scaling factor to title if provided
        if title:
            if scaling_factor:
                title = f"{title} (f_s = {scaling_factor:.3f})"
        elif scaling_factor:
            title = f"Wavefunction (f_s = {scaling_factor:.3f})"
        else:
            title = "Quantum Wavefunction"
        
        # Configure axis
        configure_axis(ax, title=title, xlabel='Position', ylabel='Probability')
        
        # Tight layout
        fig.tight_layout()
        
        # Save figure if filename provided
        if output_filename:
            self._save_figure(fig, output_filename)
        
        return fig
    
    def visualize_metrics(
        self,
        states: List[Qobj],
        times: List[float],
        metrics: List[str] = None,
        title: Optional[str] = None,
        output_filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize quantum metrics for a time evolution.
        
        Parameters:
        -----------
        states : List[Qobj]
            List of quantum states at different time points
        times : List[float]
            Time points corresponding to states
        metrics : List[str]
            List of metrics to visualize
        title : Optional[str]
            Plot title
        output_filename : Optional[str]
            Filename for saving the figure
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['entropy', 'coherence', 'purity']
        
        # Create figure using metric_plots
        fig = plot_metric_evolution(
            states, times, title=title, 
            figsize=self.config['figsize_standard'],
            metrics=metrics
        )
        
        # Save figure if filename provided
        if output_filename:
            self._save_figure(fig, output_filename)
        
        return fig
    
    def visualize_comparative_evolution(
        self,
        states1: List[Qobj],
        states2: List[Qobj],
        times: List[float],
        labels: Tuple[str, str] = None,
        title: Optional[str] = None,
        output_filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize comparative evolution of two quantum systems.
        
        Parameters:
        -----------
        states1 : List[Qobj]
            States for first system
        states2 : List[Qobj]
            States for second system
        times : List[float]
            Time points
        labels : Tuple[str, str]
            Labels for the two systems
        title : Optional[str]
            Plot title
        output_filename : Optional[str]
            Filename for saving the figure
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        if labels is None:
            labels = ("System 1", "System 2")
        
        # Create figure
        fig = plt.figure(figsize=self.config['figsize_large'])
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # 1. Trajectory plot (positions over time)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Extract position expectation for both systems
        pos1 = []
        pos2 = []
        
        for state1, state2 in zip(states1, states2):
            # Calculate position expectation value
            try:
                # For pure states
                if state1.isket:
                    probs1 = np.abs(state1.full().flatten())**2
                else:
                    probs1 = np.real(state1.diag())
                
                if state2.isket:
                    probs2 = np.abs(state2.full().flatten())**2
                else:
                    probs2 = np.real(state2.diag())
                
                # Create position coordinates
                dim1 = len(probs1)
                dim2 = len(probs2)
                coords1 = np.linspace(0, 1, dim1)
                coords2 = np.linspace(0, 1, dim2)
                
                # Calculate expectation values
                pos1.append(np.sum(coords1 * probs1))
                pos2.append(np.sum(coords2 * probs2))
            except Exception as e:
                logger.warning(f"Error calculating position: {e}")
                # Use placeholder
                pos1.append(np.nan)
                pos2.append(np.nan)
        
        # Plot trajectories
        valid_times = times[:len(pos1)]
        ax1.plot(valid_times, pos1, '-', label=labels[0])
        ax1.plot(valid_times, pos2, '--', label=labels[1])
        
        # Configure axis
        configure_axis(ax1, 
                      title="Position Expectation Value", 
                      xlabel="Time", 
                      ylabel="Position")
        ax1.legend()
        
        # 2. Initial state comparison
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Only create if we have states
        if states1 and states2:
            # Create coordinates
            if states1[0].isket:
                dim1 = states1[0].shape[0]
            else:
                dim1 = states1[0].shape[0]
            
            if states2[0].isket:
                dim2 = states2[0].shape[0]
            else:
                dim2 = states2[0].shape[0]
            
            coords1 = np.linspace(0, 1, dim1)
            coords2 = np.linspace(0, 1, dim2)
            
            # Calculate probabilities
            probs1 = compute_wavepacket_probability(states1[0], coords1)
            probs2 = compute_wavepacket_probability(states2[0], coords2)
            
            # Plot initial states
            ax2.plot(coords1, probs1, '-', label=labels[0])
            ax2.plot(coords2, probs2, '--', label=labels[1])
            
            # Configure axis
            configure_axis(ax2, 
                          title="Initial States", 
                          xlabel="Position", 
                          ylabel="Probability")
            ax2.legend()
        
        # 3. Final state comparison
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Only create if we have states
        if states1 and states2:
            # Calculate probabilities of final states
            probs1 = compute_wavepacket_probability(states1[-1], coords1)
            probs2 = compute_wavepacket_probability(states2[-1], coords2)
            
            # Plot final states
            ax3.plot(coords1, probs1, '-', label=labels[0])
            ax3.plot(coords2, probs2, '--', label=labels[1])
            
            # Configure axis
            configure_axis(ax3, 
                          title="Final States", 
                          xlabel="Position", 
                          ylabel="Probability")
            ax3.legend()
        
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
        
        # Tight layout
        fig.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
        
        # Save figure if filename provided
        if output_filename:
            self._save_figure(fig, output_filename)
        
        return fig
    
    def visualize_state_evolution(
        self,
        states: List[Qobj],
        times: List[float],
        coordinates: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        output_filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the evolution of a quantum wavepacket over time.
        
        Parameters:
        -----------
        states : List[Qobj]
            List of quantum states at different time points
        times : List[float]
            Time points corresponding to states
        coordinates : Optional[np.ndarray]
            Spatial coordinates for probability distribution
        title : Optional[str]
            Plot title
        output_filename : Optional[str]
            Filename for saving the figure
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        # Create figure using wavepacket_plots
        fig = plot_wavepacket_evolution(
            states, times, coordinates=coordinates, 
            title=title, figsize=self.config['figsize_large']
        )
        
        # Save figure if filename provided
        if output_filename:
            self._save_figure(fig, output_filename)
        
        return fig
    
    def visualize_energy_spectrum(
        self,
        spectra: List[np.ndarray],
        scaling_factors: List[float],
        title: Optional[str] = None,
        output_filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize energy spectra for different scaling factors.
        
        Parameters:
        -----------
        spectra : List[np.ndarray]
            List of energy spectra (eigenvalues)
        scaling_factors : List[float]
            Corresponding scaling factors
        title : Optional[str]
            Plot title
        output_filename : Optional[str]
            Filename for saving the figure
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        # Ensure equal length
        if len(spectra) != len(scaling_factors):
            raise ValueError("Length of spectra must match length of scaling_factors")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config['figsize_standard'])
        
        # Sort scaling factors for better visualization
        sorted_indices = np.argsort(scaling_factors)
        sorted_factors = [scaling_factors[i] for i in sorted_indices]
        sorted_spectra = [spectra[i] for i in sorted_indices]
        
        # Find phi index if present
        phi_idx = None
        for i, factor in enumerate(sorted_factors):
            if abs(factor - PHI) < 1e-6:
                phi_idx = i
                break
        
        # Plot each spectrum
        for i, (factor, spectrum) in enumerate(zip(sorted_factors, sorted_spectra)):
            # Normalize spectrum for better comparison
            norm_spectrum = spectrum - np.min(spectrum)
            if np.max(norm_spectrum) > 0:
                norm_spectrum = norm_spectrum / np.max(norm_spectrum)
            
            # Use different style for phi
            if phi_idx is not None and i == phi_idx:
                ax.plot([factor] * len(norm_spectrum), norm_spectrum, 'o', 
                       markersize=4, alpha=0.7, label=f"φ = {PHI:.3f}")
            else:
                ax.plot([factor] * len(norm_spectrum), norm_spectrum, '.', 
                       markersize=3, alpha=0.5)
        
        # Connect eigenvalues between scaling factors
        max_eigenvalues = min(len(s) for s in sorted_spectra)
        for level in range(max_eigenvalues):
            level_values = []
            for i, spectrum in enumerate(sorted_spectra):
                if level < len(spectrum):
                    # Normalize
                    norm_val = (spectrum[level] - np.min(spectrum))
                    if np.max(spectrum - np.min(spectrum)) > 0:
                        norm_val = norm_val / np.max(spectrum - np.min(spectrum))
                    level_values.append(norm_val)
                else:
                    level_values.append(np.nan)
            
            # Plot connecting lines - semi-transparent
            ax.plot(sorted_factors, level_values, '-', linewidth=0.5, alpha=0.3, color='gray')
        
        # Highlight phi if present
        if phi_idx is not None:
            ax.axvline(x=PHI, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.legend()
        
        # Configure axis
        configure_axis(ax, 
                      title=title or "Energy Spectrum vs Scaling Factor",
                      xlabel="Scaling Factor", 
                      ylabel="Normalized Energy")
        
        # Tight layout
        fig.tight_layout()
        
        # Save figure if filename provided
        if output_filename:
            self._save_figure(fig, output_filename)
        
        return fig
    
    def visualize_phi_significance(
        self,
        data_by_factor: Dict[float, np.ndarray],
        phi_factor: float = PHI,
        metric_name: str = "Metric",
        title: Optional[str] = None,
        output_filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize statistical significance of phi in measurements.
        
        Parameters:
        -----------
        data_by_factor : Dict[float, np.ndarray]
            Dictionary mapping scaling factors to arrays of metric values
        phi_factor : float
            The phi value to highlight (default: golden ratio)
        metric_name : str
            Name of the metric being visualized
        title : Optional[str]
            Plot title
        output_filename : Optional[str]
            Filename for saving the figure
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        # Create figure with 2 plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Sort factors for consistent plotting
        factors = sorted(data_by_factor.keys())
        
        # Extract data
        mean_values = []
        std_errors = []
        all_data = []
        
        for factor in factors:
            data = data_by_factor[factor]
            all_data.append(data)
            mean_values.append(np.mean(data))
            
            # Standard error of the mean
            if len(data) > 1:
                std_errors.append(np.std(data, ddof=1) / np.sqrt(len(data)))
            else:
                std_errors.append(0)
        
        # Plot mean values with error bars
        ax1.errorbar(factors, mean_values, yerr=std_errors, fmt='o-', 
                    capsize=5, markersize=8)
        
        # Highlight phi
        if phi_factor in data_by_factor:
            phi_idx = factors.index(phi_factor)
            ax1.plot(phi_factor, mean_values[phi_idx], 'ro', markersize=12, 
                   label=f'φ = {phi_factor:.6f}')
            ax1.axvline(x=phi_factor, color='red', linestyle='--', alpha=0.3)
            ax1.legend()
        
        # Configure axis 1
        configure_axis(ax1, 
                      title=f"{metric_name} vs Scaling Factor",
                      xlabel="Scaling Factor", 
                      ylabel=metric_name)
        
        # 2. Box plot of distributions
        # Create labels for boxplot with scaling factors
        if any(len(d) > 1 for d in all_data):  # Only create if we have distributions
            box_labels = [f"{f:.2f}" for f in factors]
            ax2.boxplot(all_data, labels=box_labels)
            
            # Highlight phi
            if phi_factor in data_by_factor:
                phi_idx = factors.index(phi_factor)
                # Add highlighting to phi box
                box_elements = ax2.boxplot([data_by_factor[phi_factor]], positions=[phi_idx+1],
                                        patch_artist=True)
                for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                    plt.setp(box_elements[element], color='red')
                plt.setp(box_elements['boxes'], facecolor='pink', alpha=0.5)
            
            # Configure axis 2
            configure_axis(ax2, 
                          title=f"Distribution of {metric_name} by Scaling Factor",
                          xlabel="Scaling Factor", 
                          ylabel=metric_name)
            
            # Rotate x-tick labels for readability
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
        
        # Tight layout
        fig.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
        
        # Save figure if filename provided
        if output_filename:
            self._save_figure(fig, output_filename)
        
        return fig
    
    def visualize_comprehensive_quantum_metrics(
        self,
        states: List[Qobj],
        times: List[float],
        title: Optional[str] = None,
        output_filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive visualization of quantum metrics.
        
        Parameters:
        -----------
        states : List[Qobj]
            List of quantum states at different time points
        times : List[float]
            Time points corresponding to states
        title : Optional[str]
            Plot title
        output_filename : Optional[str]
            Filename for saving the figure
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        # Use metric_plots comprehensive visualization
        fig = plot_quantum_metrics(
            states, times, title=title,
            figsize=self.config['figsize_grid']
        )
        
        # Save figure if filename provided
        if output_filename:
            self._save_figure(fig, output_filename)
        
        return fig
    
    def generate_paper_figures(
        self,
        data_dict: Dict[str, Any],
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate a complete set of paper-quality figures from a data dictionary.
        
        Parameters:
        -----------
        data_dict : Dict[str, Any]
            Dictionary containing data for different figure types:
            - 'wavefunctions': Dict mapping names to (state, coordinates, scaling_factor)
            - 'evolutions': Dict mapping names to (states, times, coordinates)
            - 'comparisons': Dict mapping names to (states1, states2, times, labels)
            - 'metrics': Dict mapping names to (states, times, metric_list)
            - 'significance': Dict mapping names to (data_by_factor, phi_factor, metric_name)
            - 'spectra': Dict mapping names to (spectra, scaling_factors)
        output_dir : Optional[str]
            Directory to save figures (overrides default)
        
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping figure names to output paths
        """
        # Override output directory if provided
        original_output_dir = self.config['output_dir']
        if output_dir:
            self.config['output_dir'] = output_dir
            # Create directory
            Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        # Store output paths
        output_paths = {}
        
        try:
            # 1. Generate wavefunction figures
            if 'wavefunctions' in data_dict:
                for name, (state, coordinates, scaling_factor) in data_dict['wavefunctions'].items():
                    filename = f"wavefunction_{name}.png"
                    fig = self.visualize_wavefunction(
                        state, coordinates, scaling_factor,
                        title=f"Wavefunction - {name.replace('_', ' ').title()}",
                        output_filename=filename
                    )
                    plt.close(fig)
                    output_paths[f"wavefunction_{name}"] = Path(self.config['output_dir']) / filename
            
            # 2. Generate evolution figures
            if 'evolutions' in data_dict:
                for name, (states, times, coordinates) in data_dict['evolutions'].items():
                    filename = f"evolution_{name}.png"
                    fig = self.visualize_state_evolution(
                        states, times, coordinates,
                        title=f"State Evolution - {name.replace('_', ' ').title()}",
                        output_filename=filename
                    )
                    plt.close(fig)
                    output_paths[f"evolution_{name}"] = Path(self.config['output_dir']) / filename
            
            # 3. Generate comparison figures
            if 'comparisons' in data_dict:
                for name, (states1, states2, times, labels) in data_dict['comparisons'].items():
                    filename = f"comparison_{name}.png"
                    fig = self.visualize_comparative_evolution(
                        states1, states2, times, labels,
                        title=f"Comparative Evolution - {name.replace('_', ' ').title()}",
                        output_filename=filename
                    )
                    plt.close(fig)
                    output_paths[f"comparison_{name}"] = Path(self.config['output_dir']) / filename
            
            # 4. Generate metrics figures
            if 'metrics' in data_dict:
                for name, (states, times, metrics) in data_dict['metrics'].items():
                    filename = f"metrics_{name}.png"
                    fig = self.visualize_metrics(
                        states, times, metrics,
                        title=f"Quantum Metrics - {name.replace('_', ' ').title()}",
                        output_filename=filename
                    )
                    plt.close(fig)
                    output_paths[f"metrics_{name}"] = Path(self.config['output_dir']) / filename
                    
                    # Also generate comprehensive metrics
                    comp_filename = f"comprehensive_metrics_{name}.png"
                    fig = self.visualize_comprehensive_quantum_metrics(
                        states, times,
                        title=f"Comprehensive Metrics - {name.replace('_', ' ').title()}",
                        output_filename=comp_filename
                    )
                    plt.close(fig)
                    output_paths[f"comprehensive_metrics_{name}"] = Path(self.config['output_dir']) / comp_filename
            
            # 5. Generate significance figures
            if 'significance' in data_dict:
                for name, (data_by_factor, phi_factor, metric_name) in data_dict['significance'].items():
                    filename = f"significance_{name}.png"
                    fig = self.visualize_phi_significance(
                        data_by_factor, phi_factor, metric_name,
                        title=f"Statistical Analysis - {name.replace('_', ' ').title()}",
                        output_filename=filename
                    )
                    plt.close(fig)
                    output_paths[f"significance_{name}"] = Path(self.config['output_dir']) / filename
            
            # 6. Generate spectra figures
            if 'spectra' in data_dict:
                for name, (spectra, scaling_factors) in data_dict['spectra'].items():
                    filename = f"spectrum_{name}.png"
                    fig = self.visualize_energy_spectrum(
                        spectra, scaling_factors,
                        title=f"Energy Spectrum - {name.replace('_', ' ').title()}",
                        output_filename=filename
                    )
                    plt.close(fig)
                    output_paths[f"spectrum_{name}"] = Path(self.config['output_dir']) / filename
        
        finally:
            # Restore original output directory
            self.config['output_dir'] = original_output_dir
        
        return output_paths
    
    def _calculate_metrics(self, states: List[Qobj], metrics: List[str] = None) -> Dict[str, List[float]]:
        """
        Calculate quantum metrics for a sequence of states.
        
        Parameters:
        -----------
        states : List[Qobj]
            List of quantum states
        metrics : List[str]
            List of metrics to calculate
        
        Returns:
        --------
        Dict[str, List[float]]
            Dictionary mapping metric names to lists of values
        """
        if metrics is None:
            metrics = ['vn_entropy', 'l1_coherence', 'purity']
        
        # Use metric_plots module to calculate metrics
        return calculate_metrics(states)
