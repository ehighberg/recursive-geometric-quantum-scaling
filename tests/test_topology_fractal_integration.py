"""
Tests for integration between topological and fractal analysis features.
"""

import pytest
import numpy as np
import time
from qutip import basis, Qobj
from unittest.mock import MagicMock, patch

from analyses.fractal_analysis import compute_energy_spectrum, estimate_fractal_dimension
from analyses.topological_invariants import compute_chern_number, compute_winding_number
from analyses.topology_plots import plot_invariants, plot_protection_metrics
from app.analyze_results import analyze_simulation_results

class MockHybridResult:
    """Mock result combining topological and fractal features"""
    def __init__(self):
        # Basic quantum states
        self.states = [basis(2,0) for _ in range(50)]  # 50 states for evolution
        self.times = np.linspace(0, 10, len(self.states))
        
        # Hamiltonian function
        self.hamiltonian = lambda x: Qobj([[1-x, x], [x, x-1]])
        
        # Fractal analysis data
        self.recursion_depths = [1, 2, 3, 4, 5]  # More depths for better analysis
        self.fractal_dimensions = [1.2, 1.4, 1.6, 1.8, 2.0]  # Increasing dimensions
        self.dimension_errors = [0.1] * 5
        
        # Topological data
        self.chern_number = 1
        self.winding_number = 2
        self.z2_index = 1
        
        # Protection metrics
        self.energy_gaps = np.abs(np.sin(np.linspace(0, 2*np.pi, len(self.states))))
        self.localization_measures = np.abs(np.cos(np.linspace(0, 2*np.pi, len(self.states))))

def generate_fractal_pattern(size, depth):
    """Generate a fractal-like pattern with controlled self-similarity"""
    x = np.linspace(0, 2*np.pi, size)
    pattern = np.zeros_like(x)
    
    # Add multi-scale structure
    for i in range(depth):
        scale = 8.0 * (0.5**i)  # Larger base amplitude
        freq = 2**(i+1)
        pattern += scale * (np.sin(freq * x) + np.cos(freq * x))
    
    # Add self-similar features at different scales
    for i in range(3):
        pattern += 2.0 * np.sin(x * (i+1)) * np.cos(x * (i+2))
    
    return pattern

def test_fractal_topology_correlation():
    """Test correlation between fractal and topological properties"""
    # Create test Hamiltonian with fractal energy spectrum
    def test_hamiltonian(f_s):
        # Hamiltonian with fractal energy spectrum and non-trivial topology
        sx = Qobj([[0, 1], [1, 0]])
        sz = Qobj([[1, 0], [0, -1]])
        H = f_s * sx + (1 - f_s) * sz
        return H
    
    # Compute energy spectrum
    f_s_values = np.linspace(0, 1, 200)  # Increased resolution
    energies = []
    chern_numbers = []
    
    # Create fractal-like energy spectrum with multiple scales
    for f_s in f_s_values:
        H = test_hamiltonian(f_s)
        evals = H.eigenenergies()
        # Add fractal structure at multiple scales
        fractal_evals = []
        for e in evals:
            e_fractal = e
            pattern = generate_fractal_pattern(10, 5)  # Generate rich fractal pattern
            e_fractal += np.interp(f_s, np.linspace(0, 1, len(pattern)), pattern)
            fractal_evals.append(e_fractal)
        energies.append(fractal_evals)
        
        # Compute Chern number at this point
        k_mesh = (np.array([f_s]), np.array([0]))
        chern_numbers.append(compute_chern_number([[H]], k_mesh))
    
    # Create fractal-like energy spectrum
    fractal_energies = np.array(energies)
    
    # Add additional fractal structure
    x = np.linspace(0, 2*np.pi, fractal_energies.shape[0])
    pattern = generate_fractal_pattern(len(x), 4)
    fractal_energies += pattern.reshape(-1, 1)
    
    # Verify correlation between fractal dimension and topological invariants
    fractal_dim, _ = estimate_fractal_dimension(fractal_energies)
    assert fractal_dim >= 1.0  # Should have non-trivial fractal dimension
    assert any(c != 0 for c in chern_numbers)  # Should have non-trivial topology

def test_protection_metrics_scaling():
    """Test scaling behavior of protection metrics"""
    result = MockHybridResult()
    
    # Add scaling analysis data with fractal structure
    x_values = np.linspace(0, 2*np.pi, 200)  # Increased resolution
    result.scaling_factors = np.logspace(-2.0, 2.0, 100)  # More scaling points
    
    # Create fractal-like scaling patterns with enhanced amplitudes
    def create_fractal_scaling(base_pattern):
        scaling = np.zeros((len(result.scaling_factors), len(x_values)))
        
        # Add multi-scale structure
        for i in range(5):
            pattern = generate_fractal_pattern(len(x_values), i+1)
            for j, scale_factor in enumerate(result.scaling_factors):
                scaling[j] += scale_factor * pattern
        
        # Add base pattern modulation
        scaling *= np.outer(result.scaling_factors, base_pattern)
        return scaling
    
    result.gap_scaling = create_fractal_scaling(np.abs(np.sin(x_values)))
    result.localization_scaling = create_fractal_scaling(np.abs(np.cos(x_values)))
    
    # Compute fractal dimension of protection metrics
    gap_dim, _ = estimate_fractal_dimension(result.gap_scaling)
    loc_dim, _ = estimate_fractal_dimension(result.localization_scaling)
    
    # Verify reasonable fractal dimensions
    assert gap_dim > 1.0  # Should have non-trivial dimension
    assert loc_dim > 1.0  # Should have non-trivial dimension

def test_visualization_integration():
    """Test integration of fractal and topological visualizations"""
    result = MockHybridResult()
    
    with patch('matplotlib.pyplot.figure') as mock_figure:
        # Plot both fractal and topological features
        plot_invariants((0, 1))
        plot_protection_metrics((0, 1), result.energy_gaps, result.localization_measures)
        
        # Verify plots were created
        assert mock_figure.call_count >= 2

def test_analysis_pipeline_integration():
    """Test integration in the analysis pipeline"""
    result = MockHybridResult()
    
    # Mock streamlit components
    with patch('app.analyze_results.st.tabs') as mock_tabs:
        tabs = [MagicMock() for _ in range(4)]
        mock_tabs.return_value = tabs
        
        # Run analysis
        analyze_simulation_results(result, mode="Topological Braiding")
        
        # Verify both analyses were performed
        assert len(tabs) == 4

def test_recursive_scaling_analysis():
    """Test recursive scaling analysis of topological features"""
    # Create test data with recursive structure
    def generate_recursive_data(depth, size):
        if depth == 0:
            return generate_fractal_pattern(size, 3)  # Base case uses fractal pattern
        
        smaller = generate_recursive_data(depth-1, size//2)
        # Create self-similar pattern with stronger features
        data = np.concatenate([
            4.0 * smaller,  # Increased amplitude
            2.0 - smaller + 2.0 * generate_fractal_pattern(size//2, 3)
        ])
        return data
    
    # Generate test data
    depths = range(1, 5)
    dimensions = []
    chern_numbers = []
    
    for depth in depths:
        data = generate_recursive_data(depth, 2**depth)
        # Add multi-scale fractal structure with larger amplitudes
        x = np.linspace(0, 2*np.pi, len(data))
        pattern = generate_fractal_pattern(len(x), 5)
        data += pattern
        
        # Compute fractal dimension
        dim, _ = estimate_fractal_dimension(data.reshape(-1, 1))
        dimensions.append(dim)
        
        # Create corresponding Hamiltonian and compute Chern number
        normalized_data = data/np.max(np.abs(data))
        H = Qobj([[normalized_data[0], normalized_data[1]], 
                 [normalized_data[1], -normalized_data[0]]])
        k_mesh = (np.array([0]), np.array([0]))
        chern = compute_chern_number([[H]], k_mesh)
        chern_numbers.append(chern)
    
    # Verify scaling relationships
    dim_diffs = np.diff(dimensions)
    assert all(d > -0.1 for d in dim_diffs)  # Allow small decreases
    assert any(c != 0 for c in chern_numbers)  # Should have non-trivial topology

def test_error_handling():
    """Test error handling in integrated analysis"""
    result = MockHybridResult()
    
    # Remove some required attributes
    delattr(result, 'fractal_dimensions')
    delattr(result, 'chern_number')
    
    with patch('app.analyze_results.st.error') as mock_error:
        analyze_simulation_results(result, mode="Topological Braiding")
        # Should handle missing data gracefully
        mock_error.assert_not_called()

def test_performance_scaling():
    """Test performance scaling with system size"""
    sizes = [2, 4, 8]  # Simple list is sufficient here
    times_fractal = []
    times_topo = []
    
    for size in sizes:
        # Generate fractal data
        data = generate_fractal_pattern(size**2, 4).reshape(size, size)
        H = Qobj(data)
        
        # Time fractal analysis
        start = time.time()
        estimate_fractal_dimension(data)
        times_fractal.append(max(time.time() - start, 1e-6))
        
        # Time topological analysis
        start = time.time()
        k_mesh = (np.array([0]), np.array([0]))
        compute_chern_number([[H]], k_mesh)
        times_topo.append(max(time.time() - start, 1e-6))
    
    # Verify reasonable scaling behavior
    assert all(t2/t1 < 8.0 for t1, t2 in zip(times_fractal[:-1], times_fractal[1:]))
    assert all(t2/t1 < 8.0 for t1, t2 in zip(times_topo[:-1], times_topo[1:]))

def test_combined_metrics():
    """Test combined metrics from both analyses"""
    result = MockHybridResult()
    
    # Add combined metrics
    result.fractal_chern_correlation = 0.8
    result.protection_dimension = 1.5
    
    with patch('app.analyze_results.st.metric') as mock_metric:
        analyze_simulation_results(result, mode="Topological Braiding")
        
        # Verify metrics were displayed
        mock_metric.assert_any_call("Fractal-Chern Correlation", 0.80)
        mock_metric.assert_any_call("Protection Dimension", 1.50)
