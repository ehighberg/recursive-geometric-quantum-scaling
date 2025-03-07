
"""
Functions for generating computational performance tables for quantum simulations.
"""

import time
from typing import Dict, List, Optional, Callable, Any
import pandas as pd

# Try to import psutil, but make it optional
# Note: psutil is an optional dependency for memory tracking
# It's listed in requirements.txt as optional
try:
    import psutil  # pylint: disable=import-error
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def generate_performance_table(
    system_sizes: Optional[List[int]] = None,
    methods: Optional[List[str]] = None,
    results: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate a computational performance table.
    
    Parameters:
        system_sizes: List of system sizes (number of qubits)
        methods: List of simulation methods
        results: Optional pre-computed results dictionary
        
    Returns:
        pd.DataFrame: Performance table
    """
    if system_sizes is None:
        system_sizes = [1, 2, 3, 4]
    
    if methods is None:
        methods = ["Standard Evolution", "Phi-Scaled Evolution", "Topological Braiding"]
    
    # If results are provided, use them
    if results is not None:
        return _generate_performance_table_from_results(results)
    
    # Otherwise, create a template table
    performance_data = []
    for size in system_sizes:
        for method in methods:
            performance_data.append({
                "System Size": size,
                "Method": method,
                "CPU Time (s)": "N/A",
                "Memory (MB)": "N/A",
                "Convergence": "N/A"
            })
    
    return pd.DataFrame(performance_data)

def _generate_performance_table_from_results(results: Dict) -> pd.DataFrame:
    """
    Generate performance table from pre-computed results.
    
    Parameters:
        results: Dictionary containing performance results
        
    Returns:
        pd.DataFrame: Performance table
    """
    if 'computation_times' not in results:
        return pd.DataFrame()
    
    performance_data = []
    
    # Extract computation times
    for component, time_value in results['computation_times'].items():
        performance_data.append({
            "Component": component,
            "CPU Time (s)": f"{time_value:.2f}",
            "Memory (MB)": "N/A",  # Memory data not available in current implementation
            "Convergence": "N/A"   # Convergence data not available in current implementation
        })
    
    return pd.DataFrame(performance_data)

def measure_performance(
    func: Callable, 
    *args: Any, 
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Measure performance metrics for a function.
    
    Parameters:
        func: Function to measure
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Dict: Performance metrics
    """
    # Start timing
    start_time = time.time()
    
    # Track memory if psutil is available
    start_memory = 0
    if HAS_PSUTIL:
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Run function
    result = func(*args, **kwargs)
    
    # End timing
    end_time = time.time()
    
    # Track memory if psutil is available
    end_memory = 0
    if HAS_PSUTIL:
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    memory_used = end_memory - start_memory if HAS_PSUTIL else 0
    
    # Extract convergence metrics if available
    if hasattr(result, 'convergence_metrics'):
        convergence = result.convergence_metrics
    else:
        convergence = "N/A"
    
    return {
        "Execution Time (s)": elapsed_time,
        "Memory Usage (MB)": memory_used,
        "Convergence Metric": convergence,
        "Result": result
    }

def generate_scaling_performance_table(
    system_sizes: List[int],
    method_func: Callable,
    method_args: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generate a table showing how performance scales with system size.
    
    Parameters:
        system_sizes: List of system sizes to test
        method_func: Function to measure
        method_args: Base arguments for the function
        
    Returns:
        pd.DataFrame: Scaling performance table
    """
    performance_data = []
    
    for size in system_sizes:
        # Update arguments with current system size
        args = method_args.copy()
        args['num_qubits'] = size
        
        # Measure performance
        perf = measure_performance(method_func, **args)
        
        # Add to table
        performance_data.append({
            "System Size": size,
            "CPU Time (s)": f"{perf['Execution Time (s)']:.2f}",
            "Memory (MB)": f"{perf['Memory Usage (MB)']:.1f}",
            "Scaling Factor": f"{perf['Execution Time (s)'] / (2**size):.4f}"  # Time per Hilbert space dimension
        })
    
    return pd.DataFrame(performance_data)

def generate_convergence_table(
    result,
    error_metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate a table showing convergence metrics for a simulation.
    
    Parameters:
        result: Simulation result object
        error_metrics: List of error metric names to include
        
    Returns:
        pd.DataFrame: Convergence table
    """
    if error_metrics is None:
        error_metrics = ['fidelity', 'trace_distance', 'relative_error']
    
    convergence_data = []
    
    # Extract convergence metrics if available
    if hasattr(result, 'convergence_metrics'):
        metrics = result.convergence_metrics
        for metric_name, value in metrics.items():
            if metric_name in error_metrics:
                convergence_data.append({
                    "Metric": metric_name,
                    "Value": f"{value:.6e}",
                    "Threshold": "1e-6",  # Default threshold
                    "Converged": "Yes" if value < 1e-6 else "No"
                })
    
    # If no convergence metrics available, create placeholder
    if not convergence_data:
        for metric in error_metrics:
            convergence_data.append({
                "Metric": metric,
                "Value": "N/A",
                "Threshold": "1e-6",
                "Converged": "N/A"
            })
    
    return pd.DataFrame(convergence_data)

def generate_method_comparison_table(
    methods: Dict[str, Callable],
    test_case: Dict[str, Any],
    system_size: int = 2
) -> pd.DataFrame:
    """
    Generate a table comparing different numerical methods.
    
    Parameters:
        methods: Dictionary mapping method names to functions
        test_case: Arguments for the test case
        system_size: System size to use
        
    Returns:
        pd.DataFrame: Method comparison table
    """
    comparison_data = []
    
    # Update test case with system size
    test_args = test_case.copy()
    test_args['num_qubits'] = system_size
    
    for method_name, method_func in methods.items():
        # Measure performance
        perf = measure_performance(method_func, **test_args)
        
        # Add to table
        comparison_data.append({
            "Method": method_name,
            "CPU Time (s)": f"{perf['Execution Time (s)']:.2f}",
            "Memory (MB)": f"{perf['Memory Usage (MB)']:.1f}",
            "Accuracy": "N/A"  # Would need reference solution to compute accuracy
        })
    
    return pd.DataFrame(comparison_data)
