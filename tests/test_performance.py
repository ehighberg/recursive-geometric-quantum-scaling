"""
tests/test_performance.py

Performance benchmarks for visualization components of the Quantum Simulation Codebase.
"""

import unittest
import timeit
from qutip import Qobj, basis
import matplotlib.pyplot as plt

from analyses.visualization.state_plots import plot_state_evolution, plot_bloch_sphere, plot_state_matrix
from analyses.visualization.metric_plots import plot_metric_evolution, plot_metric_comparison, plot_metric_distribution
from analyses.visualization.circuit_diagrams import plot_circuit_diagram, export_circuit_diagram

class TestPerformance(unittest.TestCase):
    def setUp(self):
        # Create sample quantum states
        self.state_zero = Qobj(basis(2, 0))
        self.state_one = Qobj(basis(2, 1))
        self.state_plus = (self.state_zero + self.state_one).unit()
        self.density_matrix = self.state_plus * self.state_plus.dag()
        
        self.states = [self.state_zero, self.state_plus, self.state_one]
        self.times = [0, 1, 2]
        
        # Sample circuit components and connections
        self.components = [
            {'id': 0, 'type': 'gate', 'name': 'H', 'position': (1, 1)},
            {'id': 1, 'type': 'gate', 'name': 'CNOT', 'position': (2, 1)},
            {'id': 2, 'type': 'measure', 'name': 'M', 'position': (3, 1)}
        ]
        self.connections = [
            (0, 1),
            (1, 2)
        ]
    
    def test_plot_state_evolution_performance(self):
        """Benchmark the performance of plot_state_evolution."""
        def task():
            fig = plot_state_evolution(self.states, self.times, title="Performance Test")
            plt.close(fig)
        
        execution_time = timeit.timeit(task, number=10)
        print(f"plot_state_evolution executed 10 times in {execution_time:.2f} seconds.")
        self.assertLess(execution_time, 5.0, "plot_state_evolution is too slow.")
    
    def test_plot_bloch_sphere_performance(self):
        """Benchmark the performance of plot_bloch_sphere."""
        def task():
            fig = plot_bloch_sphere(self.state_plus, title="Performance Test")
            plt.close(fig)
        
        execution_time = timeit.timeit(task, number=10)
        print(f"plot_bloch_sphere executed 10 times in {execution_time:.2f} seconds.")
        self.assertLess(execution_time, 5.0, "plot_bloch_sphere is too slow.")
    
    def test_plot_state_matrix_performance(self):
        """Benchmark the performance of plot_state_matrix."""
        def task():
            fig = plot_state_matrix(self.density_matrix, title="Performance Test")
            plt.close(fig)
        
        execution_time = timeit.timeit(task, number=10)
        print(f"plot_state_matrix executed 10 times in {execution_time:.2f} seconds.")
        self.assertLess(execution_time, 5.0, "plot_state_matrix is too slow.")
    
    def test_plot_metric_evolution_performance(self):
        """Benchmark the performance of plot_metric_evolution."""
        def task():
            fig = plot_metric_evolution(self.states, self.times, title="Performance Test")
            plt.close(fig)
        
        execution_time = timeit.timeit(task, number=10)
        print(f"plot_metric_evolution executed 10 times in {execution_time:.2f} seconds.")
        self.assertLess(execution_time, 5.0, "plot_metric_evolution is too slow.")
    
    def test_plot_metric_comparison_performance(self):
        """Benchmark the performance of plot_metric_comparison."""
        metric_pairs = [('vn_entropy', 'l1_coherence'), ('vn_entropy', 'negativity')]
        
        def task():
            fig = plot_metric_comparison(self.states, metric_pairs, title="Performance Test")
            plt.close(fig)
        
        execution_time = timeit.timeit(task, number=10)
        print(f"plot_metric_comparison executed 10 times in {execution_time:.2f} seconds.")
        self.assertLess(execution_time, 5.0, "plot_metric_comparison is too slow.")
    
    def test_plot_metric_distribution_performance(self):
        """Benchmark the performance of plot_metric_distribution."""
        def task():
            fig = plot_metric_distribution(self.states, title="Performance Test")
            plt.close(fig)
        
        execution_time = timeit.timeit(task, number=10)
        print(f"plot_metric_distribution executed 10 times in {execution_time:.2f} seconds.")
        self.assertLess(execution_time, 5.0, "plot_metric_distribution is too slow.")
    
    def test_plot_circuit_diagram_performance(self):
        """Benchmark the performance of plot_circuit_diagram."""
        def task():
            fig = plot_circuit_diagram(self.components, self.connections, title="Performance Test")
            plt.close(fig)
        
        execution_time = timeit.timeit(task, number=10)
        print(f"plot_circuit_diagram executed 10 times in {execution_time:.2f} seconds.")
        self.assertLess(execution_time, 10.0, "plot_circuit_diagram is too slow.")
    
    def test_export_circuit_diagram_performance(self):
        """Benchmark the performance of export_circuit_diagram."""
        def task():
            fig = plot_circuit_diagram(self.components, self.connections, title="Performance Test")
            export_circuit_diagram(fig, "test_circuit_diagram", format="png")
            plt.close(fig)
        
        execution_time = timeit.timeit(task, number=10)
        print(f"export_circuit_diagram executed 10 times in {execution_time:.2f} seconds.")
        self.assertLess(execution_time, 10.0, "export_circuit_diagram is too slow.")

if __name__ == '__main__':
    unittest.main()