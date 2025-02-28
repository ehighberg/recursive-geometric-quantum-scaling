"""
tests/test_performance.py

Performance benchmarks for visualization components and quantum evolution.
"""

import unittest
import timeit
import numpy as np
from qutip import Qobj, basis, sigmaz, qeye, tensor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from analyses.visualization.state_plots import plot_state_evolution, plot_bloch_sphere, plot_state_matrix
from analyses.visualization.metric_plots import plot_metric_evolution, plot_metric_comparison, plot_metric_distribution
from analyses.visualization.circuit_diagrams import plot_circuit_diagram, export_circuit_diagram
from simulations.quantum_circuit import StandardCircuit, ScaledCircuit
from simulations.quantum_state import state_plus, state_zero

class TestVisualizationPerformance(unittest.TestCase):
    def setUp(self):
        # Create sample quantum states
        # Create density matrices for testing
        self.state_zero = Qobj(basis(2, 0))
        self.state_one = Qobj(basis(2, 1))
        self.state_plus = (self.state_zero + self.state_one).unit()
        
        # Convert all states to density matrices
        self.rho_zero = self.state_zero * self.state_zero.dag()
        self.rho_one = self.state_one * self.state_one.dag()
        self.rho_plus = self.state_plus * self.state_plus.dag()
        
        self.states = [self.rho_zero, self.rho_plus, self.rho_one]
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
            fig = plot_state_matrix(self.rho_plus, title="Performance Test")
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
            export_circuit_diagram(fig, "tests/test_circuit_diagram", format="png")
            plt.close(fig)
        
        execution_time = timeit.timeit(task, number=10)
        print(f"export_circuit_diagram executed 10 times in {execution_time:.2f} seconds.")
        self.assertLess(execution_time, 10.0, "export_circuit_diagram is too slow.")

class TestEvolutionPerformance(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        self.H0 = sigmaz()  # Simple σz Hamiltonian
        self.psi_init = state_plus()  # Initial |+⟩ state
        self.scale_factors = [0.5, 1.0, 1.5, 2.0]  # Use smaller scale factors to avoid overflow
        
    def test_standard_evolution_performance(self):
        """Benchmark standard evolution performance."""
        def task():
            circuit = StandardCircuit(self.H0, total_time=1.0, n_steps=100)
            result = circuit.evolve_closed(self.psi_init)
            return len(result.states)
        
        execution_time = timeit.timeit(task, number=10)
        print(f"Standard evolution (100 steps) executed 10 times in {execution_time:.2f} seconds.")
        self.assertLess(execution_time, 5.0, "Standard evolution is too slow.")
        
    def test_scaled_evolution_performance(self):
        """Benchmark scaled evolution performance with different scale factors."""
        results = {}
        
        def run_scaled_evolution(scale_factor):
            circuit = ScaledCircuit(self.H0, scaling_factor=scale_factor)
            result = circuit.evolve_closed(self.psi_init, n_steps=100)
            return len(result.states)
            
        for sf in self.scale_factors:
            execution_time = timeit.timeit(lambda: run_scaled_evolution(sf), number=10)
            results[sf] = execution_time
            print(f"Scaled evolution (sf={sf}, 100 steps) executed 10 times in {execution_time:.2f} seconds.")
            self.assertLess(execution_time, 5.0, f"Scaled evolution with sf={sf} is too slow.")
        
        # Verify performance scaling is reasonable
        for sf1, sf2 in zip(self.scale_factors[:-1], self.scale_factors[1:]):
            ratio = results[sf2] / results[sf1]
            # Allow up to 3x performance difference between scale factors
            # Some variation is expected due to numerical complexity
            self.assertLess(ratio, 3.0, 
                f"Performance degradation too high between sf={sf1} and sf={sf2}")
    
    def test_multiqubit_evolution_performance(self):
        """Benchmark evolution performance with increasing number of qubits."""
        n_qubits_range = [1, 2, 3, 4]
        results = {}
        
        for n in n_qubits_range:
            # Create n-qubit initial state and Hamiltonian
            initial_state = state_zero(num_qubits=n)
            # Create n-qubit Hamiltonian
            hamiltonian = 0
            for i in range(n):
                op_list_i = [qeye(2) for _ in range(n)]
                op_list_i[i] = sigmaz()
                hamiltonian += tensor(op_list_i)
            
            execution_time = timeit.timeit(
                lambda: ScaledCircuit(hamiltonian, scaling_factor=1.0).evolve_closed(initial_state, n_steps=50).states.__len__(),
                number=5
            )
            results[n] = execution_time
            print(f"{n}-qubit evolution (50 steps) executed 5 times in {execution_time:.2f} seconds.")
            
            # Time should scale approximately exponentially with number of qubits
            if n > 1:
                ratio = results[n] / results[n-1]
                self.assertLess(ratio, 4.0, 
                    f"Performance scaling too high between {n-1} and {n} qubits")
    
    def test_numerical_stability_performance(self):
        """Test numerical stability and performance with different precisions."""
        scale_factors = [0.1, 1.0, 2.0]  # Use more reasonable scale factors
        n_steps = 100
        
        def run_stability_test(circuit, n_steps):
            result = circuit.evolve_closed(self.psi_init, n_steps=n_steps)
            # Check unitarity preservation
            for state in result.states:
                tr = state.tr()
                assert np.allclose(tr, 1.0, atol=1e-10)
        
        for sf in scale_factors:
            circuit = ScaledCircuit(self.H0, scaling_factor=sf)
            execution_time = timeit.timeit(lambda: run_stability_test(circuit, n_steps), number=5)
            print(f"Stability test (sf={sf}, {n_steps} steps) executed 5 times in {execution_time:.2f} seconds.")
            self.assertLess(execution_time, 5.0, 
                f"Numerical stability checks too slow for sf={sf}")

if __name__ == '__main__':
    unittest.main()
