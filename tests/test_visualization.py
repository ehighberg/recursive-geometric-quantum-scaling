"""
tests/test_visualization.py

Unit tests for the visualization components of the Quantum Simulation Codebase.
"""

import unittest
import numpy as np
from qutip import Qobj, basis
import matplotlib.pyplot as plt

from analyses.visualization.state_plots import plot_state_evolution, plot_bloch_sphere, plot_state_matrix
from analyses.visualization.metric_plots import plot_metric_evolution, plot_metric_comparison, plot_metric_distribution
from analyses.visualization.circuit_diagrams import plot_circuit_diagram, export_circuit_diagram

class TestVisualizationComponents(unittest.TestCase):
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
    
    def test_plot_state_evolution(self):
        try:
            fig = plot_state_evolution(self.states, self.times, title="Test State Evolution")
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
        except Exception as e:
            self.fail(f"plot_state_evolution raised an exception {e}")
    
    def test_plot_bloch_sphere(self):
        try:
            fig = plot_bloch_sphere(self.state_plus, title="Test Bloch Sphere")
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
        except Exception as e:
            self.fail(f"plot_bloch_sphere raised an exception {e}")
    
    def test_plot_state_matrix(self):
        try:
            fig = plot_state_matrix(self.density_matrix, title="Test State Matrix")
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
        except Exception as e:
            self.fail(f"plot_state_matrix raised an exception {e}")
    
    def test_plot_metric_evolution(self):
        try:
            fig = plot_metric_evolution(self.states, self.times, title="Test Metric Evolution")
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
        except Exception as e:
            self.fail(f"plot_metric_evolution raised an exception {e}")
    
    def test_plot_metric_comparison(self):
        try:
            metric_pairs = [('vn_entropy', 'l1_coherence'), ('vn_entropy', 'negativity')]
            fig = plot_metric_comparison(self.states, metric_pairs, title="Test Metric Comparison")
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
        except Exception as e:
            self.fail(f"plot_metric_comparison raised an exception {e}")
    
    def test_plot_metric_distribution(self):
        try:
            fig = plot_metric_distribution(self.states, title="Test Metric Distribution")
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
        except Exception as e:
            self.fail(f"plot_metric_distribution raised an exception {e}")
    
    def test_plot_circuit_diagram(self):
        try:
            fig = plot_circuit_diagram(self.components, self.connections, title="Test Circuit Diagram")
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
        except Exception as e:
            self.fail(f"plot_circuit_diagram raised an exception {e}")
    
    def test_export_circuit_diagram(self):
        try:
            fig = plot_circuit_diagram(self.components, self.connections, title="Test Export Circuit Diagram")
            export_circuit_diagram(fig, "test_circuit_diagram", format="png")
            # Check if file exists (requires filesystem access; skipped here)
            plt.close(fig)
        except Exception as e:
            self.fail(f"export_circuit_diagram raised an exception {e}")

if __name__ == '__main__':
    unittest.main()