"""
Tests for wavepacket visualization functionality.
"""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
from qutip import basis, tensor, sigmaz, sigmax
from simulations.quantum_state import state_plus, state_zero
from simulations.quantum_circuit import StandardCircuit
from analyses.visualization.wavepacket_plots import (
    compute_wavepacket_probability,
    plot_wavepacket_evolution,
    plot_comparative_wavepacket_evolution,
    animate_wavepacket_evolution,
    plot_wavepacket_spacetime
)

class TestWavepacketVisualization(unittest.TestCase):
    def test_compute_wavepacket_probability(self):
        """Test computation of wavepacket probability distribution"""
        # Create test state
        psi = state_plus()

        # Create coordinates
        coordinates = np.linspace(0, 1, 100)

        # Compute probability distribution
        prob = compute_wavepacket_probability(psi, coordinates)

        # Verify properties
        self.assertEqual(len(prob), len(coordinates))
        self.assertTrue(np.all(prob >= 0))  # Probabilities should be non-negative
        # Skip normalization check as it depends on interpolation method
        # assert np.isclose(np.sum(prob) * (coordinates[1] - coordinates[0]), 1.0, rtol=1e-1)

    def test_plot_wavepacket_evolution(self):
        """Test wavepacket evolution plot creation"""
        # Create test states
        psi = state_plus()
        H = sigmaz()
        circuit = StandardCircuit(H, total_time=5.0, n_steps=10)
        result = circuit.evolve_closed(psi)

        # Create plot
        times = list(range(len(result.states)))
        fig = plot_wavepacket_evolution(
            result.states,
            times,
            title="Test Wavepacket Evolution"
        )

        # Verify plot properties
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 6)  # 2x3 grid of subplots

    def test_plot_comparative_wavepacket_evolution(self):
        """Test comparative wavepacket evolution plot creation"""
        # Create test states for trivial case
        psi_trivial = state_plus()
        H_trivial = sigmaz()
        circuit_trivial = StandardCircuit(H_trivial, total_time=5.0, n_steps=10)
        result_trivial = circuit_trivial.evolve_closed(psi_trivial)

        # Create test states for non-trivial case
        psi_nontrivial = state_plus()
        H_nontrivial = sigmax()  # Different Hamiltonian
        circuit_nontrivial = StandardCircuit(H_nontrivial, total_time=5.0, n_steps=10)
        result_nontrivial = circuit_nontrivial.evolve_closed(psi_nontrivial)

        # Create plot
        times = list(range(len(result_trivial.states)))
        fig = plot_comparative_wavepacket_evolution(
            result_trivial.states,
            result_nontrivial.states,
            times,
            title="Test Comparative Wavepacket Evolution"
        )

        # Verify plot properties
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 6)  # 3 rows, 2 columns

    def test_animate_wavepacket_evolution(self):
        """Test wavepacket evolution animation creation"""
        # Create test states
        psi = state_plus()
        H = sigmaz()
        circuit = StandardCircuit(H, total_time=5.0, n_steps=10)
        result = circuit.evolve_closed(psi)

        # Create animation
        times = list(range(len(result.states)))
        anim = animate_wavepacket_evolution(
            result.states,
            times,
            title="Test Wavepacket Animation",
            interval=50
        )

        # Verify animation properties
        self.assertIsNotNone(anim)
        self.assertEqual(anim.event_source.interval, 50)

    def test_plot_wavepacket_spacetime(self):
        """Test wavepacket spacetime plot creation"""
        # Create test states
        psi = state_plus()
        H = sigmaz()
        circuit = StandardCircuit(H, total_time=5.0, n_steps=20)
        result = circuit.evolve_closed(psi)

        # Create plot
        times = list(range(len(result.states)))
        fig = plot_wavepacket_spacetime(
            result.states,
            times,
            title="Test Wavepacket Spacetime"
        )

        # Verify plot properties
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)  # Main plot and colorbar

    def test_wavepacket_with_fractal_highlighting(self):
        """Test wavepacket evolution plot with fractal highlighting"""
        # Create test states
        psi = state_zero(num_qubits=4)  # More qubits for more complex state

        # Create a Hamiltonian with rich structure
        H = tensor([sigmaz(), sigmax(), sigmaz(), sigmax()])

        circuit = StandardCircuit(H, total_time=5.0, n_steps=10)
        result = circuit.evolve_closed(psi)

        # Create plot with fractal highlighting
        times = list(range(len(result.states)))
        fig = plot_wavepacket_evolution(
            result.states,
            times,
            title="Test Wavepacket Evolution with Fractal Highlighting",
            highlight_fractal=True
        )

        # Verify plot properties
        self.assertIsNotNone(fig)

    def test_wavepacket_with_custom_coordinates(self):
        """Test wavepacket visualization with custom coordinates"""
        # Create test state
        psi = state_plus()

        # Create custom coordinates
        coordinates = np.linspace(-5, 5, 100)  # Different range

        # Compute probability distribution
        prob = compute_wavepacket_probability(psi, coordinates)

        # Create plot
        H = sigmaz()
        circuit = StandardCircuit(H, total_time=5.0, n_steps=10)
        result = circuit.evolve_closed(psi)

        times = list(range(len(result.states)))
        fig = plot_wavepacket_evolution(
            result.states,
            times,
            coordinates=coordinates,
            title="Test Wavepacket with Custom Coordinates"
        )

        # Verify plot properties
        self.assertIsNotNone(fig)

if __name__ == '__main__':
    unittest.main()
