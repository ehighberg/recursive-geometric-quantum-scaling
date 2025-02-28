"""
Tests for entanglement dynamics analysis functionality.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
from qutip import basis, tensor, sigmaz, sigmax, qeye
from simulations.quantum_state import state_plus, state_zero, state_ghz
from simulations.quantum_circuit import StandardCircuit
from analyses.entanglement_dynamics import (
    compute_entanglement_entropy_vs_time,
    compute_mutual_information_vs_time,
    plot_entanglement_entropy_vs_time,
    analyze_entanglement_scaling,
    compare_boundary_conditions,
    plot_entanglement_spectrum,
    plot_entanglement_growth_rate
)

def test_compute_entanglement_entropy_vs_time():
    """Test computation of entanglement entropy over time"""
    # Create test states for a 2-qubit system
    psi_init = tensor(basis(2, 0), basis(2, 0))  # |00⟩
    
    # Create Hamiltonian that entangles qubits
    H = tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmax()) + 0.5 * tensor(sigmax(), sigmax())
    
    # Evolve state
    circuit = StandardCircuit(H, total_time=5.0, n_steps=10)
    result = circuit.evolve_closed(psi_init)
    
    # Compute entanglement entropy
    entropies = compute_entanglement_entropy_vs_time(result.states)
    
    # Verify properties
    assert len(entropies) == len(result.states)
    assert all(e >= 0 for e in entropies)  # Entropies should be non-negative
    assert all(e <= 1.0 for e in entropies)  # Max entropy for 2-qubit system is 1.0

def test_compute_mutual_information_vs_time():
    """Test computation of mutual information over time"""
    # Create test states for a 2-qubit system
    psi_init = tensor(basis(2, 0), basis(2, 0))  # |00⟩
    
    # Create Hamiltonian that entangles qubits
    H = tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmax()) + 0.5 * tensor(sigmax(), sigmax())
    
    # Evolve state
    circuit = StandardCircuit(H, total_time=5.0, n_steps=10)
    result = circuit.evolve_closed(psi_init)
    
    # Compute mutual information
    mutual_info = compute_mutual_information_vs_time(result.states, 0, 1)
    
    # Verify properties
    assert len(mutual_info) == len(result.states)
    assert all(mi >= 0 for mi in mutual_info)  # Mutual information should be non-negative

def test_plot_entanglement_entropy_vs_time():
    """Test entanglement entropy plot creation"""
    # Create test states for a 2-qubit system
    psi_init = tensor(basis(2, 0), basis(2, 0))  # |00⟩
    
    # Create Hamiltonian that entangles qubits
    H = tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmax()) + 0.5 * tensor(sigmax(), sigmax())
    
    # Evolve state
    circuit = StandardCircuit(H, total_time=5.0, n_steps=10)
    result = circuit.evolve_closed(psi_init)
    
    # Create plot
    times = list(range(len(result.states)))
    fig = plot_entanglement_entropy_vs_time(
        result.states,
        times,
        title="Test Entanglement Entropy Evolution"
    )
    
    # Verify plot properties
    assert fig is not None
    assert len(fig.axes) == 1  # Single plot
    
    # Save figure to verify
    fig.savefig('test_entanglement_entropy.png')

def test_analyze_entanglement_scaling():
    """Test entanglement scaling analysis"""
    # Create test states for different system sizes
    states_dict = {}
    
    # 2-qubit system
    psi_2 = tensor(basis(2, 0), basis(2, 0))  # |00⟩
    H_2 = tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmax()) + 0.5 * tensor(sigmax(), sigmax())
    circuit_2 = StandardCircuit(H_2, total_time=5.0, n_steps=10)
    result_2 = circuit_2.evolve_closed(psi_2)
    states_dict[2] = result_2.states
    
    # 3-qubit system
    psi_3 = tensor(basis(2, 0), basis(2, 0), basis(2, 0))  # |000⟩
    H_3 = tensor(sigmaz(), qeye(2), qeye(2)) + tensor(qeye(2), sigmax(), qeye(2)) + tensor(qeye(2), qeye(2), sigmax())
    circuit_3 = StandardCircuit(H_3, total_time=5.0, n_steps=10)
    result_3 = circuit_3.evolve_closed(psi_3)
    states_dict[3] = result_3.states
    
    # 4-qubit system
    psi_4 = tensor(basis(2, 0), basis(2, 0), basis(2, 0), basis(2, 0))  # |0000⟩
    H_4 = (tensor(sigmaz(), qeye(2), qeye(2), qeye(2)) + 
           tensor(qeye(2), sigmax(), qeye(2), qeye(2)) + 
           tensor(qeye(2), qeye(2), sigmax(), qeye(2)) +
           tensor(qeye(2), qeye(2), qeye(2), sigmax()))
    circuit_4 = StandardCircuit(H_4, total_time=5.0, n_steps=10)
    result_4 = circuit_4.evolve_closed(psi_4)
    states_dict[4] = result_4.states
    
    # Create plot
    times = list(range(10))
    fig = analyze_entanglement_scaling(
        states_dict,
        times,
        title="Test Entanglement Scaling"
    )
    
    # Verify plot properties
    assert fig is not None
    assert len(fig.axes) == 2  # Two subplots
    
    # Save figure to verify
    fig.savefig('test_entanglement_scaling.png')

def test_compare_boundary_conditions():
    """Test comparison of boundary conditions"""
    # Create test states for periodic boundary conditions
    psi_pbc = tensor(basis(2, 0), basis(2, 0))  # |00⟩
    H_pbc = tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmax()) + 0.5 * tensor(sigmax(), sigmax())
    circuit_pbc = StandardCircuit(H_pbc, total_time=5.0, n_steps=10)
    result_pbc = circuit_pbc.evolve_closed(psi_pbc)
    
    # Create test states for open boundary conditions
    psi_obc = tensor(basis(2, 0), basis(2, 0))  # |00⟩
    H_obc = tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmax())  # No interaction term
    circuit_obc = StandardCircuit(H_obc, total_time=5.0, n_steps=10)
    result_obc = circuit_obc.evolve_closed(psi_obc)
    
    # Create plot
    times = list(range(len(result_pbc.states)))
    fig = compare_boundary_conditions(
        result_pbc.states,
        result_obc.states,
        times,
        title="Test Boundary Conditions Comparison"
    )
    
    # Verify plot properties
    assert fig is not None
    assert len(fig.axes) == 2  # Two subplots
    
    # Save figure to verify
    fig.savefig('test_boundary_conditions.png')

def test_plot_entanglement_spectrum():
    """Test entanglement spectrum plot creation"""
    # Create test states for a 4-qubit system (to get interesting spectrum)
    psi_init = state_ghz(num_qubits=4)  # GHZ state
    
    # Create Hamiltonian
    H = (tensor(sigmaz(), qeye(2), qeye(2), qeye(2)) + 
         tensor(qeye(2), sigmax(), qeye(2), qeye(2)) + 
         tensor(qeye(2), qeye(2), sigmax(), qeye(2)) +
         tensor(qeye(2), qeye(2), qeye(2), sigmax()))
    
    # Evolve state
    circuit = StandardCircuit(H, total_time=5.0, n_steps=10)
    result = circuit.evolve_closed(psi_init)
    
    # Create plot
    times = list(range(len(result.states)))
    fig = plot_entanglement_spectrum(
        result.states,
        times,
        title="Test Entanglement Spectrum"
    )
    
    # Verify plot properties
    assert fig is not None
    assert len(fig.axes) == 6  # 2x3 grid of subplots
    
    # Save figure to verify
    fig.savefig('test_entanglement_spectrum.png')

def test_plot_entanglement_growth_rate():
    """Test entanglement growth rate plot creation"""
    # Create test states for a 2-qubit system
    psi_init = tensor(basis(2, 0), basis(2, 0))  # |00⟩
    
    # Create Hamiltonian that entangles qubits
    H = tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmax()) + 0.5 * tensor(sigmax(), sigmax())
    
    # Evolve state
    circuit = StandardCircuit(H, total_time=5.0, n_steps=20)  # More steps for better derivative
    result = circuit.evolve_closed(psi_init)
    
    # Create plot
    times = list(range(len(result.states)))
    fig = plot_entanglement_growth_rate(
        result.states,
        times,
        title="Test Entanglement Growth Rate"
    )
    
    # Verify plot properties
    assert fig is not None
    assert len(fig.axes) == 2  # Main plot and twin y-axis
    
    # Save figure to verify
    fig.savefig('test_entanglement_growth_rate.png')
