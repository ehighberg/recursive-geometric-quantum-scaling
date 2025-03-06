"""
Tests for dynamical evolution integration with the UI.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import streamlit as st
from unittest.mock import patch, MagicMock
from qutip import basis, tensor, sigmaz, sigmax, qeye
from simulations.quantum_state import state_plus, state_zero, state_ghz
from simulations.quantum_circuit import StandardCircuit
from simulations.scripts.evolve_state import run_state_evolution
from analyses.visualization.wavepacket_plots import (
    plot_wavepacket_evolution,
    plot_wavepacket_spacetime,
    animate_wavepacket_evolution
)
from analyses.entanglement_dynamics import (
    plot_entanglement_entropy_vs_time,
    plot_entanglement_spectrum,
    plot_entanglement_growth_rate
)

def test_wavepacket_visualization_integration():
    """Test integration of wavepacket visualization with simulation results"""
    # Run a simulation
    result = run_state_evolution(
        num_qubits=2,
        state_label="plus",
        n_steps=10,
        scaling_factor=1.0
    )
    
    # Verify result has required attributes
    assert hasattr(result, 'states')
    assert hasattr(result, 'times')
    assert len(result.states) > 0
    assert len(result.times) > 0
    
    # Create coordinates for wavepacket visualization
    if result.states[0].isket:
        dim = len(result.states[0].full().flatten())
    else:
        dim = result.states[0].shape[0]
    coordinates = np.linspace(0, 1, dim)
    
    # Test wavepacket evolution plot
    fig_wavepacket = plot_wavepacket_evolution(
        result.states,
        result.times,
        coordinates=coordinates,
        title="Test Wavepacket Evolution"
    )
    assert fig_wavepacket is not None
    
    # Test wavepacket spacetime plot
    fig_spacetime = plot_wavepacket_spacetime(
        result.states,
        result.times,
        coordinates=coordinates,
        title="Test Wavepacket Spacetime"
    )
    assert fig_spacetime is not None
    
    # Test wavepacket animation
    anim = animate_wavepacket_evolution(
        result.states,
        result.times,
        coordinates=coordinates,
        title="Test Wavepacket Animation"
    )
    assert anim is not None
    
    # Save outputs for verification
    fig_wavepacket.savefig('test_wavepacket_integration.png')
    fig_spacetime.savefig('test_spacetime_integration.png')
    anim.save('test_wavepacket_anim_integration.gif', writer='pillow')

def test_entanglement_dynamics_integration():
    """Test integration of entanglement dynamics with simulation results"""
    # Run a simulation with multiple qubits
    result = run_state_evolution(
        num_qubits=3,  # Use 3 qubits for more interesting entanglement
        state_label="ghz",  # GHZ state is highly entangled
        n_steps=10,
        scaling_factor=1.0
    )
    
    # Verify result has required attributes
    assert hasattr(result, 'states')
    assert hasattr(result, 'times')
    assert len(result.states) > 0
    assert len(result.times) > 0
    
    # Test entanglement entropy plot
    fig_entropy = plot_entanglement_entropy_vs_time(
        result.states,
        result.times,
        title="Test Entanglement Entropy"
    )
    assert fig_entropy is not None
    
    # Test entanglement growth rate plot
    fig_growth = plot_entanglement_growth_rate(
        result.states,
        result.times,
        title="Test Entanglement Growth"
    )
    assert fig_growth is not None
    
    # Test entanglement spectrum plot
    fig_spectrum = plot_entanglement_spectrum(
        result.states,
        result.times,
        title="Test Entanglement Spectrum"
    )
    assert fig_spectrum is not None
    
    # Save outputs for verification
    fig_entropy.savefig('test_entropy_integration.png')
    fig_growth.savefig('test_growth_integration.png')
    fig_spectrum.savefig('test_spectrum_integration.png')

import pytest

@pytest.mark.skip(reason="UI integration test requires special setup")
@patch('streamlit.tabs')
def test_ui_tabs_integration(mock_tabs):
    """Test integration of new tabs with Streamlit UI"""
    # This test is skipped because it requires special setup
    # The UI integration is tested manually
    pass

def test_full_pipeline_with_dynamical_evolution():
    """Test the full pipeline including dynamical evolution analysis"""
    # Run a simulation
    result = run_state_evolution(
        num_qubits=2,
        state_label="ghz",
        n_steps=20,
        scaling_factor=1.0
    )
    
    # Verify result has required attributes
    assert hasattr(result, 'states')
    assert hasattr(result, 'times')
    assert len(result.states) > 0
    assert len(result.times) > 0
    
    # Create coordinates for wavepacket visualization
    if result.states[0].isket:
        dim = len(result.states[0].full().flatten())
    else:
        dim = result.states[0].shape[0]
    coordinates = np.linspace(0, 1, dim)
    
    # Test wavepacket evolution
    fig_wavepacket = plot_wavepacket_evolution(
        result.states,
        result.times,
        coordinates=coordinates,
        title="Full Pipeline Wavepacket Evolution"
    )
    
    # Test entanglement entropy
    fig_entropy = plot_entanglement_entropy_vs_time(
        result.states,
        result.times,
        title="Full Pipeline Entanglement Entropy"
    )
    
    # Verify outputs
    assert fig_wavepacket is not None
    assert fig_entropy is not None
    
    # Save outputs for verification
    fig_wavepacket.savefig('test_full_pipeline_wavepacket.png')
    fig_entropy.savefig('test_full_pipeline_entropy.png')
