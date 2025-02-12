"""
tests/test_noise.py

Tests for noise models and their integration with quantum circuits.
"""

import pytest
import numpy as np
from qutip import basis, expect, sigmax, sigmay, sigmaz, qeye
from simulations.quantum_circuit import StandardCircuit, ScaledCircuit, NoiseChannel
from simulations.quantum_state import state_plus, state_zero

def test_noise_channel_initialization():
    """Test NoiseChannel class initialization with different configs"""
    # Default config (all noise disabled)
    noise = NoiseChannel()
    c_ops = noise.get_collapse_operators(2)
    assert len(c_ops) == 0
    
    # Custom config with depolarizing noise
    config = {
        'noise': {
            'depolarizing': {
                'enabled': True,
                'rate': 0.5  # Increased rate for faster decoherence
            }
        }
    }
    noise = NoiseChannel(config)
    c_ops = noise.get_collapse_operators(2)
    assert len(c_ops) == 3  # σx, σy, σz operators

def test_depolarizing_noise():
    """Test depolarizing noise effects on quantum state"""
    # Create config with only depolarizing noise
    config = {
        'noise': {
            'depolarizing': {
                'enabled': True,
                'rate': 0.5  # Increased rate for faster decoherence
            }
        }
    }
    
    # Initialize circuit with noise
    H0 = sigmaz()
    circuit = StandardCircuit(H0, total_time=5.0, n_steps=100, noise_config=config)
    
    # Evolve |+⟩ state
    psi_init = state_plus()
    result = circuit.evolve_open(psi_init)
    
    # For depolarizing noise, state should approach maximally mixed state
    # Initial |+⟩ state has <σx> = 1, final state should be mixed
    sx_expect = [expect(sigmax(), state) for state in result.states]
    assert abs(sx_expect[-1]) < 0.2  # Should decay significantly

def test_dephasing_noise():
    """Test dephasing noise effects"""
    config = {
        'noise': {
            'dephasing': {
                'enabled': True,
                'rate': 0.5  # Increased rate for faster decoherence
            }
        }
    }
    
    H0 = sigmaz()
    circuit = StandardCircuit(H0, total_time=5.0, n_steps=100, noise_config=config)
    psi_init = state_plus()
    result = circuit.evolve_open(psi_init)
    
    # For dephasing noise:
    # - x component should decay (loss of coherence)
    # - z component should be preserved (no energy exchange)
    sx_expect = [expect(sigmax(), state) for state in result.states]
    sz_expect = [expect(sigmaz(), state) for state in result.states]
    
    assert abs(sx_expect[-1]) < 0.2  # x component should decay significantly
    assert np.allclose(sz_expect[-1], sz_expect[0], atol=1e-2)  # z component preserved

def test_amplitude_damping():
    """Test amplitude damping effects"""
    config = {
        'noise': {
            'amplitude_damping': {
                'enabled': True,
                'rate': 1.0  # Increased rate for faster decay
            }
        }
    }
    
    H0 = sigmaz()
    circuit = StandardCircuit(H0, total_time=10.0, n_steps=200, noise_config=config)
    # Start in excited state |1⟩
    psi_init = basis(2, 1)
    result = circuit.evolve_open(psi_init)
    
    # System should decay to ground state |0⟩
    sz_expect = [expect(sigmaz(), state) for state in result.states]
    assert sz_expect[-1] > sz_expect[0]  # Moving towards |0⟩ state
    assert sz_expect[-1] > 0.9  # Close to ground state

def test_thermal_noise():
    """Test thermal noise effects"""
    config = {
        'noise': {
            'thermal': {
                'enabled': True,
                'nth': 0.1,
                'rate': 0.1
            }
        }
    }
    
    H0 = sigmaz()
    circuit = StandardCircuit(H0, total_time=5.0, n_steps=100, noise_config=config)
    # Start in ground state |0⟩
    psi_init = state_zero()
    result = circuit.evolve_open(psi_init)
    
    # System should approach thermal equilibrium
    sz_expect = [expect(sigmaz(), state) for state in result.states]
    # Should move away from pure |0⟩ state
    assert sz_expect[-1] < sz_expect[0]
    # But not reach completely mixed state (due to low temperature)
    assert sz_expect[-1] > 0.5

def test_combined_noise():
    """Test multiple noise channels together"""
    config = {
        'noise': {
            'depolarizing': {
                'enabled': True,
                'rate': 0.3  # Increased rate for combined effects
            },
            'dephasing': {
                'enabled': True,
                'rate': 0.3  # Increased rate for combined effects
            }
        }
    }
    
    H0 = sigmaz()
    circuit = StandardCircuit(H0, total_time=5.0, n_steps=100, noise_config=config)
    psi_init = state_plus()
    result = circuit.evolve_open(psi_init)
    
    # For combined noise, check final state is more mixed
    sx_expect = [expect(sigmax(), state) for state in result.states]
    assert abs(sx_expect[-1]) < 0.1  # Should decay more than single channel
    
    # Compare with single noise channel
    config_single = {
        'noise': {
            'depolarizing': {
                'enabled': True,
                'rate': 0.3  # Single channel rate for comparison
            }
        }
    }
    circuit_single = StandardCircuit(H0, total_time=5.0, n_steps=100, noise_config=config_single)
    result_single = circuit_single.evolve_open(psi_init)
    sx_expect_single = [expect(sigmax(), state) for state in result_single.states]
    assert abs(sx_expect[-1]) < abs(sx_expect_single[-1])  # Combined noise should cause more decay

def test_scaled_evolution_with_noise():
    """Test noise effects in scaled evolution"""
    config = {
        'noise': {
            'dephasing': {
                'enabled': True,
                'rate': 0.5  # Increased rate for clearer effects
            }
        }
    }
    
    H0 = sigmaz()
    # Compare standard vs scaled evolution with same noise
    circuit1 = StandardCircuit(H0, total_time=5.0, n_steps=100, noise_config=config)
    circuit2 = ScaledCircuit(H0, scaling_factor=2.0, total_time=5.0, n_steps=100, noise_config=config)
    
    psi_init = state_plus()
    result1 = circuit1.evolve_open(psi_init)
    result2 = circuit2.evolve_open(psi_init)
    
    # Scaled evolution should show faster oscillations but similar noise effects
    sx_expect1 = [expect(sigmax(), state) for state in result1.states]
    sx_expect2 = [expect(sigmax(), state) for state in result2.states]
    
    # Check zero crossings to measure oscillation frequency
    crossings1 = sum(1 for i in range(1, len(sx_expect1)) if sx_expect1[i-1] * sx_expect1[i] <= 0)
    crossings2 = sum(1 for i in range(1, len(sx_expect2)) if sx_expect2[i-1] * sx_expect2[i] <= 0)
    
    # Should see more oscillations with scale_factor=2
    assert crossings2 > crossings1
