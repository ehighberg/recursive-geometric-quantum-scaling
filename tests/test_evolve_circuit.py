"""
Tests for circuit evolution implementations.
"""

import numpy as np
from qutip import sigmaz, tensor, qeye
from simulations.scripts.evolve_circuit import (
    run_standard_twoqubit_circuit,
    run_phi_scaled_twoqubit_circuit,
    run_fibonacci_braiding_circuit,
    analyze_circuit_noise_effects
)

def test_standard_twoqubit_circuit():
    """Test standard two-qubit circuit evolution."""
    # Test without noise
    result = run_standard_twoqubit_circuit()
    assert len(result.states) == 51  # n_steps + 1
    assert result.states[0].dims == [[2, 2], [1]]  # 2-qubit state
    
    # Test with noise
    c_ops = [np.sqrt(0.01) * tensor(sigmaz(), qeye(2))]  # Dephasing noise on first qubit
    result_noisy = run_standard_twoqubit_circuit(noise_config={'c_ops': c_ops})
    assert len(result_noisy.states) == 51
    assert not result_noisy.states[-1].isket  # Should be mixed state due to noise

def test_phi_scaled_twoqubit_circuit():
    """Test phi-scaled two-qubit circuit evolution."""
    # Test without noise
    result = run_phi_scaled_twoqubit_circuit(scaling_factor=1.0)
    assert len(result.states) == 5  # Default n_steps for phi-scaled
    assert result.states[0].dims == [[2, 2], [1]]  # 2-qubit state
    
    # Test with different scaling factor
    result_scaled = run_phi_scaled_twoqubit_circuit(scaling_factor=0.5)
    assert len(result_scaled.states) == 5
    
    # Test with noise
    c_ops = [np.sqrt(0.01) * tensor(sigmaz(), qeye(2))]  # Dephasing noise on first qubit
    result_noisy = run_phi_scaled_twoqubit_circuit(
        scaling_factor=1.0,
        noise_config={'c_ops': c_ops}
    )
    assert len(result_noisy.states) == 5
    assert not result_noisy.states[-1].isket  # Should be mixed state due to noise

def test_fibonacci_braiding_circuit():
    """Test Fibonacci anyon braiding circuit."""
    result = run_fibonacci_braiding_circuit()
    # Should return an EvolutionResult with states
    assert hasattr(result, 'states')
    final_state = result.states[-1]
    assert final_state.dims == [[2], [1]]  # 2D subspace for Fibonacci anyons

def test_circuit_noise_analysis():
    """Test noise analysis for circuits."""
    # Test standard circuit noise analysis
    results_std = analyze_circuit_noise_effects(
        circuit_type="standard",
        noise_rates=[0.0, 0.1]
    )
    
    assert len(results_std['fidelities']) == 2
    assert len(results_std['purities']) == 2
    assert results_std['purities'][0] > results_std['purities'][1]  # Higher purity without noise
    
    # Test phi-scaled circuit noise analysis
    results_phi = analyze_circuit_noise_effects(
        circuit_type="phi_scaled",
        noise_rates=[0.0, 0.1]
    )
    
    assert len(results_phi['fidelities']) == 2
    assert len(results_phi['purities']) == 2
    assert results_phi['purities'][0] > results_phi['purities'][1]  # Higher purity without noise

def test_circuit_gate_operations():
    """Test that circuit gates perform expected operations."""
    # Test standard circuit with CNOT + RX
    result_std = run_standard_twoqubit_circuit()
    initial_state = result_std.states[0]
    final_state = result_std.states[-1]
    
    # Verify state changed
    fidelity = abs((initial_state.dag() * final_state).tr())
    assert fidelity < 0.99  # State should have evolved
    
    # Test phi-scaled circuit
    result_phi = run_phi_scaled_twoqubit_circuit(scaling_factor=1.0)
    initial_state = result_phi.states[0]
    final_state = result_phi.states[-1]
    
    # Verify state changed with scaling
    fidelity = abs((initial_state.dag() * final_state).tr())
    assert fidelity < 0.99  # State should have evolved

def test_noise_effects_on_entanglement():
    """Test how noise affects entanglement in circuits."""
    # Run circuit without noise
    result_clean = run_standard_twoqubit_circuit()
    
    # Run circuit with noise
    c_ops = [np.sqrt(0.1) * tensor(sigmaz(), qeye(2))]  # Dephasing noise on first qubit
    result_noisy = run_standard_twoqubit_circuit(noise_config={'c_ops': c_ops})
    
    # Get final states
    rho_clean = result_clean.states[-1]
    if rho_clean.isket:
        rho_clean = rho_clean * rho_clean.dag()
    
    rho_noisy = result_noisy.states[-1]
    
    # Calculate purities
    purity_clean = (rho_clean * rho_clean).tr().real
    purity_noisy = (rho_noisy * rho_noisy).tr().real
    
    # Noisy evolution should reduce purity
    assert purity_noisy < purity_clean
