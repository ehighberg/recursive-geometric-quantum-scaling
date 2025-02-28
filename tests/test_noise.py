"""
Tests for noise models and their effects on quantum state evolution.
"""

import numpy as np
from qutip import basis, sigmax, sigmaz
from simulations.scripts.evolve_state import simulate_evolution

def test_noise_types():
    """Test different types of noise configurations"""
    # Initial state |0⟩
    psi0 = basis(2, 0)
    H = 2 * np.pi * 0.1 * sigmax()
    times = np.linspace(0.0, 10.0, 100)
    
    # Test relaxation noise
    noise_config_relaxation = {'relaxation': 0.1, 'dephasing': 0.0, 'thermal': 0.0, 'measurement': 0.0}
    result_relaxation = simulate_evolution(H, psi0, times, noise_config_relaxation)
    
    # Test dephasing noise
    noise_config_dephasing = {'relaxation': 0.0, 'dephasing': 0.1, 'thermal': 0.0, 'measurement': 0.0}
    result_dephasing = simulate_evolution(H, psi0, times, noise_config_dephasing)
    
    # Test thermal noise
    noise_config_thermal = {'relaxation': 0.0, 'dephasing': 0.0, 'thermal': 0.1, 'measurement': 0.0}
    result_thermal = simulate_evolution(H, psi0, times, noise_config_thermal)
    
    # Test measurement noise
    noise_config_measurement = {'relaxation': 0.0, 'dephasing': 0.0, 'thermal': 0.0, 'measurement': 0.1}
    result_measurement = simulate_evolution(H, psi0, times, noise_config_measurement)
    
    # All results should have the same number of states
    assert len(result_relaxation.states) == len(times)
    assert len(result_dephasing.states) == len(times)
    assert len(result_thermal.states) == len(times)
    assert len(result_measurement.states) == len(times)
    
    # All final states should be mixed (not pure) due to noise
    assert not result_relaxation.states[-1].isket
    assert not result_dephasing.states[-1].isket
    assert not result_thermal.states[-1].isket
    assert not result_measurement.states[-1].isket

def test_noise_strength():
    """Test that stronger noise leads to faster decoherence"""
    psi0 = basis(2, 0)
    H = 2 * np.pi * 0.1 * sigmax()
    times = np.linspace(0.0, 10.0, 100)
    
    # Weak noise
    weak_noise = {'relaxation': 0.01, 'dephasing': 0.01, 'thermal': 0.0, 'measurement': 0.0}
    result_weak = simulate_evolution(H, psi0, times, weak_noise)
    
    # Strong noise
    strong_noise = {'relaxation': 0.1, 'dephasing': 0.1, 'thermal': 0.0, 'measurement': 0.0}
    result_strong = simulate_evolution(H, psi0, times, strong_noise)
    
    # Calculate purities at final time
    rho_weak = result_weak.states[-1]
    rho_strong = result_strong.states[-1]
    
    purity_weak = (rho_weak * rho_weak).tr().real
    purity_strong = (rho_strong * rho_strong).tr().real
    
    # Strong noise should lead to lower purity
    assert purity_strong < purity_weak

def test_noise_free_evolution():
    """Test that evolution without noise remains pure"""
    psi0 = basis(2, 0)
    H = 2 * np.pi * 0.1 * sigmax()
    times = np.linspace(0.0, 10.0, 100)
    
    # No noise
    result = simulate_evolution(H, psi0, times, None)
    
    # All states should remain pure
    for state in result.states:
        assert state.isket
        # Pure states have Tr(ρ2) = 1
        rho = state * state.dag()
        purity = (rho * rho).tr().real
        assert np.abs(purity - 1.0) < 1e-10

def test_combined_noise_effects():
    """Test combined effects of different noise types"""
    psi0 = basis(2, 0)
    H = 2 * np.pi * 0.1 * sigmax()
    times = np.linspace(0.0, 10.0, 100)
    
    # Single noise type
    single_noise = {'relaxation': 0.1, 'dephasing': 0.0, 'thermal': 0.0, 'measurement': 0.0}
    result_single = simulate_evolution(H, psi0, times, single_noise)
    
    # Combined noise types
    combined_noise = {'relaxation': 0.1, 'dephasing': 0.1, 'thermal': 0.1, 'measurement': 0.1}
    result_combined = simulate_evolution(H, psi0, times, combined_noise)
    
    # Calculate purities at final time
    rho_single = result_single.states[-1]
    rho_combined = result_combined.states[-1]
    
    purity_single = (rho_single * rho_single).tr().real
    purity_combined = (rho_combined * rho_combined).tr().real
    
    # Combined noise should lead to lower purity
    assert purity_combined < purity_single

def test_thermal_noise_equilibrium():
    """Test that thermal noise drives system towards thermal equilibrium"""
    psi0 = basis(2, 0)  # Start in ground state
    H = sigmaz()  # Energy splitting between |0⟩ and |1⟩
    times = np.linspace(0.0, 50.0, 500)  # Longer evolution to reach equilibrium
    
    # Strong thermal noise
    thermal_noise = {'relaxation': 0.0, 'dephasing': 0.0, 'thermal': 1.0, 'measurement': 0.0}
    result = simulate_evolution(H, psi0, times, thermal_noise)
    
    # Get final state populations
    final_state = result.states[-1]
    populations = final_state.diag()
    
    # In high temperature limit, populations should be approximately equal
    assert np.abs(populations[0] - populations[1]) < 0.1
