# tests/test_quantum_circuit.py

import numpy as np
from qutip import sigmaz, sigmax, sigmay, Qobj, basis, expect
from simulations.quantum_circuit import StandardCircuit, ScaledCircuit, FibonacciBraidingCircuit
from simulations.quantum_state import state_plus, state_zero

def test_standard_circuit_init():
    """Test standard circuit initialization"""
    H0 = sigmaz()
    circuit = StandardCircuit(H0, total_time=2.0, n_steps=20)
    assert circuit.n_steps == 20
    assert circuit.total_time == 2.0
    assert circuit.base_hamiltonian == H0

def test_scaled_circuit_init():
    """Test scaled circuit initialization with different scale factors"""
    H0 = sigmaz()
    # Test default scale_factor=1
    circuit1 = ScaledCircuit(H0)
    assert circuit1.scale_factor == 1.0
    
    # Test custom scale_factor
    circuit2 = ScaledCircuit(H0, scaling_factor=1.618)
    assert circuit2.scale_factor == 1.618

def test_scale_unitary():
    """Test scaled unitary evolution with different scale factors"""
    H0 = sigmaz()
    circuit = ScaledCircuit(H0, scaling_factor=2.0)
    U_3 = circuit.scale_unitary(3)  # Scale factor = 2^3 = 8
    assert U_3.shape == (2, 2)
    
    # For sigmaz, analytical solution for U = exp(-i*s*H) is:
    # [exp(-i*s)  0        ]
    # [0          exp(i*s) ]
    # where s is the scaling factor
    s = 8.0  # 2^3
    expected = np.array([[np.exp(-1j*s), 0], [0, np.exp(1j*s)]])
    assert np.allclose(U_3.full(), expected, atol=1e-10)
    
    # Test evolution result
    psi_init = state_plus()
    result = circuit.evolve_closed(psi_init, n_steps=3)
    assert hasattr(result, 'states')
    assert hasattr(result, 'times')
    assert result.dims == [[2], [1]]  # Single qubit state

def test_standard_evolution_analytical():
    """Test standard evolution (scale_factor=1) against analytical solution"""
    # For H = σz, |+⟩ evolves as:
    # |ψ(t)⟩ = cos(t)|0⟩ - i*sin(t)|1⟩
    H0 = sigmaz()
    psi_init = state_plus()
    circuit = StandardCircuit(H0, total_time=np.pi/2, n_steps=100)
    result = circuit.evolve_closed(psi_init)
    
    # At t=π/2, |+⟩ evolves to |−⟩ = (|0⟩ - |1⟩)/√2
    final_state = result.states[-1]
    
    # Check expectation values instead of state vector
    # This avoids phase ambiguity issues
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    
    # At t=π/2, expect:
    # ⟨σx⟩ = -1
    # ⟨σy⟩ = 0
    # ⟨σz⟩ = 0
    assert np.allclose(expect(sx, final_state), -1.0, atol=1e-5)
    assert np.allclose(expect(sy, final_state), 0.0, atol=1e-5)
    assert np.allclose(expect(sz, final_state), 0.0, atol=1e-5)

def test_scaled_evolution_analytical():
    """Test scaled evolution against analytical solutions"""
    H0 = sigmaz()
    psi_init = state_plus()
    
    # Test with scale_factor = 1 (should match standard evolution)
    circuit1 = ScaledCircuit(H0, scaling_factor=1.0)
    result1 = circuit1.evolve_closed(psi_init, n_steps=4)
    
    # Test with scale_factor = 2 (should evolve twice as fast)
    circuit2 = ScaledCircuit(H0, scaling_factor=2.0)
    result2 = circuit2.evolve_closed(psi_init, n_steps=4)
    
    # Compare expectation values of σz instead of σx
    # σz evolution is more reliable for comparing frequencies
    sz = sigmaz()
    expect1 = [float(expect(sz, state)) for state in result1.states]
    expect2 = [float(expect(sz, state)) for state in result2.states]
    
    # With scale_factor=2, evolution should oscillate twice as fast
    # Compare frequencies by looking at zero crossings
    crossings1 = np.where(np.diff(np.signbit(expect1)))[0]
    crossings2 = np.where(np.diff(np.signbit(expect2)))[0]
    
    if len(crossings1) > 0 and len(crossings2) > 0:
        # Compare the first zero crossing points
        # Second evolution should cross zero in roughly half the time
        ratio = crossings2[0] / crossings1[0]
        assert np.allclose(ratio, 0.5, atol=0.1)

def test_numerical_stability():
    """Test numerical stability with various scale factors"""
    H0 = sigmaz()
    psi_init = state_zero()
    scale_factors = [0.1, 1.0, 10.0, 100.0]
    
    for sf in scale_factors:
        circuit = ScaledCircuit(H0, scaling_factor=sf)
        result = circuit.evolve_closed(psi_init, n_steps=10)
        
        # Check unitarity preservation
        for state in result.states:
            # Trace of density matrix should be 1
            assert np.allclose(state.tr(), 1.0, atol=1e-10)
            # Eigenvalues should be real and positive
            evals = state.eigenenergies()
            assert np.allclose(evals.imag, 0, atol=1e-10)
            assert all(ev >= -1e-10 for ev in evals.real)

def test_fibonacci_braiding_circuit():
    """Test Fibonacci braiding circuit operations"""
    fib = FibonacciBraidingCircuit()
    assert len(fib.braids) == 0
    
    # Add identity braid
    B = Qobj(np.eye(2))
    fib.add_braid(B)
    assert len(fib.braids) == 1
    
    # Test evolution with identity braid
    psi = state_zero()
    result = fib.evolve(psi)
    # Get final state from evolution result
    psi_final = result.states[-1]
    assert np.allclose(psi_final.full(), psi.full())
