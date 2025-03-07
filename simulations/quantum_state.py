# simulations/quantum_state.py
"""
This module provides quantum state initialization functions.

In addition to standard quantum computing states (zero, one, plus, GHZ, W),
this module also provides specialized states for fractal and phi-resonant
quantum phenomena.
"""

import numpy as np
from qutip import Qobj, basis, tensor, sigmax, qeye
from constants import PHI

def state_zero(num_qubits=1):
    """All-|0> state."""
    if num_qubits == 1:
        return basis(2, 0)
    return tensor(*[basis(2, 0) for _ in range(num_qubits)])

def state_one(num_qubits=1):
    """All-|1> state."""
    if num_qubits == 1:
        return basis(2, 1)
    return tensor(*[basis(2, 1) for _ in range(num_qubits)])

def state_plus(num_qubits=1):
    """All-|+> = (|0>+|1>)/√2 (tensored for multiple qubits)."""
    single_plus = (basis(2, 0) + basis(2, 1)).unit()
    if num_qubits == 1:
        return single_plus
    return tensor(*[single_plus for _ in range(num_qubits)])

def state_ghz(num_qubits=3):
    """
    GHZ = (|000...> + |111...>)/√2, for >=2 qubits.
    """
    if num_qubits < 2:
        raise ValueError("GHZ requires >=2 qubits.")
    zero_ket = tensor(*[basis(2, 0) for _ in range(num_qubits)])
    one_ket = tensor(*[basis(2, 1) for _ in range(num_qubits)])
    return (zero_ket + one_ket).unit()

def state_w(num_qubits=3):
    """
    W = (|100..> + |010..> + ... )/√(num_qubits).
    """
    if num_qubits < 2:
        raise ValueError("W requires >=2 qubits.")
    if num_qubits == 0:
        raise ValueError("W requires > 0 qubits.")
    states = []
    for i in range(num_qubits):
        ket_list = []
        for j in range(num_qubits):
            ket_list.append(basis(2, 1) if j == i else basis(2, 0))
        states.append(tensor(*ket_list))
    psi = sum(states)
    if isinstance(psi, Qobj):
        return psi.unit()
    else:
        return psi

def positivity_projection(rho):
    """
    Clamps negative eigenvalues in a density matrix and re-normalizes.
    """
    evals, evecs = rho.eigenstates()
    evals_clipped = [max(ev, 0.0) for ev in evals]
    rho_fixed = 0 * rho
    for val, vec in zip(evals_clipped, evecs):
        rho_fixed += val * vec * vec.dag()

    if rho_fixed.norm() < 1e-15:
        return rho_fixed
    tr_val = rho_fixed.tr()
    if tr_val > 1e-15:
        rho_fixed /= tr_val
    return rho_fixed

# Toy states for fibonacci anyons:
#TODO: replace with actual anyon states
def fib_anyon_state_2d():
    """
    2D subspace for 3 anyons => dimension=2. 
    We'll do an equal superposition for demonstration.
    """
    vec = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    return Qobj(vec)

def fib_anyon_state_3d():
    """
    3D subspace => dimension=3, for 4 anyons total charge τ.
    We'll do a basis vector [1,0,0].
    """
    vec = np.array([1.0, 0.0, 0.0], dtype=complex)
    return Qobj(vec)


def state_fractal(num_qubits=8, depth=3, phi_param=None):
    """
    Create a state with fractal-like quantum correlations.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits
    depth : int
        Recursion depth
    phi_param : float or None
        If provided, incorporates phi in the state construction
    
    Returns:
    --------
    Qobj
        Quantum state with fractal properties
    """
    if phi_param is None:
        phi = PHI
    else:
        phi = phi_param
        
    # Start with all-zero state
    state = state_zero(num_qubits)
    
    # Add fractal structure via recursive patterns
    for d in range(depth):
        scale = phi**(d+1)
        
        # For each recursion level, create patterns at different scales
        for i in range(num_qubits):
            # Skip indices based on Fibonacci-like pattern
            if i % max(1, int(scale)) == 0:
                # Apply controlled operations between qubits at Fibonacci-separated indices
                controller = i
                target_idx = int((i + scale) % num_qubits)
                
                # Create controlled gate
                # Create identity and projection operators
                id2 = qeye(2)
                proj0 = Qobj([[1, 0], [0, 0]])
                proj1 = Qobj([[0, 0], [0, 1]])
                
                # Create controlled operation
                if num_qubits == 1:
                    control_op = sigmax()  # No control for single qubit
                else:
                    # For multiple qubits, create controlled operation
                    op_list = []
                    for j in range(num_qubits):
                        if j == controller:
                            op_list.append(proj0)
                        elif j == target_idx:
                            op_list.append(id2)
                        else:
                            op_list.append(id2)
                    
                    # First term: |0⟩⟨0| ⊗ I
                    term1 = tensor(op_list)
                    
                    # Second term: |1⟩⟨1| ⊗ U
                    op_list[controller] = proj1
                    op_list[target_idx] = sigmax()
                    term2 = tensor(op_list)
                    
                    control_op = term1 + term2
                
                state = control_op * state
    
    return state.unit()  # Normalize


def state_fibonacci(num_qubits=8):
    """
    Create a state based on Fibonacci sequence bit patterns.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits
    
    Returns:
    --------
    Qobj
        Quantum state with Fibonacci-based structure
    """
    if num_qubits < 2:
        return state_plus(num_qubits)
    
    # Generate Fibonacci sequence
    fib = [1, 1]
    while len(fib) < num_qubits:
        fib.append(fib[-1] + fib[-2])
    
    # Create superposition based on Fibonacci pattern
    coeffs = np.zeros(2**num_qubits, dtype=complex)
    
    # Generate computational basis state indices based on Fibonacci pattern
    for i in range(len(coeffs)):
        # Convert index to binary representation
        binary = format(i, f'0{num_qubits}b')
        
        # Check if binary pattern contains consecutive 1s
        # (Fibonacci coding doesn't allow consecutive 1s)
        if '11' not in binary:
            # Weight states by golden ratio powers
            weight = 1.0
            for j, bit in enumerate(binary):
                if bit == '1' and j < len(fib):
                    weight *= 1.0 / fib[j]
            coeffs[i] = weight
    
    # Normalize the state
    coeffs /= np.linalg.norm(coeffs)
    
    # Create Qobj state
    dims = [[2] * num_qubits, [1] * num_qubits]
    return Qobj(coeffs, dims=dims)


def state_phi_sensitive(num_qubits=1, scaling_factor=None):
    """
    Create a quantum state that exhibits different behavior 
    based on proximity to the golden ratio.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits
    scaling_factor : float or None
        Scaling factor that influences state structure
        
    Returns:
    --------
    Qobj
        Quantum state with phi-sensitive properties
    """
    # Use PHI as default scaling factor
    if scaling_factor is None:
        scaling_factor = PHI
    
    # Calculate proximity to phi
    phi = PHI
    phi_proximity = np.exp(-(scaling_factor - phi)**2 / 0.1)  # Gaussian centered at phi
    
    if num_qubits == 1:
        # For 1 qubit, create superposition weighted by phi proximity
        alpha = np.cos(phi_proximity * np.pi/4)
        beta = np.sin(phi_proximity * np.pi/4)
        return (alpha * basis(2, 0) + beta * basis(2, 1)).unit()
    
    # For multiple qubits, create a more complex structure
    # Start with a uniform superposition
    state = state_plus(num_qubits)
    
    # Create different entanglement structures based on phi proximity
    if phi_proximity > 0.9:  # Very close to phi
        # Create GHZ-like state
        return state_ghz(num_qubits)
    elif phi_proximity > 0.5:  # Moderately close
        # Create W-like state
        return state_w(num_qubits)
    else:  # Far from phi
        # Create product state with phase differences
        phases = [np.exp(1j * scaling_factor * i * np.pi / num_qubits) for i in range(num_qubits)]
        single_states = [(basis(2, 0) + phases[i] * basis(2, 1)).unit() for i in range(num_qubits)]
        return tensor(*single_states)


def state_recursive_superposition(num_qubits=8, depth=3, scaling_factor=None):
    """
    Create a quantum state with recursive superposition structure.
    The state exhibits self-similar patterns at different scales.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits
    depth : int
        Recursion depth
    scaling_factor : float or None
        Scaling factor for recursive patterns (PHI by default)
        
    Returns:
    --------
    Qobj
        Quantum state with recursive superposition structure
    """
    if scaling_factor is None:
        scaling_factor = PHI
    
    if depth <= 0 or num_qubits <= 1:
        return state_plus(num_qubits)
    
    # Generate sub-states recursively
    if num_qubits >= 2:
        # Divide qubits into two groups for recursion
        n1 = num_qubits // 2
        n2 = num_qubits - n1
        
        # Create sub-states with reduced depth
        state1 = state_recursive_superposition(n1, depth-1, scaling_factor / PHI)
        state2 = state_recursive_superposition(n2, depth-1, scaling_factor / PHI**2)
        
        # Combine sub-states
        combined_state = tensor(state1, state2)
        
        # Add entanglement between the sub-states at the top level
        for i in range(min(n1, n2)):
            # Create controlled gate
            # Create identity and projection operators
            id2 = qeye(2)
            proj0 = Qobj([[1, 0], [0, 0]])
            proj1 = Qobj([[0, 0], [0, 1]])
            
            target_idx = n1 + i
            controller = i
            
            # Create controlled operation
            if num_qubits == 1:
                entangler = sigmax()  # No control for single qubit
            else:
                # For multiple qubits, create controlled operation
                op_list = []
                for j in range(num_qubits):
                    if j == controller:
                        op_list.append(proj0)
                    elif j == target_idx:
                        op_list.append(id2)
                    else:
                        op_list.append(id2)
                
                # First term: |0⟩⟨0| ⊗ I
                term1 = tensor(op_list)
                
                # Second term: |1⟩⟨1| ⊗ U
                op_list[controller] = proj1
                op_list[target_idx] = sigmax()
                term2 = tensor(op_list)
                
                entangler = term1 + term2
            
            combined_state = entangler * combined_state
        
        return combined_state
    else:
        return state_plus()
