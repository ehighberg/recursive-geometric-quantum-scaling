"""
Quantum circuit implementations using qutip and qutip-qip.

This module includes:
- Standard circuit evolution
- Scale-dependent quantum evolution
- Fibonacci braiding operations
- Optimized Hamiltonian creation for large systems
- Selective subspace evolution for performance
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Union, Any
from qutip import Qobj, ket2dm, mesolve, Options, sigmax, sigmaz, identity, tensor
from qutip_qip.circuit import QubitCircuit
from qutip_qip.noise import Noise
from qutip_qip.operations import Gate
from .config import load_config
from constants import PHI
# Import canonical unitary scaling functions from scaled_unitary.py
from .scaled_unitary import get_scaled_unitary, get_phi_recursive_unitary
# Import scipy functions
from scipy.linalg import polar
from scipy import sparse

class EvolutionResult:
    """Result object for quantum evolution containing states and times."""
    def __init__(self, states: List[Qobj], times: List[float]):
        self.states = states
        self.times = times
        self.eigenvalues: Optional[np.ndarray] = None
        self.e_ops = []
        self.options = {}
        
    @property
    def dims(self):
        """Get dimensions of the final state."""
        if self.states and len(self.states) > 0:
            return self.states[-1].dims
        return None
    
    def full(self):
        """Get the final state."""
        return self.states[-1] if self.states else None

class QuantumCircuit:
    """
    Base class for quantum circuits using qutip-qip's circuit building features.
    """
    def __init__(self, num_qubits, base_hamiltonian=None):
        self.num_qubits = num_qubits
        self.base_hamiltonian = base_hamiltonian
        self.circuit = QubitCircuit(num_qubits)
        self.config = load_config()
        
    def add_gate(self, gate_name, targets, controls=None, arg_value=None):
        """Add a gate to the circuit."""
        self.circuit.add_gate(gate_name, targets=targets, controls=controls, arg_value=arg_value)
    
    def get_unitary(self):
        """Get the unitary matrix for the entire circuit."""
        return self.circuit.compute_unitary()
    
    def run(self, initial_state):
        """Run the circuit on an initial state."""
        U = self.get_unitary()
        if initial_state.isket:
            return U * initial_state
        else:
            return U * initial_state * U.dag()

class StandardCircuit(QuantumCircuit):
    """
    Standard circuit evolution using qutip's solvers.
    
    Note:
    This class is now a thin wrapper around ScaledCircuit with scaling_factor=1.0
    For new code, use ScaledCircuit directly.
    """
    def __init__(self, base_hamiltonian, total_time=None, n_steps=None, c_ops=None):
        # Initialize parent class
        super().__init__(base_hamiltonian.dims[0][0] if isinstance(base_hamiltonian.dims[0], list) else 2)
        self.base_hamiltonian = base_hamiltonian
        self.config = load_config()
        
        # Get configuration values with proper type handling
        config_total_time = float(self.config.get('total_time', 1.0))
        config_n_steps = int(self.config.get('n_steps', 10))
        
        # Use provided values or defaults from config
        self.total_time = float(total_time if total_time is not None else config_total_time)
        self.n_steps = int(n_steps if n_steps is not None else config_n_steps)
        
        # Create the underlying ScaledCircuit with scaling_factor=1.0
        self._scaled_circuit = ScaledCircuit(
            base_hamiltonian, 
            scaling_factor=1.0,
            c_ops=c_ops
        )
        
        # Store c_ops for compatibility with old code
        self.c_ops = self._scaled_circuit.c_ops
        
        # Initialize noise channels
        self._noise = self._scaled_circuit.noise
    
    @property
    def noise(self):
        """Get the noise channel"""
        return self._noise

    def evolve_closed(self, initial_state, n_steps=None):
        """
        Evolution using qutip's sesolve for closed systems.
        
        This now delegates to ScaledCircuit.evolve_closed with scaling_factor=1.0
        """
        # Forward to the ScaledCircuit implementation
        steps = int(n_steps if n_steps is not None else self.n_steps)
        result = self._scaled_circuit.evolve_closed(initial_state, steps)
        
        # Add eigenvalues for compatibility with old code that may expect it
        result.eigenvalues = self.base_hamiltonian.eigenenergies()
        
        return result

    def evolve_open(self, initial_state, c_ops=None):
        """
        Evolution using qutip's mesolve for open systems.
        
        This now delegates to ScaledCircuit.evolve_open with scaling_factor=1.0
        """
        # Forward to the ScaledCircuit implementation
        return self._scaled_circuit.evolve_open(initial_state, c_ops, self.n_steps)

class CustomCircuit(QuantumCircuit):
    """
    Custom circuit implementation for arbitrary gate sequences.
    """
    def __init__(self, num_qubits=2, total_time=None, n_steps=None):
        super().__init__(num_qubits)
        self.config = load_config()
        
        # Get configuration values with proper type handling
        config_total_time = float(self.config.get('total_time', 1.0))
        config_n_steps = int(self.config.get('n_steps', 10))
        
        # Use provided values or defaults from config
        self.total_time = float(total_time if total_time is not None else config_total_time)
        self.n_steps = int(n_steps if n_steps is not None else config_n_steps)
        
        # Initialize gates list
        self.gates = []
    
    def add_gate(self, gate_type, qubits, params=None, angle=None):
        """
        Add a gate to the circuit.
        
        Parameters:
        -----------
        gate_type : str
            Type of gate (e.g., "RZ", "CNOT")
        qubits : list
            List of qubit indices the gate acts on
        params : list, optional
            Additional parameters for the gate
        angle : float, optional
            Rotation angle for rotation gates
        """
        if gate_type == "RZ":
            # Rotation around Z axis
            target = qubits[0]
            self.circuit.add_gate("RZ", targets=[target], arg_value=angle)
        elif gate_type == "CNOT":
            # CNOT gate
            control, target = qubits
            self.circuit.add_gate("CNOT", targets=[target], controls=[control])
        else:
            # Generic gate
            self.circuit.add_gate(gate_type, targets=qubits, arg_value=angle)
        
        # Store gate for reference
        self.gates.append((gate_type, qubits, params, angle))
    
    def evolve_closed(self, initial_state):
        """
        Evolve the initial state through the circuit without noise.
        
        Parameters:
        -----------
        initial_state : Qobj
            Initial quantum state
            
        Returns:
        --------
        EvolutionResult
            Result object containing states and times
        """
        # Get the unitary for the entire circuit
        U = self.circuit.compute_unitary()
        
        # Apply the unitary to the initial state
        if initial_state.isket:
            final_state = U * initial_state
        else:
            final_state = U * initial_state * U.dag()
        
        # Create intermediate states for visualization
        n_gates = len(self.gates)
        if n_gates > 0:
            # Create n_gates + 1 time points
            times = np.linspace(0, self.total_time, n_gates + 1)
            
            # Initialize states list with initial state
            states = [initial_state]
            
            # Apply gates one by one to create intermediate states
            current_state = initial_state
            for i in range(n_gates):
                # Build circuit up to this gate
                partial_circuit = QubitCircuit(self.num_qubits)
                for j in range(i + 1):
                    gate_type, qubits, params, angle = self.gates[j]
                    if gate_type == "RZ":
                        partial_circuit.add_gate("RZ", targets=[qubits[0]], arg_value=angle)
                    elif gate_type == "CNOT":
                        partial_circuit.add_gate("CNOT", targets=[qubits[1]], controls=[qubits[0]])
                    else:
                        partial_circuit.add_gate(gate_type, targets=qubits, arg_value=angle)
                
                # Get unitary for partial circuit
                U_partial = partial_circuit.compute_unitary()
                
                # Apply to initial state
                if initial_state.isket:
                    current_state = U_partial * initial_state
                else:
                    current_state = U_partial * initial_state * U_partial.dag()
                
                states.append(current_state)
        else:
            # No gates, just initial and final states (which are the same)
            times = [0, self.total_time]
            states = [initial_state, initial_state]
        
        # Create result object
        result = EvolutionResult(states, times)
        result.e_ops = []
        result.options = {}
        
        return result
    
    def evolve_open(self, initial_state, c_ops):
        """
        Evolve the initial state through the circuit with noise.
        
        Parameters:
        -----------
        initial_state : Qobj
            Initial quantum state
        c_ops : list
            List of collapse operators for noise
            
        Returns:
        --------
        EvolutionResult
            Result object containing states and times
        """
        # Convert to density matrix if needed
        if initial_state.isket:
            rho0 = ket2dm(initial_state)
        else:
            rho0 = initial_state
        
        # Create time list
        tlist = np.linspace(0, self.total_time, self.n_steps + 1)
        
        # Create effective Hamiltonian from the circuit
        # This is a simplification - in a real implementation, we would
        # need to create a time-dependent Hamiltonian that implements the gates
        U = self.circuit.compute_unitary()
        H_eff = -1j * U.dag() * U.logm()  # Effective Hamiltonian
        
        # Solve master equation
        mesolve_result = mesolve(
            H_eff,
            rho0,
            tlist,
            c_ops,
            options=Options(store_states=True)
        )
        
        # Create result object
        result = EvolutionResult(mesolve_result.states, tlist)
        result.e_ops = []
        result.options = {}
        
        return result

class ScaledCircuit(QuantumCircuit):
    """
    Geometrically-scaled circuit evolution using qutip's features.
    """
    def __init__(self, base_hamiltonian, scaling_factor=None, c_ops=None, recursion_depth=None):
        super().__init__(base_hamiltonian.dims[0][0] if isinstance(base_hamiltonian.dims[0], list) else 2)
        self.base_hamiltonian = base_hamiltonian
        self.config = load_config()
        
        # Get configuration value with proper type handling
        config_scale_factor = float(self.config.get('scale_factor', 1.0))
        config_recursion_depth = int(self.config.get('recursion_depth', 3))
        
        # Use provided value or default from config
        self.scale_factor = float(scaling_factor if scaling_factor is not None else config_scale_factor)
        self.recursion_depth = int(recursion_depth if recursion_depth is not None else config_recursion_depth)
        
        # Initialize noise channels
        self._noise = Noise()  # Initialize without arguments
        
        # Configure noise if needed
        noise_config = self.config.get('noise', {})
        if isinstance(noise_config, dict) and noise_config:
            # Create collapse operators based on noise configuration
            self.c_ops = []
            
            # Handle dephasing noise
            dephasing = noise_config.get('dephasing', {})
            if isinstance(dephasing, dict) and dephasing.get('enabled', False):
                rate = float(dephasing.get('rate', 0.01))  # Using 0.01 as default rate
                self.c_ops.append(np.sqrt(rate) * sigmaz())
            
            # Handle amplitude damping noise
            damping = noise_config.get('amplitude_damping', {})
            if isinstance(damping, dict) and damping.get('enabled', False):
                rate = float(damping.get('rate', 0.01))  # Using 0.01 as default rate
                self.c_ops.append(np.sqrt(rate) * sigmax())
        else:
            self.c_ops = c_ops if c_ops is not None else []
        
        # Flag for using phi-recursive evolution instead of standard scaling
        self.use_phi_recursive = False
    
    @property
    def noise(self):
        """Get the noise channel"""
        return self._noise

    def get_scaled_hamiltonian(self, step_idx):
        """Get Hamiltonian scaled by scale_factor^step_idx."""
        scale = (self.scale_factor ** step_idx)
        return scale * self.base_hamiltonian
    
    def scale_unitary(self, step_idx):
        """
        Get the unitary operator for a given scaling step.
        
        Args:
            step_idx (int): The scaling step index
            
        Returns:
            Qobj: The scaled unitary operator
        """
        # Use the canonical implementation from scaled_unitary.py
        scaling_factor = self.scale_factor ** step_idx
        return get_scaled_unitary(self.base_hamiltonian, time=1.0, scaling_factor=scaling_factor)
    
    def get_phi_recursive_unitary(self, step_idx):
        """
        Get a phi-recursive unitary operator for a given scaling step.
        
        Args:
            step_idx (int): The scaling step index
            
        Returns:
            Qobj: The phi-recursive unitary operator
        """
        H = self.base_hamiltonian
        time = 1.0  # Unit time
        scaling_factor = self.scale_factor ** step_idx
        return get_phi_recursive_unitary(H, time, scaling_factor, self.recursion_depth)
    
    def evolve_closed(self, initial_state, n_steps=None):
        """
        Phi-scaled evolution for closed systems.
        """
        steps = int(n_steps if n_steps is not None else self.config.get('n_steps', 10))
        
        if initial_state.isket:
            state = initial_state
        else:
            # If density matrix provided, we'll evolve it directly
            state = initial_state
        
        states = []
        times = []
        current_time = 0
        
        for idx in range(steps):
            if self.use_phi_recursive:
                # Use phi-recursive unitary
                U = self.get_phi_recursive_unitary(idx)
            else:
                # Use standard scaled unitary from the canonical implementation
                U = self.scale_unitary(idx)
                
            if state.isket:
                state = U * state
            else:
                state = U * state * U.dag()
            
            states.append(state)
            current_time += 1.0  # Unit time step
            times.append(current_time)
        
        result = EvolutionResult(states, times)
        result.e_ops = []
        result.options = {}
        return result

    def evolve_open(self, initial_state, c_ops=None, n_steps=None):
        """
        Phi-scaled evolution for open systems using qutip's master equation solver.
        """
        steps = int(n_steps if n_steps is not None else self.config.get('n_steps', 10))
        tlist = np.linspace(0, steps, steps + 1)
        
        if initial_state.isket:
            rho0 = ket2dm(initial_state)
        else:
            rho0 = initial_state
        
        # Use provided c_ops or default to instance c_ops
        collapse_ops = c_ops if c_ops is not None else self.c_ops
        
        # Build effective Hamiltonian based on the evolution type
        if self.use_phi_recursive:
            # For phi-recursive evolution, use a simpler approximation 
            # We can't directly simulate the recursive unitary with mesolve
            # So we'll use a modified scaled Hamiltonian that approximates it
            # This is less accurate than the closed system approach
            
            # Create a weighted sum of scaled Hamiltonians
            H_eff = 0
            phi = PHI
            
            for idx in range(steps):
                # Base scaling with adjustment for phi proximity
                scale = (self.scale_factor ** idx)
                phi_proximity = np.exp(-((self.scale_factor - phi) ** 2) / 0.1)
                
                # Modify scaling based on phi proximity
                if phi_proximity > 0.8:
                    # For values close to phi, add recursive terms
                    primary_scale = scale * (1 + 0.1 * phi_proximity)
                    H_eff += primary_scale * self.base_hamiltonian
                    
                    # Add secondary scale terms if enough recursion depth
                    if self.recursion_depth > 1 and idx < steps - 1:
                        secondary_scale = scale/phi * 0.2 * phi_proximity
                        H_eff += secondary_scale * self.base_hamiltonian
                else:
                    # For values not close to phi, just use regular scaling
                    H_eff += scale * self.base_hamiltonian
        else:
            # For standard scaling, simply sum the scaled Hamiltonians
            H_eff = sum(self.get_scaled_hamiltonian(idx) for idx in range(steps))
        
        # Use the effective Hamiltonian for time evolution
        H_terms = H_eff
        
        # Solve master equation
        mesolve_result = mesolve(
            H_terms,
            rho0,
            tlist,
            collapse_ops,
            options=Options(store_states=True)
        )
        
        # Create our EvolutionResult
        result = EvolutionResult(mesolve_result.states, tlist)
        result.e_ops = []
        result.options = {}
        return result

def create_optimized_hamiltonian(
    num_qubits: int, 
    hamiltonian_type: str = "x", 
    sparse_threshold: int = 3,
    enable_memory_optimization: bool = True
) -> Union[Qobj, sparse.csr_matrix]:
    """
    Create optimized Hamiltonian representation for quantum systems.
    
    This function automatically selects the most efficient representation based on
    system size, using dense matrices for small systems (<= sparse_threshold qubits)
    and sparse matrices for larger systems to minimize memory requirements.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the system
    hamiltonian_type : str, optional
        Type of Hamiltonian to create:
        - "x": Sum of sigma_x operators on each qubit
        - "z": Sum of sigma_z operators on each qubit
        - "y": Sum of sigma_y operators on each qubit
        - "xy": Heisenberg XY model with nearest-neighbor coupling
        - "ising": Transverse field Ising model
    sparse_threshold : int, optional
        Threshold (in number of qubits) above which to use sparse representation
    enable_memory_optimization : bool, optional
        Whether to enable memory optimizations for very large systems
    
    Returns:
    --------
    Union[Qobj, sparse.csr_matrix]
        Optimized Hamiltonian representation
    """
    # For small systems, use standard dense representation
    if num_qubits <= sparse_threshold:
        # Use the built-in StandardCircuit approach for small systems
        if hamiltonian_type == "x":
            H = sum(tensor([sigmax() if i == j else identity(2) for j in range(num_qubits)]) 
                   for i in range(num_qubits))
        elif hamiltonian_type == "z":
            H = sum(tensor([sigmaz() if i == j else identity(2) for j in range(num_qubits)]) 
                   for i in range(num_qubits))
        elif hamiltonian_type == "y":
            from qutip import sigmay
            H = sum(tensor([sigmay() if i == j else identity(2) for j in range(num_qubits)]) 
                   for i in range(num_qubits))
        elif hamiltonian_type == "xy":
            from qutip import sigmap, sigmam
            # XY model with nearest-neighbor coupling
            H = sum(tensor([sigmap() if i == j else sigmam() if i == j+1 % num_qubits else identity(2) 
                          for j in range(num_qubits)]) 
                   for i in range(num_qubits-1))
            H += H.dag()  # Make Hermitian
        elif hamiltonian_type == "ising":
            # Transverse field Ising model with periodic boundary conditions
            h_field = 1.0  # Transverse field strength
            J = 1.0  # Coupling strength
            
            # Field term
            H_field = sum(tensor([sigmax() if i == j else identity(2) for j in range(num_qubits)]) 
                         for i in range(num_qubits))
            
            # Interaction term
            H_int = sum(tensor([sigmaz() if i == j or i == (j+1) % num_qubits else identity(2) 
                              for j in range(num_qubits)]) 
                       for i in range(num_qubits))
            
            H = -h_field * H_field - J * H_int
        else:
            raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}")
            
        return H
    
    # For larger systems, use sparse representation
    else:
        # Define single-qubit operators as sparse matrices
        sx = sparse.csr_matrix([[0, 1], [1, 0]])
        sy = sparse.csr_matrix([[0, -1j], [1j, 0]])
        sz = sparse.csr_matrix([[1, 0], [0, -1]])
        sm = sparse.csr_matrix([[0, 0], [1, 0]])
        sp = sparse.csr_matrix([[0, 1], [0, 0]])
        si = sparse.eye(2, format='csr')  # Sparse identity
        
        # Select operator based on Hamiltonian type
        if hamiltonian_type == "x":
            single_op = sx
        elif hamiltonian_type == "y":
            single_op = sy
        elif hamiltonian_type == "z":
            single_op = sz
        elif hamiltonian_type in ["xy", "ising"]:
            # These models need special handling due to multi-site interactions
            return _construct_multisite_hamiltonian_sparse(num_qubits, hamiltonian_type, 
                                                         enable_memory_optimization)
        else:
            raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}")

        # Build full Hamiltonian efficiently using sparse operations
        hamiltonian = None
        
        # Create operators that act on each qubit
        for i in range(num_qubits):
            # Create the terms list - identity on all qubits except target
            terms = [si] * num_qubits
            terms[i] = single_op
            
            # Compute tensor product efficiently
            term_op = _sparse_tensor_product(terms)
            
            # Add to Hamiltonian
            if hamiltonian is None:
                hamiltonian = term_op
            else:
                hamiltonian += term_op
        
        # Optionally convert to Qobj for compatibility with QuTiP functions
        if not enable_memory_optimization:
            return Qobj(hamiltonian, dims=[[2]*num_qubits, [2]*num_qubits])
        else:
            # Return raw sparse matrix for maximum memory efficiency
            return hamiltonian

def _sparse_tensor_product(operators: List[sparse.spmatrix]) -> sparse.csr_matrix:
    """
    Compute tensor product of sparse matrices efficiently.
    
    Parameters:
    -----------
    operators : List[sparse.spmatrix]
        List of sparse matrices to tensor together
    
    Returns:
    --------
    sparse.csr_matrix
        Sparse tensor product result
    """
    # Base case for recursion
    if len(operators) == 1:
        return operators[0]
    
    # Start with the first operator
    result = operators[0]
    
    # Iterate through remaining operators using sparse kron
    for op in operators[1:]:
        result = sparse.kron(result, op, format='csr')
    
    return result

def _construct_multisite_hamiltonian_sparse(
    num_qubits: int, 
    hamiltonian_type: str,
    memory_optimization: bool = True
) -> Union[Qobj, sparse.csr_matrix]:
    """
    Construct multi-site interaction Hamiltonian using sparse matrices.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the system
    hamiltonian_type : str
        Type of Hamiltonian ("xy" or "ising")
    memory_optimization : bool
        Whether to return raw sparse matrix instead of Qobj
    
    Returns:
    --------
    Union[Qobj, sparse.csr_matrix]
        Sparse Hamiltonian
    """
    # Define single-qubit operators as sparse matrices
    sx = sparse.csr_matrix([[0, 1], [1, 0]])
    sy = sparse.csr_matrix([[0, -1j], [1j, 0]])
    sz = sparse.csr_matrix([[1, 0], [0, -1]])
    sm = sparse.csr_matrix([[0, 0], [1, 0]])
    sp = sparse.csr_matrix([[0, 1], [0, 0]])
    si = sparse.eye(2, format='csr')  # Sparse identity
    
    hamiltonian = None
    
    if hamiltonian_type == "xy":
        # XY model with nearest neighbor interactions
        for i in range(num_qubits - 1):
            # Create terms for i, i+1 interaction (X_i X_{i+1} + Y_i Y_{i+1})
            
            # X_i X_{i+1} term
            terms_xx = [si] * num_qubits
            terms_xx[i] = sx
            terms_xx[i+1] = sx
            xx_term = _sparse_tensor_product(terms_xx)
            
            # Y_i Y_{i+1} term
            terms_yy = [si] * num_qubits
            terms_yy[i] = sy
            terms_yy[i+1] = sy
            yy_term = _sparse_tensor_product(terms_yy)
            
            # Add to Hamiltonian
            if hamiltonian is None:
                hamiltonian = xx_term + yy_term
            else:
                hamiltonian += xx_term + yy_term
                
    elif hamiltonian_type == "ising":
        # Transverse field Ising model
        h_field = 1.0  # Transverse field strength
        J = 1.0  # Coupling strength
        
        # Field term: sum_i X_i
        field_term = None
        for i in range(num_qubits):
            terms = [si] * num_qubits
            terms[i] = sx
            term_op = _sparse_tensor_product(terms)
            
            if field_term is None:
                field_term = term_op
            else:
                field_term += term_op
        
        # Interaction term: sum_i Z_i Z_{i+1}
        interact_term = None
        for i in range(num_qubits - 1):
            terms = [si] * num_qubits
            terms[i] = sz
            terms[i+1] = sz
            term_op = _sparse_tensor_product(terms)
            
            if interact_term is None:
                interact_term = term_op
            else:
                interact_term += term_op
                
        # Add periodic boundary term if more than 2 qubits
        if num_qubits > 2:
            terms = [si] * num_qubits
            terms[0] = sz
            terms[num_qubits-1] = sz
            interact_term += _sparse_tensor_product(terms)
        
        # Combine terms with appropriate coefficients
        hamiltonian = -h_field * field_term - J * interact_term
        
    # Return either Qobj or raw sparse matrix
    if not memory_optimization:
        return Qobj(hamiltonian, dims=[[2]*num_qubits, [2]*num_qubits])
    else:
        return hamiltonian

def evolve_selective_subspace(
    initial_state: Qobj,
    hamiltonian: Union[Qobj, sparse.spmatrix],
    times: np.ndarray,
    importance_threshold: float = 0.01,
    important_states_mask: Optional[np.ndarray] = None,
    c_ops: Optional[List] = None
) -> List[Qobj]:
    """
    Evolve only the most relevant part of the state vector for efficient large-system simulation.
    
    This function identifies the most significant basis states in the initial wavefunction 
    and evolves only that subspace, which can dramatically improve performance for large systems
    where most of the Hilbert space remains unpopulated.
    
    Parameters:
    -----------
    initial_state : Qobj
        Initial quantum state
    hamiltonian : Union[Qobj, sparse.spmatrix]
        Hamiltonian for the evolution (can be sparse or dense)
    times : np.ndarray
        Array of time points for evolution
    importance_threshold : float, optional
        Threshold for amplitude magnitude to be considered important (ignored if mask provided)
    important_states_mask : Optional[np.ndarray], optional
        Boolean mask specifying which basis states to include in the evolution
    c_ops : Optional[List], optional
        List of collapse operators for open system evolution
        
    Returns:
    --------
    List[Qobj]
        List of evolved quantum states
    """
    # Convert to ket if density matrix
    is_dm_input = not initial_state.isket
    if is_dm_input:
        # Extract diagonal for importance analysis
        diag_elems = np.abs(initial_state.diag())
        state_for_mask = initial_state
    else:
        # Use amplitudes for importance analysis
        diag_elems = np.abs(initial_state.full().flatten())**2
        state_for_mask = initial_state
    
    # Identify important basis states if mask not provided
    if important_states_mask is None:
        # Determine threshold adaptively if not manually set
        if importance_threshold <= 0:
            # Use cumulative probability method: keep states that make up 99.9% of total probability
            sorted_probs = np.sort(diag_elems)[::-1]  # Descending order
            cumulative = np.cumsum(sorted_probs)
            cutoff_idx = np.searchsorted(cumulative, 0.999)
            adaptive_threshold = sorted_probs[min(cutoff_idx, len(sorted_probs)-1)]
            importance_threshold = max(adaptive_threshold, 1e-6)  # Ensure minimum threshold
        
        # Create mask for states above threshold
        important_states_mask = diag_elems > importance_threshold
        
        # Ensure at least a few states are included
        if np.sum(important_states_mask) < 5:
            # Take top 5 states
            top_indices = np.argsort(diag_elems)[-5:]
            important_states_mask[top_indices] = True
    
    # Check if reduction is worthwhile
    n_important = np.sum(important_states_mask)
    total_states = len(diag_elems)
    
    if n_important > 0.5 * total_states:
        # If more than half the states are important, use regular evolution
        # This is more efficient than the reduction overhead
        return _evolve_full_system(initial_state, hamiltonian, times, c_ops)
    
    # Get indices of important states
    important_indices = np.where(important_states_mask)[0]
    
    # Project initial state to subspace
    if is_dm_input:
        # For density matrix, extract submatrix
        reduced_state = _extract_submatrix(state_for_mask, important_indices)
    else:
        # For ket, extract amplitudes
        amplitudes = initial_state.full().flatten()
        reduced_amplitudes = amplitudes[important_indices]
        # Normalize
        norm = np.sqrt(np.sum(np.abs(reduced_amplitudes)**2))
        if norm > 0:
            reduced_amplitudes = reduced_amplitudes / norm
        # Create reduced state
        reduced_state = Qobj(reduced_amplitudes.reshape(-1, 1))
    
    # Project Hamiltonian to subspace
    if isinstance(hamiltonian, Qobj):
        # QuTiP Hamiltonian
        H_matrix = hamiltonian.full()
        reduced_H_matrix = H_matrix[np.ix_(important_indices, important_indices)]
        reduced_H = Qobj(reduced_H_matrix)
    else:
        # Sparse Hamiltonian
        reduced_H_matrix = hamiltonian[np.ix_(important_indices, important_indices)]
        reduced_H = Qobj(reduced_H_matrix)
    
    # Project collapse operators to subspace if provided
    reduced_c_ops = None
    if c_ops:
        reduced_c_ops = []
        for c_op in c_ops:
            if isinstance(c_op, Qobj):
                c_matrix = c_op.full()
                reduced_c_matrix = c_matrix[np.ix_(important_indices, important_indices)]
                reduced_c_ops.append(Qobj(reduced_c_matrix))
            else:
                # Handle non-Qobj c_ops (e.g., function callbacks)
                # Skip if can't be projected
                pass
    
    # Evolve the reduced system
    if reduced_c_ops:
        # Open system evolution
        result = mesolve(reduced_H, reduced_state, times, reduced_c_ops, options=Options(store_states=True))
        reduced_states = result.states
    else:
        # Closed system evolution
        if is_dm_input:
            # Use mesolve for density matrices even in closed system
            result = mesolve(reduced_H, reduced_state, times, [], options=Options(store_states=True))
            reduced_states = result.states
        else:
            # Use faster method for pure states
            from qutip import sesolve
            result = sesolve(reduced_H, reduced_state, times, options=Options(store_states=True))
            reduced_states = result.states
    
    # Project back to full Hilbert space
    full_states = []
    for reduced_state in reduced_states:
        if reduced_state.isket:
            # Expand ket state
            full_vector = np.zeros(total_states, dtype=complex)
            full_vector[important_indices] = reduced_state.full().flatten()
            full_state = Qobj(full_vector.reshape(-1, 1))
        else:
            # Expand density matrix
            full_matrix = np.zeros((total_states, total_states), dtype=complex)
            reduced_matrix = reduced_state.full()
            for i, row_idx in enumerate(important_indices):
                for j, col_idx in enumerate(important_indices):
                    full_matrix[row_idx, col_idx] = reduced_matrix[i, j]
            full_state = Qobj(full_matrix)
        
        full_states.append(full_state)
    
    return full_states

def _extract_submatrix(dm: Qobj, indices: np.ndarray) -> Qobj:
    """
    Extract submatrix from density matrix corresponding to selected indices.
    
    Parameters:
    -----------
    dm : Qobj
        Density matrix to extract from
    indices : np.ndarray
        Indices to include in submatrix
    
    Returns:
    --------
    Qobj
        Reduced density matrix
    """
    if dm.isket:
        # Handle ket state
        full_vector = dm.full().flatten()
        reduced_vector = full_vector[indices]
        # Normalize
        norm = np.sqrt(np.sum(np.abs(reduced_vector)**2))
        if norm > 0:
            reduced_vector = reduced_vector / norm
        return Qobj(reduced_vector.reshape(-1, 1))
    else:
        # Extract submatrix from density matrix
        full_matrix = dm.full()
        reduced_matrix = full_matrix[np.ix_(indices, indices)]
        # Normalize if trace != 1
        trace = np.trace(reduced_matrix)
        if abs(trace) > 1e-10:
            reduced_matrix = reduced_matrix / trace
        return Qobj(reduced_matrix)

def _evolve_full_system(
    initial_state: Qobj,
    hamiltonian: Union[Qobj, sparse.spmatrix],
    times: np.ndarray,
    c_ops: Optional[List] = None
) -> List[Qobj]:
    """
    Evolve the full system without any dimension reduction.
    
    Parameters:
    -----------
    initial_state : Qobj
        Initial quantum state
    hamiltonian : Union[Qobj, sparse.spmatrix]
        Hamiltonian for the evolution
    times : np.ndarray
        Array of time points for evolution
    c_ops : Optional[List], optional
        List of collapse operators for open system evolution
        
    Returns:
    --------
    List[Qobj]
        List of evolved quantum states
    """
    # Ensure Hamiltonian is Qobj
    if not isinstance(hamiltonian, Qobj):
        hamiltonian = Qobj(hamiltonian)
    
    # Determine evolution type
    if c_ops or not initial_state.isket:
        # Open system evolution or density matrix input
        result = mesolve(hamiltonian, initial_state, times, c_ops or [], options=Options(store_states=True))
    else:
        # Closed system evolution with pure state
        from qutip import sesolve
        result = sesolve(hamiltonian, initial_state, times, options=Options(store_states=True))
    
    return result.states

class FibonacciBraidingCircuit(QuantumCircuit):
    """
    Circuit implementation for Fibonacci anyon braiding using qutip's features.
    """
    def __init__(self, num_qubits=2):
        super().__init__(num_qubits)
        self.braids = []
    
    def add_braid(self, braid_operator):
        """Add a braid operation to the circuit."""
        # Create a custom gate from the braid operator
        gate = Gate(name=f"braid_{len(self.braids)}", targets=[0,1], arg_value=0)
        gate.matrix = braid_operator
        self.circuit.add_gate(gate)
        self.braids.append(braid_operator)
    
    def evolve(self, initial_state):
        """
        Evolve the initial state through the braiding circuit.
        
        Returns:
        - result: Evolution result containing states and times
        """
        # Create result object to match other evolution methods
        # Store states and times
        states = [initial_state]
        times = [0.0]
        
        # Apply each braid operation sequentially
        current_state = initial_state
        for i, braid in enumerate(self.braids):
            if current_state.isket:
                current_state = braid * current_state
            else:
                current_state = braid * current_state * braid.dag()
            states.append(current_state)
            times.append(float(i + 1))
        
        result = EvolutionResult(states, times)
        result.e_ops = []
        result.options = {}
        return result
        
    def evolve_with_noise(self, initial_state, c_ops=None):
        """
        Evolve the initial state through the braiding circuit with noise.
        
        Parameters:
        - initial_state: Initial quantum state
        - c_ops: List of collapse operators for noise
        
        Returns:
        - result: Evolution result containing states and times
        """
        # Import Options and mesolve from qutip explicitly to ensure they're available
        from qutip import Options, sigmaz, identity
        
        # Create result object to match other evolution methods
        # Store states and times
        states = [initial_state]
        times = [0.0]
        
        # Convert to density matrix if needed
        if initial_state.isket:
            current_state = ket2dm(initial_state)
        else:
            current_state = initial_state
        
        # Apply each braid operation sequentially with noise
        for i, braid in enumerate(self.braids):
            # Apply braid
            current_state = braid * current_state * braid.dag()
            
            # Apply noise if c_ops exist
            if c_ops:
                # Manual noise application for compatibility
                for c_op in c_ops:
                    # Apply noise manually with fixed strength
                    noise_strength = 0.05  # Default noise strength of 5%
                    
                    # Ensure compatibility by checking dimensions
                    if hasattr(c_op, 'dims') and hasattr(current_state, 'dims'):
                        try:
                            # Try to apply the collapse operator
                            noise_effect = c_op * current_state * c_op.dag()
                            current_state = (1-noise_strength) * current_state + noise_strength * noise_effect
                        except:
                            # If dimensions don't match, skip this operator
                            pass
                
                # Ensure trace preservation
                if hasattr(current_state, 'tr'):
                    tr = current_state.tr()
                    if abs(tr) > 1e-10:
                        current_state = current_state / tr
            
            states.append(current_state)
            times.append(float(i + 1))
        
        result = EvolutionResult(states, times)
        result.e_ops = []
        result.options = {}
        return result
