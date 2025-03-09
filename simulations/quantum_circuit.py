"""
Quantum circuit implementations using qutip and qutip-qip.
"""

import numpy as np
from typing import List, Optional
from qutip import Qobj, ket2dm, mesolve, Options, sigmax, sigmaz
from qutip_qip.circuit import QubitCircuit
from qutip_qip.noise import Noise
from qutip_qip.operations import Gate
from .config import load_config
from constants import PHI

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

def get_scaled_unitary(H, time, scaling_factor=1.0):
    """
    Get the linearly scaled unitary operator for a given Hamiltonian.
    
    Parameters:
    - H (Qobj): Hamiltonian operator
    - time (float): Evolution time
    - scaling_factor (float): Factor to scale the unitary
    
    Returns:
    - Qobj: Scaled unitary operator
    """
    # Use matrix exponentiation directly for time evolution
    U_base = (-1j * H * time).expm()
    
    # Scale the unitary using matrix exponentiation
    # U^s = exp(s * log(U)) where s is the scaling factor
    if scaling_factor == 1.0:
        return U_base
    else:
        logU = U_base.logm()  # Matrix logarithm
        return (scaling_factor * logU).expm()


# Import necessary libraries for the get_phi_recursive_unitary function
from scipy.linalg import polar

def get_phi_recursive_unitary(H, time, scaling_factor=1.0, recursion_depth=3):
    """
    Create a unitary with recursive golden-ratio-based structure.
    
    Parameters:
    - H (Qobj): Hamiltonian operator
    - time (float): Evolution time
    - scaling_factor (float): Factor to scale the unitary
    - recursion_depth (int): Depth of recursion for phi-based patterns
    
    Returns:
    - Qobj: Scaled unitary with potential phi-resonance
    """
    # Use matrix exponentiation directly instead of propagator to avoid integration issues
    U_base = (-1j * H * time).expm()
    
    # Base case for recursion
    if recursion_depth <= 0 or scaling_factor == 1.0:
        return U_base
    
    # Create recursive structure based on Fibonacci/golden ratio pattern
    phi = PHI  # Golden ratio
    
    # Calculate proximity to phi for resonance effects
    phi_proximity = np.exp(-(scaling_factor - phi)**2 / 0.1)  # Gaussian centered at phi
    
    if phi_proximity > 0.9:  # Close to phi
        # At phi, create recursive operator structure with unitarity preservation
        U_phi1 = get_phi_recursive_unitary(H, time/phi, scaling_factor/phi, recursion_depth-1)
        U_phi2 = get_phi_recursive_unitary(H, time/phi**2, scaling_factor/phi**2, recursion_depth-1)
        
        # Enforce unitarity with polar decomposition
        product = U_phi1 * U_phi2
        # Use SVD to ensure unitarity
        u, p = polar(product.full())
        return Qobj(u)
    else:
        # For non-phi values, use different composition rule with unitarity enforcement
        logU = U_base.logm()
        # Ensure the exponent is anti-Hermitian (for unitarity)
        logU_anti = 0.5 * (logU - logU.dag())
        U_scaled = (scaling_factor * logU_anti).expm()
        
        # Add non-linear term that's most significant near phi
        correction_factor = scaling_factor/(1 + abs(scaling_factor - phi))
        
        if correction_factor > 0.2:
            # Create a unitary correction
            logU2 = (logU**2 - (logU**2).dag()) * 0.5j  # Make anti-Hermitian
            correction = (correction_factor * logU2).expm()
            result = U_scaled * correction
            
            # Final unitarity check
            u, p = polar(result.full())
            return Qobj(u)
        else:
            return U_scaled

<<<<<<< HEAD
# StandardCircuit class has been refactored and is now deprecated.
# Use ScaledCircuit with scaling_factor=1.0 for standard (unscaled) evolution.
=======
class StandardCircuit(QuantumCircuit):
    # TODO: refactor to remove this in favor of ScaledCircuit
    """
    Standard circuit evolution using qutip's solvers.
    """
    def __init__(self, base_hamiltonian, total_time=None, n_steps=None, c_ops=None):
        super().__init__(base_hamiltonian.dims[0][0] if isinstance(base_hamiltonian.dims[0], list) else 2)
        self.base_hamiltonian = base_hamiltonian
        self.config = load_config()
        
        # Get configuration values with proper type handling
        config_total_time = float(self.config.get('total_time', 1.0))
        config_n_steps = int(self.config.get('n_steps', 10))
        
        # Use provided values or defaults from config
        self.total_time = float(total_time if total_time is not None else config_total_time)
        self.n_steps = int(n_steps if n_steps is not None else config_n_steps)
        
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
                rate = float(dephasing.get('rate', 0.01))
                self.c_ops.append(np.sqrt(rate) * sigmaz())
            
            # Handle amplitude damping noise
            damping = noise_config.get('amplitude_damping', {})
            if isinstance(damping, dict) and damping.get('enabled', False):
                rate = float(damping.get('rate', 0.01))
                self.c_ops.append(np.sqrt(rate) * sigmax())
        else:
            self.c_ops = c_ops if c_ops is not None else []
        
    @property
    def noise(self):
        """Get the noise channel"""
        return self._noise

    def evolve_closed(self, initial_state, n_steps=None):
        """
        Evolution using qutip's sesolve for closed systems.
        """
        steps = int(n_steps if n_steps is not None else self.n_steps)
        tlist = np.linspace(0, self.total_time, steps + 1)
        
        if initial_state.isket:
            rho0 = ket2dm(initial_state)
        else:
            rho0 = initial_state
        
        mesolve_result = mesolve(
            self.base_hamiltonian,
            rho0,
            tlist,
            c_ops=[],  # No collapse operators for closed evolution
            options=Options(store_states=True, store_final_state=True)
        )
        
        # Create our EvolutionResult
        result = EvolutionResult(mesolve_result.states, tlist)
        result.eigenvalues = self.base_hamiltonian.eigenenergies()
        result.e_ops = []
        result.options = {}
        
        return result

    def evolve_open(self, initial_state, c_ops=None):
        """
        Evolution using qutip's mesolve for open systems.
        """
        tlist = np.linspace(0, self.total_time, self.n_steps + 1)
        
        if initial_state.isket:
            rho0 = ket2dm(initial_state)
        else:
            rho0 = initial_state
        
        # Use provided c_ops or default to instance c_ops
        collapse_ops = c_ops if c_ops is not None else self.c_ops
        
        mesolve_result = mesolve(
            self.base_hamiltonian,
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
>>>>>>> 033b46c71c02f6ef3bb74dc3fcb185487cd672aa

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
                rate = float(dephasing.get('rate', 0.01))
                self.c_ops.append(np.sqrt(rate) * sigmaz())
            
            # Handle amplitude damping noise
            damping = noise_config.get('amplitude_damping', {})
            if isinstance(damping, dict) and damping.get('enabled', False):
                rate = float(damping.get('rate', 0.01))
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
        H_scaled = self.get_scaled_hamiltonian(step_idx)
        dt = 1.0  # Unit time step
        return (-1j * dt * H_scaled).expm()
    
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
                # Use standard scaled unitary
                H_scaled = self.get_scaled_hamiltonian(idx)
                dt = 1.0  # Unit time step
                U = (-1j * dt * H_scaled).expm()
                
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
                    noise_strength = 0.05  # Default noise strength
                    
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
