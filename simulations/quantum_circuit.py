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

class EvolutionResult:
    """Result object for quantum evolution containing states and times."""
    def __init__(self, states: List[Qobj], times: List[float]):
        self.states = states
        self.times = times
        self.eigenvalues: Optional[np.ndarray] = None
        
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
    """
    def __init__(self, base_hamiltonian, total_time=None, n_steps=None, c_ops=None):
        super().__init__(base_hamiltonian.dims[0][0] if isinstance(base_hamiltonian.dims[0], list) else 2)
        self.base_hamiltonian = base_hamiltonian
        self.config = load_config()
        self.total_time = total_time if total_time is not None else self.config.get('total_time', 1.0)
        self.n_steps = n_steps if n_steps is not None else self.config.get('n_steps', 10)
        
        # Initialize noise channels
        noise_config = self.config.get('noise', {})
        self._noise = Noise(noise_config)
        self.c_ops = c_ops if c_ops is not None else self.config.get('c_ops', [])
        
    @property
    def noise(self):
        """Get the noise channel"""
        return self._noise

    def evolve_closed(self, initial_state, n_steps=None):
        """
        Evolution using qutip's sesolve for closed systems.
        """
        steps = n_steps if n_steps is not None else self.n_steps
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
        
        return result

    def evolve_open(self, initial_state):
        """
        Evolution using qutip's mesolve for open systems.
        """
        tlist = np.linspace(0, self.total_time, self.n_steps + 1)
        
        if initial_state.isket:
            rho0 = ket2dm(initial_state)
        else:
            rho0 = initial_state
        
        mesolve_result = mesolve(
            self.base_hamiltonian,
            rho0,
            tlist,
            self.c_ops,
            options=Options(store_states=True)
        )
        
        # Create our EvolutionResult
        result = EvolutionResult(mesolve_result.states, tlist)
        return result

class ScaledCircuit(QuantumCircuit):
    """
    Geometrically-scaled circuit evolution using qutip's features.
    """
    def __init__(self, base_hamiltonian, scaling_factor=None, c_ops=None):
        super().__init__(base_hamiltonian.dims[0][0] if isinstance(base_hamiltonian.dims[0], list) else 2)
        self.base_hamiltonian = base_hamiltonian
        self.config = load_config()
        self.scale_factor = scaling_factor if scaling_factor is not None else self.config.get('scale_factor', 1)
        self.c_ops = c_ops if c_ops is not None else self.config.get('c_ops', [])
    
    def get_scaled_hamiltonian(self, step_idx):
        """Get Hamiltonian scaled by phi^step_idx."""
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
    
    def evolve_closed(self, initial_state, n_steps=None):
        """
        Phi-scaled evolution for closed systems.
        """
        steps = n_steps if n_steps is not None else self.config.get('n_steps', 10)
        
        if initial_state.isket:
            state = initial_state
        else:
            # If density matrix provided, we'll evolve it directly
            state = initial_state
        
        states = []
        times = []
        current_time = 0
        
        for idx in range(steps):
            H_scaled = self.get_scaled_hamiltonian(idx)
            dt = 1.0  # Unit time step
            
            # Evolve for unit time with scaled Hamiltonian
            U = (-1j * dt * H_scaled).expm()
            if state.isket:
                state = U * state
            else:
                state = U * state * U.dag()
            
            states.append(state)
            current_time += dt
            times.append(current_time)
        
        result = EvolutionResult(states, times)
        return result

    def evolve_open(self, initial_state, n_steps=None):
        """
        Phi-scaled evolution for open systems using qutip's master equation solver.
        """
        steps = n_steps if n_steps is not None else self.config.get('n_steps', 10)
        tlist = np.linspace(0, steps, steps + 1)
        
        if initial_state.isket:
            rho0 = ket2dm(initial_state)
        else:
            rho0 = initial_state
        
        # Build effective Hamiltonian
        H_eff = sum(self.get_scaled_hamiltonian(idx) for idx in range(steps))
        
        mesolve_result = mesolve(
            H_eff,
            rho0,
            tlist,
            self.c_ops,
            options=Options(store_states=True)
        )
        
        # Create our EvolutionResult
        result = EvolutionResult(mesolve_result.states, tlist)
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
        return result
