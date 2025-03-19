"""
Utility functions for quantum simulations and analysis.

This module provides standardized implementations of common quantum operations,
ensuring consistency across the codebase. It includes functions for eigenvalue computation,
quantum state evolution, and Hamiltonian manipulation.

All functions in this module apply scaling factors EXACTLY ONCE.
"""

import numpy as np
import warnings
from qutip import Qobj, Options, sesolve, mesolve, sigmaz, sigmax, tensor, qeye
from constants import PHI

class HamiltonianFactory:
    """
    Factory class for creating and manipulating quantum Hamiltonians.

    This class ensures that scaling factors are applied consistently
    and EXACTLY ONCE to all Hamiltonians. It provides standardized
    methods for constructing different types of Hamiltonians.
    """

    @staticmethod
    def create_hamiltonian(hamiltonian_type, num_qubits=1, scaling_factor=1.0, parameters=None):
        """
        Create a Hamiltonian with proper scaling applied EXACTLY ONCE.

        Parameters:
        -----------
        hamiltonian_type : str
            Type of Hamiltonian to create. Options: 'z', 'x', 'ising', 'heisenberg', 'custom'
        num_qubits : int, optional
            Number of qubits in the system
        scaling_factor : float, optional
            Factor to scale the Hamiltonian (applied ONCE)
        parameters : dict, optional
            Additional parameters for specific Hamiltonian types

        Returns:
        --------
        Qobj
            QuTiP quantum object representing the scaled Hamiltonian
        """
        # Create base unscaled Hamiltonian
        H_unscaled = HamiltonianFactory._create_base_hamiltonian(
            hamiltonian_type, num_qubits, parameters
        )

        # Apply scaling factor EXACTLY ONCE
        return scaling_factor * H_unscaled

    @staticmethod
    def create_parameterized_hamiltonian(hamiltonian_type, parameter_name, parameter_values,
                                        num_qubits=1, scaling_factor=1.0, additional_params=None):
        """
        Create a parameterized series of Hamiltonians for sweeps.

        Parameters:
        -----------
        hamiltonian_type : str
            Type of Hamiltonian to create. Options: 'z', 'x', 'ising', 'heisenberg', 'custom'
        parameter_name : str
            Name of the parameter to vary
        parameter_values : array_like
            Values of the parameter to sweep over
        num_qubits : int, optional
            Number of qubits in the system
        scaling_factor : float, optional
            Factor to scale the Hamiltonian (applied ONCE)
        additional_params : dict, optional
            Additional fixed parameters

        Returns:
        --------
        list
            List of (parameter_value, scaled_hamiltonian) pairs
        """
        results = []

        # Create a Hamiltonian for each parameter value
        for value in parameter_values:
            # Set up parameters dictionary
            params = additional_params.copy() if additional_params else {}
            params[parameter_name] = value

            # Create base unscaled Hamiltonian
            H_unscaled = HamiltonianFactory._create_base_hamiltonian(
                hamiltonian_type, num_qubits, params
            )

            # Apply scaling factor EXACTLY ONCE
            H_scaled = scaling_factor * H_unscaled

            # Add to results
            results.append((value, H_scaled))

        return results

    @staticmethod
    def create_topological_hamiltonian(base_type, momentum_parameter, momentum_values,
                                       num_qubits=2, scaling_factor=1.0, coupling=0.1):
        """
        Create Hamiltonians for topological invariant calculations with consistent scaling.

        Parameters:
        -----------
        base_type : str
            Base Hamiltonian type ('z', 'x', 'ising', etc.)
        momentum_parameter : str
            Parameter name for momentum ('k', 'kx', 'ky', etc.)
        momentum_values : array_like
            Values of momentum to sweep over
        num_qubits : int, optional
            Number of qubits in the system
        scaling_factor : float, optional
            Factor to scale the entire Hamiltonian (applied ONCE to the combined Hamiltonian)
        coupling : float, optional
            Strength of momentum coupling term relative to base Hamiltonian

        Returns:
        --------
        list
            List of (momentum_value, scaled_hamiltonian) pairs
        """
        results = []

        # Create base Hamiltonian (unscaled)
        H_base = HamiltonianFactory._create_base_hamiltonian(base_type, num_qubits)

        # Create momentum coupling operator (unscaled)
        H_coupling = HamiltonianFactory._create_momentum_coupling(momentum_parameter, num_qubits)

        # Create a Hamiltonian for each momentum value
        for k in momentum_values:
            # Combine base and momentum terms (still unscaled)
            H_combined = H_base + coupling * k * H_coupling

            # Apply scaling factor EXACTLY ONCE to the combined Hamiltonian
            H_scaled = scaling_factor * H_combined

            # Add to results
            results.append((k, H_scaled))

        return results

    @staticmethod
    def _create_base_hamiltonian(hamiltonian_type, num_qubits, parameters=None):
        """
        Create base (unscaled) Hamiltonian of the specified type.

        Parameters:
        -----------
        hamiltonian_type : str
            Type of Hamiltonian to create
        num_qubits : int
            Number of qubits in the system
        parameters : dict, optional
            Additional parameters

        Returns:
        --------
        Qobj
            Unscaled Hamiltonian
        """
        if parameters is None:
            parameters = {}

        if hamiltonian_type == "z":
            if num_qubits == 1:
                return sigmaz()
            else:
                H = 0
                for i in range(num_qubits):
                    op_list = [qeye(2) for _ in range(num_qubits)]
                    op_list[i] = sigmaz()
                    H += tensor(op_list)
                return H

        elif hamiltonian_type == "x":
            if num_qubits == 1:
                return sigmax()
            else:
                H = 0
                for i in range(num_qubits):
                    op_list = [qeye(2) for _ in range(num_qubits)]
                    op_list[i] = sigmax()
                    H += tensor(op_list)
                return H

        elif hamiltonian_type == "ising":
            if num_qubits == 1:
                return sigmaz()
            else:
                H = 0
                # ZZ coupling terms
                for i in range(num_qubits-1):
                    op_list = [qeye(2) for _ in range(num_qubits)]
                    op_list[i] = sigmaz()
                    op_list[i+1] = sigmaz()
                    H += tensor(op_list)

                # X field terms
                field_strength = parameters.get('field_strength', 0.5)
                for i in range(num_qubits):
                    op_list = [qeye(2) for _ in range(num_qubits)]
                    op_list[i] = sigmax()
                    H += field_strength * tensor(op_list)

                return H

        elif hamiltonian_type == "heisenberg":
            if num_qubits == 1:
                return sigmaz()
            else:
                # Import sigmay since it's not imported at the top
                from qutip import sigmay
                H = 0

                # Coupling terms (XX + YY + ZZ)
                for i in range(num_qubits-1):
                    # XX coupling
                    op_list_x = [qeye(2) for _ in range(num_qubits)]
                    op_list_x[i] = sigmax()
                    op_list_x[i+1] = sigmax()
                    H += tensor(op_list_x)

                    # YY coupling
                    op_list_y = [qeye(2) for _ in range(num_qubits)]
                    op_list_y[i] = sigmay()
                    op_list_y[i+1] = sigmay()
                    H += tensor(op_list_y)

                    # ZZ coupling
                    op_list_z = [qeye(2) for _ in range(num_qubits)]
                    op_list_z[i] = sigmaz()
                    op_list_z[i+1] = sigmaz()
                    H += tensor(op_list_z)

                # Optional field term
                if 'field' in parameters and parameters['field'] > 0:
                    field = parameters['field']
                    field_dir = parameters.get('field_dir', 'z')

                    for i in range(num_qubits):
                        op_list = [qeye(2) for _ in range(num_qubits)]
                        if field_dir == 'x':
                            op_list[i] = sigmax()
                        elif field_dir == 'y':
                            op_list[i] = sigmay()
                        else:  # default to z
                            op_list[i] = sigmaz()

                        H += field * tensor(op_list)

                return H

        elif hamiltonian_type == "custom":
            if 'operator' in parameters:
                return parameters['operator']
            else:
                raise ValueError("Custom Hamiltonian requires 'operator' parameter")

        else:
            raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}")

    @staticmethod
    def _create_momentum_coupling(momentum_parameter, num_qubits):
        """
        Create momentum coupling operator for topological calculations.

        Parameters:
        -----------
        momentum_parameter : str
            Parameter name for momentum
        num_qubits : int
            Number of qubits in the system

        Returns:
        --------
        Qobj
            Momentum coupling operator
        """
        if momentum_parameter in ['k', 'kx']:
            # 1D coupling along x-direction
            if num_qubits == 1:
                return sigmax()
            else:
                op_list = [qeye(2) for _ in range(num_qubits)]
                op_list[0] = sigmax()  # Couple to first qubit by default
                return tensor(op_list)

        elif momentum_parameter == 'ky':
            # Coupling along y-direction (for 2D topological systems)
            from qutip import sigmay
            if num_qubits == 1:
                return sigmay()
            else:
                op_list = [qeye(2) for _ in range(num_qubits)]
                op_list[1 if num_qubits > 1 else 0] = sigmay()  # Couple to second qubit if available
                return tensor(op_list)

        else:
            # Default coupling for other parameters
            op_list = [qeye(2) for _ in range(num_qubits)]
            op_list[0] = sigmax()  # Default to X coupling on first qubit
            return tensor(op_list)

def compute_eigenvalues(H):
    """
    Compute eigenvalues of a Hamiltonian safely, handling different object types.

    Parameters:
    -----------
    H : Qobj or numpy.ndarray
        Hamiltonian operator

    Returns:
    --------
    numpy.ndarray
        Eigenvalues of the Hamiltonian
    """
    if isinstance(H, Qobj):
        # Use QuTiP's eigenenergies method for Qobj
        return H.eigenenergies()
    elif isinstance(H, np.ndarray):
        # For numpy arrays, check if it's a square matrix
        if H.ndim == 2 and H.shape[0] == H.shape[1]:
            return np.linalg.eigvalsh(H)  # Use eigvalsh for Hermitian matrices
        else:
            raise ValueError("Hamiltonian must be a square matrix")
    else:
        raise TypeError(f"Unsupported Hamiltonian type: {type(H)}")

def evolve_quantum_state(
    initial_state,
    hamiltonian,
    times,
    scaling_factor=1.0,
    noise_config=None,
    phi_recursive=False,
    recursion_depth=3,
    **kwargs
):
    """
    Universal quantum state evolution function that unifies functionality.

    Parameters:
    -----------
    initial_state : Qobj
        Initial quantum state
    hamiltonian : Qobj
        System Hamiltonian (unscaled)
    times : array_like
        Time points for evolution
    scaling_factor : float, optional
        Factor to scale the Hamiltonian (applied ONCE)
    noise_config : dict, optional
        Noise configuration for non-unitary evolution
    phi_recursive : bool, optional
        Whether to use phi-recursive scaling
    recursion_depth : int, optional
        Recursion depth for phi-recursive scaling
    **kwargs : dict
        Additional arguments

    Returns:
    --------
    result : object
        Result object containing evolution data
    """
    if phi_recursive:
        return _evolve_phi_recursive(
            initial_state,
            hamiltonian,
            times,
            scaling_factor,
            recursion_depth,
            noise_config,
            **kwargs
        )
    else:
        return _evolve_standard(
            initial_state,
            hamiltonian,
            times,
            scaling_factor,
            noise_config,
            **kwargs
        )

def _evolve_standard(
    initial_state,
    hamiltonian,
    times,
    scaling_factor=1.0,
    noise_config=None,
    **kwargs
):
    """
    Standard quantum state evolution with linear scaling.

    Parameters:
    -----------
    initial_state : Qobj
        Initial quantum state
    hamiltonian : Qobj
        System Hamiltonian (unscaled)
    times : array_like
        Time points for evolution
    scaling_factor : float, optional
        Factor to scale the Hamiltonian (applied ONCE)
    noise_config : dict, optional
        Noise configuration for non-unitary evolution
    **kwargs : dict
        Additional arguments

    Returns:
    --------
    result : object
        Result object containing evolution data
    """
    # Scale Hamiltonian ONCE
    H_scaled = scaling_factor * hamiltonian

    # Set up evolution options
    options = Options(store_states=True)
    if 'options' in kwargs:
        for key, value in kwargs['options'].items():
            setattr(options, key, value)

    # Set up measurement operators if needed
    e_ops = kwargs.get('e_ops', None)
    if e_ops is None and hasattr(initial_state, 'dims') and len(initial_state.dims[0]) > 0:
        # Default to measuring sigmaz on each qubit
        from qutip import tensor, qeye
        num_qubits = len(initial_state.dims[0])
        if num_qubits == 1:
            e_ops = [sigmaz()]
        else:
            e_ops = []
            for i in range(num_qubits):
                op_list = [qeye(2) for _ in range(num_qubits)]
                op_list[i] = sigmaz()
                e_ops.append(tensor(op_list))

    # Simulate quantum evolution
    if noise_config:
        # Handle pre-defined collapse operators if provided
        if 'c_ops' in noise_config:
            c_ops = noise_config['c_ops']
        else:
            # Initialize collapse operators list
            c_ops = []

            # For each qubit, add noise operators
            from qutip import tensor, qeye, sigmax
            num_qubits = len(initial_state.dims[0]) if hasattr(initial_state, 'dims') else 1

            for i in range(num_qubits):
                # Create operator lists for tensor products
                op_list_x = [qeye(2) for _ in range(num_qubits)]
                op_list_z = [qeye(2) for _ in range(num_qubits)]

                # Add T1 relaxation
                if noise_config.get('relaxation', 0) > 0:
                    op_list_x[i] = sigmax()
                    c_ops.append(np.sqrt(noise_config['relaxation']) * tensor(op_list_x))

                # Add T2 dephasing
                if noise_config.get('dephasing', 0) > 0:
                    op_list_z[i] = sigmaz()
                    c_ops.append(np.sqrt(noise_config['dephasing']) * tensor(op_list_z))

        # Increase nsteps for better integration with noise
        options.nsteps = 10000
        return mesolve(H_scaled, initial_state, times, c_ops, e_ops=e_ops, options=options)
    else:
        return sesolve(H_scaled, initial_state, times, e_ops=e_ops, options=options)

def _evolve_phi_recursive(
    initial_state,
    hamiltonian,
    times,
    scaling_factor=PHI,
    recursion_depth=3,
    noise_config=None,
    **kwargs
):
    """
    Phi-recursive quantum state evolution.

    Parameters:
    -----------
    initial_state : Qobj
        Initial quantum state
    hamiltonian : Qobj
        System Hamiltonian (unscaled)
    times : array_like
        Time points for evolution
    scaling_factor : float, optional
        Factor to scale the Hamiltonian (applied ONCE)
    recursion_depth : int, optional
        Recursion depth for phi-recursive scaling
    noise_config : dict, optional
        Noise configuration for non-unitary evolution
    **kwargs : dict
        Additional arguments

    Returns:
    --------
    result : object
        Result object containing evolution data
    """
    from simulations.scaled_unitary import get_phi_recursive_unitary

    # Set up measurement operators if needed
    e_ops = kwargs.get('e_ops', None)
    if e_ops is None and hasattr(initial_state, 'dims') and len(initial_state.dims[0]) > 0:
        # Default to measuring sigmaz on each qubit
        from qutip import tensor, qeye
        num_qubits = len(initial_state.dims[0])
        if num_qubits == 1:
            e_ops = [sigmaz()]
        else:
            e_ops = []
            for i in range(num_qubits):
                op_list = [qeye(2) for _ in range(num_qubits)]
                op_list[i] = sigmaz()
                e_ops.append(tensor(op_list))

    # Create phi-recursive unitaries for each time step
    unitaries = []
    for t in times:
        U = get_phi_recursive_unitary(hamiltonian, t, scaling_factor, recursion_depth)
        unitaries.append(U)

    # Manually evolve the state using the unitaries
    states = []
    current_state = initial_state

    for i, U in enumerate(unitaries):
        # Apply unitary evolution
        evolved_state = U * current_state

        # If noise is configured, apply noise effects manually
        if noise_config:
            # Convert to density matrix if it's a ket
            if evolved_state.isket:
                evolved_state = evolved_state * evolved_state.dag()

            # Apply dephasing noise (diagonal terms decay)
            if noise_config.get('dephasing', 0) > 0:
                dephasing = noise_config['dephasing']
                # For each element in the density matrix
                data = evolved_state.full()
                for j in range(data.shape[0]):
                    for k in range(data.shape[1]):
                        if j != k:  # Off-diagonal elements
                            # Apply exponential decay to off-diagonal elements
                            data[j, k] *= np.exp(-dephasing * times[i])

                # Create new density matrix with decayed elements
                evolved_state = Qobj(data, dims=evolved_state.dims)

            # Apply relaxation noise (population decay to ground state)
            if noise_config.get('relaxation', 0) > 0:
                relaxation = noise_config['relaxation']
                # For each element in the density matrix
                data = evolved_state.full()
                # Diagonal elements decay toward ground state
                for j in range(1, data.shape[0]):  # Skip ground state
                    # Population decay
                    decay_factor = np.exp(-relaxation * times[i])
                    # Population transfers to ground state
                    ground_transfer = (1 - decay_factor) * data[j, j]
                    # Update diagonal elements
                    data[j, j] *= decay_factor
                    data[0, 0] += ground_transfer

                # Create new density matrix with decayed populations
                evolved_state = Qobj(data, dims=evolved_state.dims)

        # Add state to list
        states.append(evolved_state)

    # Create a custom result object
    class CustomResult:
        def __init__(self):
            self.times = None
            self.states = None
            self.e_ops = None
            self.options = {}
            self.expect = []

    result = CustomResult()
    result.times = times
    result.states = states
    result.e_ops = e_ops

    # Compute expectation values if e_ops are provided
    if e_ops:
        result.expect = []
        for op in e_ops:
            expect_values = []
            for state in states:
                # Handle the case where the result is already a complex number
                expectation = state.dag() * op * state
                if hasattr(expectation, 'tr'):
                    # If it's a QuTiP object with a trace method
                    expectation = expectation.tr()
                # Ensure the result is a real number if it's supposed to be
                if isinstance(expectation, complex) and abs(expectation.imag) < 1e-10:
                    expectation = expectation.real
                expect_values.append(expectation)
            result.expect.append(np.array(expect_values))

    # Add metadata
    result.scaling_factor = scaling_factor
    result.recursion_depth = recursion_depth

    return result

# Legacy function aliases for backward compatibility
def run_quantum_evolution(*args, **kwargs):
    """
    DEPRECATED: Use evolve_quantum_state instead.

    This function is maintained for backward compatibility.
    """
    warnings.warn(
        "run_quantum_evolution is deprecated. Use evolve_quantum_state instead.",
        DeprecationWarning, stacklevel=2
    )
    return evolve_quantum_state(*args, **kwargs)

def evolve_state_fixed(*args, **kwargs):
    """
    DEPRECATED: Use evolve_quantum_state instead.

    This function is maintained for backward compatibility.
    """
    warnings.warn(
        "evolve_state_fixed is deprecated. Use evolve_quantum_state instead.",
        DeprecationWarning, stacklevel=2
    )
    return evolve_quantum_state(*args, **kwargs)
