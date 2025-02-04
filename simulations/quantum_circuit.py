import numpy as np
from qutip import Qobj, ket2dm, mesolve, Options
from .quantum_state import positivity_projection
from .config import load_config  # Corrected relative import

class ClosedResult:
    """
    Custom class to store results from evolve_closed method.
    """
    def __init__(self, states, times):
        self.states = states
        self.times = times


#################################################
# StandardCircuit
#################################################
class StandardCircuit:
    """
    Uniform approach over total_time with n_steps.
    - base_hamiltonian
    - c_ops for open system
    - 'evolve_closed' => Trotter steps
    - 'evolve_open' => QuTiP mesolve
    """
    def __init__(self, base_hamiltonian, total_time=None, n_steps=None, c_ops=None):
        self.base_hamiltonian = base_hamiltonian
        self.config = load_config()
        # Give precedence to passed parameters over config values
        self.total_time = total_time if total_time is not None else self.config.get('total_time', 1.0)
        self.n_steps = n_steps if n_steps is not None else self.config.get('n_steps', 10)
        self.c_ops = c_ops if c_ops is not None else self.config.get('c_ops', [])

    def evolve_closed(self, initial_state, n_steps=None):
        """
        Trotter approach: dt = total_time / n_steps, 
        U_dt = exp(-i dt H0).
        """
        # Use passed n_steps first, then instance n_steps
        steps = n_steps if n_steps is not None else self.n_steps
        dt = self.total_time / steps
        U_dt = (-1j * dt * self.base_hamiltonian).expm()

        if initial_state.isket:
            rho = ket2dm(initial_state)
        else:
            rho = initial_state

        all_states = []
        for _ in range(steps):
            rho = U_dt * rho * U_dt.dag()
            all_states.append(rho)
        result = ClosedResult(states=all_states, times=[dt * (i + 1) for i in range(steps)])
        return result

    def evolve_open(self, initial_state):
        """
        mesolve from t=0..total_time with n_steps+1 points.
        """
        if initial_state.isket:
            rho0 = ket2dm(initial_state)
        else:
            rho0 = initial_state

        tlist = np.linspace(0, self.total_time, self.n_steps + 1)
        result = mesolve(
            self.base_hamiltonian,  # H as positional argument
            rho0,                   # rho0 as positional argument
            tlist,                  # tlist as positional argument
            self.c_ops,             # c_ops as positional argument
            e_ops=[],               # Provided as keyword argument with empty list
            options=Options(store_states=True)  # Provided as keyword argument
        )
        return result


#################################################
# PhiScaledCircuit
#################################################
class PhiScaledCircuit:
    """
    φ-based expansions:
      scale_n = (α^n * exp(-βn)) / (φ^n).
    Closed evolution: discrete repeated steps.
    Open evolution: approximate H_eff by summing log(U_n).
    """
    def __init__(self, base_hamiltonian, alpha=None, beta=None, c_ops=None, positivity=None):
        self.base_hamiltonian = base_hamiltonian
        self.config = load_config()
        self.alpha = alpha if alpha is not None else self.config.get('alpha', 1.0)
        self.beta = beta if beta is not None else self.config.get('beta', 0.1)
        self.c_ops = c_ops if c_ops is not None else self.config.get('c_ops', [])
        self.positivity = positivity if positivity is not None else self.config.get('positivity', False)
        
        # Check contraction condition: assume Lipschitz constant L = 1 for simplicity.
        L = 1.0
        if self.alpha * np.exp(-self.beta) >= 1.0 / (L + 1):
            raise ValueError("Chosen parameters (alpha, beta) may violate contraction conditions.")
            
    def phi_scaled_unitary(self, step_idx):
        from constants import PHI  # Import PHI from constants module
        # Incorporate the φ-scaling explicitly:
        # scale = (alpha^n * exp(-βn)) / (PHI^n)
        scale = (self.alpha ** step_idx) * np.exp(-self.beta * step_idx) / (PHI ** step_idx)
        return (-1j * scale * self.base_hamiltonian).expm()

    def evolve_closed(self, initial_state, n_steps=None):
        # Give precedence to passed parameter over config
        steps = n_steps if n_steps is not None else self.config.get('n_steps', 10)
        if initial_state.isket:
            rho = ket2dm(initial_state)
        else:
            rho = initial_state

        all_states = []
        for idx in range(steps):
            U_n = self.phi_scaled_unitary(idx)
            rho = U_n * rho * U_n.dag()
            if self.positivity:
                rho = positivity_projection(rho)
            all_states.append(rho)
        result = ClosedResult(states=all_states, times=list(range(steps)))
        return result

    def evolve_open(self, initial_state, n_steps=None, tlist=None):
        """
        Approximate H_eff = Σ (i/scale_n) * log(U_n).
        Then perform mesolve on H_eff.
        """
        if initial_state.isket:
            rho0 = ket2dm(initial_state)
        else:
            rho0 = initial_state

        steps = n_steps if n_steps is not None else self.config.get('n_steps', 10)
        if tlist is None:
            tlist = np.linspace(0, self.config.get('total_time', 1.0), steps + 1)
        
        H_eff = self.build_approx_total_hamiltonian(steps)
        result = mesolve(
            H_eff,                            # H_eff as positional argument
            rho0,                             # rho0 as positional argument
            tlist,                            # tlist as positional argument
            self.c_ops,                       # c_ops as positional argument
            e_ops=[],                         # Provided as keyword argument with empty list
            options=Options(store_states=True) # Provided as keyword argument
        )
        return result

    def build_approx_total_hamiltonian(self, n_steps=None):
        from qutip import Qobj
        import numpy as np
        from constants import PHI  # Import PHI constant
        steps = n_steps if n_steps is not None else self.config.get('n_steps', 10)
        dim = self.base_hamiltonian.shape[0]
        H_eff = Qobj(np.zeros((dim, dim), dtype=complex), dims=self.base_hamiltonian.dims)
        for idx in range(steps):
            U_n = self.phi_scaled_unitary(idx)
            scale = (self.alpha ** idx) * np.exp(-self.beta * idx) / (PHI ** idx)
            logU = U_n.logm()
            H_eff += (1.0j / scale) * logU
        return H_eff


#################################################
# FibonacciBraidingCircuit
#################################################
class FibonacciBraidingCircuit:
    """
    For topological anyons (Fibonacci).
    We store braids (2D or 3D Qobjs) and apply them in order to a state.
    """
    def __init__(self, braids=None):
        self.braids = braids if braids else []

    def add_braid(self, Bop):
        self.braids.append(Bop)

    def evolve(self, initial_state):
        """
        final_psi = (B_n ... B_1) * initial_state
        """
        psi = initial_state
        for B in self.braids:
            psi = B * psi
        return psi
