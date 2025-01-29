# simulations/quantum_circuit.py

import numpy as np
from qutip import Qobj, ket2dm, Result, Options
from .quantum_state import positivity_projection  # for PSD-projection if needed

##############################################################################
# 1. StandardCircuit (Classic)
##############################################################################
class StandardCircuit:
    """
    A reference circuit applying uniform pulses (non-φ-scaled).
    - base_hamiltonian
    - total_time
    - n_steps
    - c_ops (optional) for open-system
    - error_prob: chance of random error injection per step
    - pulse_shape: placeholder for potential advanced scheduling
    """
    def __init__(self, base_hamiltonian, total_time, n_steps,
                 c_ops=None, error_prob=0.0, pulse_shape='square'):
        self.H0 = base_hamiltonian
        self.total_time = total_time
        self.n_steps = n_steps
        self.c_ops = c_ops if c_ops else []
        self.dt = total_time / n_steps
        self.error_prob = error_prob
        self.pulse_shape = pulse_shape

    def evolve_closed(self, initial_state):
        from qutip import ket2dm, Qobj
        if initial_state.isket:
            rho0 = ket2dm(initial_state)
        else:
            rho0 = initial_state

        U_dt = (-1j * self.dt * self.H0).expm()
        all_states = []
        rho = rho0.copy()

        import numpy as np
        for _ in range(self.n_steps):
            rho = U_dt * rho * U_dt.dag()
            # simple error injection
            if np.random.rand() < self.error_prob:
                shape_ = rho.shape
                dims_  = rho.dims
                error_mat = np.random.rand(shape_[0], shape_[1]) * 0.01
                E_op = Qobj(error_mat, dims=dims_)
                rho = rho + E_op
            all_states.append(rho)

        result = Result()
        result.states = all_states
        result.times = [self.dt*(i+1) for i in range(self.n_steps)]
        return result

    def evolve_open(self, initial_state):
        """
        Open-system approach using mesolve with c_ops.
        """
        from qutip import mesolve, Options
        import numpy as np
        if initial_state.isket:
            rho0 = ket2dm(initial_state)
        else:
            rho0 = initial_state

        tlist = np.linspace(0, self.total_time, self.n_steps+1)
        result = mesolve(self.H0, rho0, tlist, self.c_ops, e_ops=[],
                         options=Options(store_states=True))
        # For a more advanced approach, you might do piecewise intervals
        return result


##############################################################################
# 2. PhiScaledCircuit (Fractal / φ-based)
##############################################################################
class PhiScaledCircuit:
    """
    A circuit applying φ-scaled unitaries for open/closed system.
    scale_n = alpha^n * exp(-beta*n).
    Includes optional noise injection (noise_strength, noise_type).
    """

    def __init__(self, base_hamiltonian,
                 alpha=1.0, beta=0.1,
                 noise_strength=0.0, noise_type='gaussian'):
        self.base_hamiltonian = base_hamiltonian
        self.alpha = alpha
        self.beta = beta
        self.phi = (1 + np.sqrt(5)) / 2
        self.lindblad_ops = []
        self.apply_positivity = False
        self.noise_strength = noise_strength
        self.noise_type = noise_type

    def phi_scaled_unitary(self, step_idx: int) -> Qobj:
        scale = (self.alpha ** step_idx)*np.exp(-self.beta*step_idx)
        return (-1j * scale * self.base_hamiltonian).expm()

    def build_circuit(self, n_steps: int) -> Qobj:
        from qutip import Qobj
        dim = self.base_hamiltonian.shape[0]
        circuit = Qobj(np.eye(dim), dims=self.base_hamiltonian.dims)
        for n in range(n_steps):
            U_n = self.phi_scaled_unitary(n)
            circuit = U_n @ circuit
        return circuit

    def evolve_state(self, initial_state: Qobj, n_steps: int, tlist=None) -> Result:
        from qutip import ket2dm, mesolve, Options, Result
        if initial_state.isket:
            rho0 = ket2dm(initial_state)
        else:
            rho0 = initial_state

        if self.lindblad_ops and (tlist is not None):
            # open system
            H_eff = self.build_approx_total_hamiltonian(n_steps)
            result = mesolve(H_eff, rho0, tlist, self.lindblad_ops, e_ops=[],
                             options=Options(store_states=True))
            if self.apply_positivity or self.noise_strength>0:
                for i, st in enumerate(result.states):
                    if self.apply_positivity:
                        result.states[i] = positivity_projection(st)
                    if self.noise_strength>0:
                        result.states[i] = self.apply_noise(result.states[i])
            return result
        else:
            # closed system
            if tlist is None:
                all_states = []
                rho = rho0.copy()
                for n in range(n_steps):
                    U_n = self.phi_scaled_unitary(n)
                    rho = U_n * rho * U_n.dag()
                    if self.apply_positivity:
                        rho = positivity_projection(rho)
                    if self.noise_strength>0:
                        rho = self.apply_noise(rho)
                    all_states.append(rho)
                res = Result()
                res.states = all_states
                res.times = list(range(n_steps))
                return res
            else:
                circuit = self.build_circuit(n_steps)
                all_states = []
                for _t in tlist:
                    rho_t = circuit * rho0 * circuit.dag()
                    if self.apply_positivity:
                        rho_t = positivity_projection(rho_t)
                    if self.noise_strength>0:
                        rho_t = self.apply_noise(rho_t)
                    all_states.append(rho_t)
                res = Result()
                res.states = all_states
                res.times = tlist
                return res

    def build_approx_total_hamiltonian(self, n_steps: int) -> Qobj:
        H_eff = 0*self.base_hamiltonian
        for n in range(n_steps):
            U_n = self.phi_scaled_unitary(n)
            scale = (self.alpha**n)*np.exp(-self.beta*n)
            H_n = (1.0j/scale)*U_n.logm()
            H_eff += H_n
        return H_eff

    def add_lindblad_operators(self, c_ops):
        if not isinstance(c_ops, list):
            c_ops = [c_ops]
        self.lindblad_ops = c_ops

    def enable_positivity_projection(self, enable=True):
        self.apply_positivity = enable

    def apply_noise(self, rho):
        """
        Simple toy noise injection. For real usage, define physically
        meaningful noise models.
        """
        from qutip import Qobj
        dims_ = rho.dims
        shape_ = rho.shape
        if self.noise_type=='gaussian':
            noise_array = np.random.normal(
                loc=0.0, scale=self.noise_strength, size=shape_
            )
        else:
            noise_array = np.random.uniform(
                low=-self.noise_strength, high=self.noise_strength, size=shape_
            )
        noise_op = Qobj(noise_array, dims=dims_)
        return rho + noise_op
