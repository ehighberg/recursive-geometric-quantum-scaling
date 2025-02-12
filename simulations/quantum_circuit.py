import numpy as np
from constants import PHI
from qutip import Qobj, ket2dm, mesolve, Options, sigmax, sigmay, sigmaz, destroy, tensor, qeye
from qutip_qip.noise import Noise
from qutip_qip.device import Processor
from .quantum_state import positivity_projection
from .config import load_config

class NoiseChannel:
    """
    Manages different types of noise channels for quantum circuits.
    Integrates with QuTiP-QIP's noise models.
    """
    def __init__(self, config=None):
        self.config = config if config else load_config()
        self.noise_config = self.config.get('noise', {})
        
    def get_collapse_operators(self, dims):
        """
        Generate collapse operators based on enabled noise channels.
        
        Parameters:
        -----------
        dims : int or list
            Dimension(s) of the quantum system
            
        Returns:
        --------
        list
            List of collapse operators (Qobj)
        """
        c_ops = []
        
        # Get system operators for single qubit
        a = destroy(2)  # Annihilation operator for 2-level system
        sm = a  # Lowering operator
        sp = a.dag()  # Raising operator
        sx = sigmax()
        sy = sigmay()
        sz = sigmaz()
        
        # For multi-qubit systems, apply to first qubit only
        if not isinstance(dims, int):
            dims = len(dims)
            # Create identity operators for other qubits
            id_list = [qeye(2) for _ in range(dims-1)]
            # Tensor product with first qubit operators
            a = tensor([a] + id_list)
            sm = tensor([sm] + id_list)
            sp = tensor([sp] + id_list)
            sx = tensor([sx] + id_list)
            sy = tensor([sy] + id_list)
            sz = tensor([sz] + id_list)
        
        # Depolarizing noise: equal probability of X, Y, Z errors
        if self.noise_config.get('depolarizing', {}).get('enabled', False):
            rate = self.noise_config['depolarizing']['rate']
            gamma = rate / 4.0  # Total rate divided by 4
            c_ops.extend([
                np.sqrt(gamma) * sx,
                np.sqrt(gamma) * sy,
                np.sqrt(gamma) * sz
            ])
        
        # Dephasing noise: loss of phase coherence
        if self.noise_config.get('dephasing', {}).get('enabled', False):
            rate = self.noise_config['dephasing']['rate']
            c_ops.append(np.sqrt(rate/2.0) * sz)
        
        # Amplitude damping: spontaneous emission
        if self.noise_config.get('amplitude_damping', {}).get('enabled', False):
            rate = self.noise_config['amplitude_damping']['rate']
            c_ops.append(np.sqrt(rate) * sm)
        
        # Thermal noise: both emission and absorption
        if self.noise_config.get('thermal', {}).get('enabled', False):
            nth = self.noise_config['thermal']['nth']
            rate = self.noise_config['thermal']['rate']
            # Emission and absorption rates
            gamma0 = rate * (nth + 1)  # Relaxation
            gamma1 = rate * nth        # Excitation
            c_ops.extend([
                np.sqrt(gamma0) * sm,  # Relaxation
                np.sqrt(gamma1) * sp   # Excitation
            ])
        
        return c_ops

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
class StandardCircuit(Processor):
    """
    Uniform approach over total_time with n_steps.
    - base_hamiltonian
    - c_ops for open system
    - 'evolve_closed' => Trotter steps
    - 'evolve_open' => QuTiP mesolve
    """
    def __init__(self, base_hamiltonian, total_time=None, n_steps=None, c_ops=None, noise_config=None):
        super().__init__(N=base_hamiltonian.shape[0])
        self.base_hamiltonian = base_hamiltonian
        self.config = load_config()
        # Give precedence to passed parameters over config values
        self.total_time = total_time if total_time is not None else self.config.get('total_time', 1.0)
        self.n_steps = n_steps if n_steps is not None else self.config.get('n_steps', 10)
        
        # Initialize noise channels
        self._noise = NoiseChannel(noise_config)
        base_c_ops = c_ops if c_ops is not None else self.config.get('c_ops', [])
        noise_c_ops = self._noise.get_collapse_operators(base_hamiltonian.shape[0])
        self.c_ops = base_c_ops + noise_c_ops
        
    @property
    def noise(self):
        """Get the noise channel"""
        return self._noise

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
        def schedule_pulse(self, pulse, time):
            """
            Schedule a pulse with specified parameters at a given time.
            """
            self.schedule.append({'pulse': pulse, 'time': time})

        def get_pulses_at_time(self, time):
            """
            Retrieve all pulses scheduled at a specific time.
            """
            return [p['pulse'] for p in self.schedule if p['time'] == time]

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
class ScaledCircuit(Processor):
    """
    Scale factor-based expansions:
      scale_n = (scale_factor^n).
    Closed evolution: discrete repeated steps.
    Open evolution: approximate H_eff by summing log(U_n).
    
    Parameters:
    -----------
    base_hamiltonian : Qobj
        Base Hamiltonian to scale
    scaling_factor : float, optional
        Factor to scale evolution (default=1.0)
    c_ops : list, optional
        Collapse operators for open system evolution
    positivity : bool, optional
        Whether to enforce positivity after each step
    """
    def __init__(self, base_hamiltonian, scaling_factor=None, c_ops=None, positivity=None, noise_config=None, total_time=None, n_steps=None):
        super().__init__(N=base_hamiltonian.shape[0])
        self.base_hamiltonian = base_hamiltonian
        self.config = load_config()
        self.scale_factor = scaling_factor if scaling_factor is not None else self.config.get('scale_factor', 1.0)
        self.positivity = positivity if positivity is not None else self.config.get('positivity', False)
        self.total_time = total_time if total_time is not None else self.config.get('total_time', 1.0)
        self.n_steps = n_steps if n_steps is not None else self.config.get('n_steps', 10)
        
        # Initialize noise channels with adjusted rates for scaled evolution
        self._noise = NoiseChannel(noise_config)
        base_c_ops = c_ops if c_ops is not None else self.config.get('c_ops', [])
        noise_c_ops = [op * np.sqrt(self.scale_factor) for op in self._noise.get_collapse_operators(base_hamiltonian.shape[0])]
        self.c_ops = base_c_ops + noise_c_ops
        
    @property
    def noise(self):
        """Get the noise channel"""
        return self._noise
            
    def scale_unitary(self, step_idx):
        scale = (self.scale_factor ** step_idx)
        return (-1j * scale * self.base_hamiltonian).expm()

    def evolve_closed(self, initial_state, n_steps=None):
        # Give precedence to passed parameter over config
        steps = n_steps if n_steps is not None else self.n_steps
        if initial_state.isket:
            rho = ket2dm(initial_state)
        else:
            rho = initial_state

        all_states = []
        for idx in range(steps):
            U_n = self.scale_unitary(idx)
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

        steps = n_steps if n_steps is not None else self.n_steps
        if tlist is None:
            tlist = np.linspace(0, self.total_time, steps + 1)
        
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
        # from constants import PHI  # Import PHI constant (no longer needed)
        steps = n_steps if n_steps is not None else self.config.get('n_steps', 10)
        dim = self.base_hamiltonian.shape[0]
        H_eff = Qobj(np.zeros((dim, dim), dtype=complex), dims=self.base_hamiltonian.dims)
        for idx in range(steps):
            U_n = self.scale_unitary(idx)
            scale = (self.scale_factor ** idx)
            logU = U_n.logm()
            H_eff += (1.0j / scale) * logU
        return H_eff

    class PulseScheduler:
        """
        Manages pulse scheduling within the simulation.
        """
        def __init__(self, schedule=None):
            self.schedule = schedule if schedule else []
        def add_pulse(self, pulse):
            self.schedule.append(pulse)
        def get_schedule(self):
            return self.schedule


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
