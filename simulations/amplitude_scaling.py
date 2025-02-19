"""
Amplitude scaling implementation using qutip's pulse compilation features.
"""

import numpy as np
from qutip import Qobj, sigmax, sigmay, sigmaz, basis
from qutip_qip.compiler import GateCompiler, Instruction
from qutip_qip.circuit import QubitCircuit
from qutip_qip.pulse import Pulse

class AmplitudeScalingCompiler(GateCompiler):
    """
    Custom compiler for amplitude-scaled quantum operations.
    """
    def __init__(self, num_qubits, params=None):
        super().__init__(num_qubits, params=params)
        self.gate_compiler = {
            "RX": self._compile_rx,
            "RY": self._compile_ry,
            "RZ": self._compile_rz,
            "CUSTOM": self._compile_custom
        }
    
    def _compile_rx(self, gate, args):
        """Compile RX gate with scaled amplitude."""
        q = gate.targets[0]
        angle = gate.arg_value
        H = sigmax()
        coeff = np.array([angle/2])
        tlist = np.array([0, args["duration"]])
        return [Pulse(H, tlist, coeff, args["scaling_factor"])]
    
    def _compile_ry(self, gate, args):
        """Compile RY gate with scaled amplitude."""
        q = gate.targets[0]
        angle = gate.arg_value
        H = sigmay()
        coeff = np.array([angle/2])
        tlist = np.array([0, args["duration"]])
        return [Pulse(H, tlist, coeff, args["scaling_factor"])]
    
    def _compile_rz(self, gate, args):
        """Compile RZ gate with scaled amplitude."""
        q = gate.targets[0]
        angle = gate.arg_value
        H = sigmaz()
        coeff = np.array([angle/2])
        tlist = np.array([0, args["duration"]])
        return [Pulse(H, tlist, coeff, args["scaling_factor"])]
    
    def _compile_custom(self, gate, args):
        """Compile custom gate with scaled amplitude."""
        q = gate.targets[0]
        H = gate.arg_value  # Custom Hamiltonian
        coeff = np.array([1.0])
        tlist = np.array([0, args["duration"]])
        return [Pulse(H, tlist, coeff, args["scaling_factor"])]

def compile_pulse(H_control, total_time=10.0, steps=100, scaling_factor=1.0):
    """
    Compile control Hamiltonian into time-dependent pulse coefficients.
    
    Parameters:
    - H_control (Qobj): Control Hamiltonian operator
    - total_time (float): Total duration of the pulse
    - steps (int): Number of time steps
    - scaling_factor (float): Factor to scale pulse amplitudes
    
    Returns:
    - times (numpy.ndarray): Array of time points
    - amplitudes (numpy.ndarray): Array of amplitude coefficients
    """
    # Create a circuit with a custom gate using the control Hamiltonian
    qc = QubitCircuit(1)
    qc.add_gate("CUSTOM", targets=[0], arg_value=H_control)
    
    # Create compiler instance
    compiler = AmplitudeScalingCompiler(1)
    
    # Compile the circuit
    duration = total_time / steps
    pulses = compiler.compile(qc, args={
        "duration": duration,
        "scaling_factor": scaling_factor
    })
    
    # Extract times and amplitudes
    times = np.linspace(0, total_time, steps)
    amplitudes = np.array([pulse.coeff[0] * scaling_factor for pulse in pulses])
    
    return times, amplitudes

def simulate_amplitude_scaling(H_control, total_time=10.0, steps=100, scaling_factor=1.0):
    """
    Simulates amplitude scaling for a given control Hamiltonian.
    
    Parameters:
    - H_control (Qobj): Control Hamiltonian operator
    - total_time (float): Total duration of the pulse
    - steps (int): Number of time steps
    - scaling_factor (float): Factor to scale the amplitude
    
    Returns:
    - times (numpy.ndarray): Array of time points
    - scaled_amplitudes (numpy.ndarray): Scaled amplitude coefficients
    """
    # Get base pulse sequence
    times, amplitudes = compile_pulse(H_control, total_time, steps)
    
    # Scale amplitudes
    scaled_amplitudes = scaling_factor * amplitudes
    
    return times, scaled_amplitudes

def get_pulse_sequence(H_control, total_time=10.0, steps=100, scaling_factor=1.0):
    """
    Get a complete pulse sequence including Hamiltonian and coefficients.
    
    Parameters:
    - H_control (Qobj): Control Hamiltonian operator
    - total_time (float): Total duration of the pulse
    - steps (int): Number of time steps
    - scaling_factor (float): Factor to scale the amplitude
    
    Returns:
    - H_list (list): List of Hamiltonians
    - coeff_list (list): List of coefficient arrays
    - tlist (numpy.ndarray): Array of time points
    """
    times, amplitudes = compile_pulse(H_control, total_time, steps, scaling_factor)
    
    # Create time-dependent Hamiltonian list
    H_list = [H_control]
    coeff_list = [amplitudes]
    tlist = times
    
    return H_list, coeff_list, tlist
