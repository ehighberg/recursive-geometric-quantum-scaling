"""
Amplitude scaling implementation using qutip's pulse compilation features.
"""

import numpy as np
from qutip import sigmax, sigmay, sigmaz
from qutip_qip.compiler import GateCompiler
from qutip_qip.circuit import QubitCircuit

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
        angle = gate.arg_value
        H = sigmax()
        coeffs = np.array([angle/2])
        tlist = np.array([0, args["duration"]])
        return [{"ham": H, "tlist": tlist, "coeffs": coeffs * args["scaling_factor"]}]
    
    def _compile_ry(self, gate, args):
        """Compile RY gate with scaled amplitude."""
        angle = gate.arg_value
        H = sigmay()
        coeffs = np.array([angle/2])
        tlist = np.array([0, args["duration"]])
        return [{"ham": H, "tlist": tlist, "coeffs": coeffs * args["scaling_factor"]}]
    
    def _compile_rz(self, gate, args):
        """Compile RZ gate with scaled amplitude."""
        angle = gate.arg_value
        H = sigmaz()
        coeffs = np.array([angle/2])
        tlist = np.array([0, args["duration"]])
        return [{"ham": H, "tlist": tlist, "coeffs": coeffs * args["scaling_factor"]}]
    
    def _compile_custom(self, gate, args):
        """Compile custom gate with scaled amplitude."""
        H = gate.arg_value  # Custom Hamiltonian
        coeffs = np.array([1.0])
        tlist = np.array([0, args["duration"]])
        return [{"ham": H, "tlist": tlist, "coeffs": coeffs * args["scaling_factor"]}]

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
    # Extract coefficients from pulse dictionaries
    amplitudes = np.array([pulse["coeffs"][0] for pulse in pulses])
    
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
