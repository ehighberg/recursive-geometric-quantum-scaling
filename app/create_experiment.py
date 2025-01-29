# app/create_experiment.py
"""
A minimal sample 'control file' for a Streamlit-based interface,
showing how to choose Classic vs. φ-Scaled experiments,
and noise parameters, etc.
"""

import streamlit as st
import numpy as np
from qutip import sigmaz, sigmax, tensor, qeye
from simulations.quantum_circuit import StandardCircuit, PhiScaledCircuit
from simulations.quantum_state import (state_zero, state_plus, state_ghz, state_w)

def create_experiment():
    st.title("Create Quantum Experiment")

    mode = st.selectbox("Choose simulation mode", ["Classic", "Phi-Scaled"])
    num_qubits = st.number_input("Number of Qubits", 1, 5, 1)

    if mode=="Classic":
        total_time = st.slider("Total Time", 0.1, 10.0, 5.0)
        n_steps = st.slider("# Trotter Steps", 1, 100, 50)
        error_prob = st.slider("Error Probability", 0.0, 1.0, 0.05)
        if st.button("Run Classic"):
            # Example Hamiltonian
            if num_qubits==1:
                H0 = sigmaz()
                psi_init = state_plus(1)
            else:
                H0 = tensor(sigmaz(), qeye(2))
                psi_init = state_zero(num_qubits)

            circuit = StandardCircuit(H0, total_time, n_steps, error_prob=error_prob)
            result = circuit.evolve_closed(psi_init)
            st.session_state['simulation_results'] = result
            st.success("Classic experiment completed.")
    else:
        alpha = st.slider("Alpha", 0.1, 2.0, 1.0)
        beta  = st.slider("Beta", 0.0, 1.0, 0.1)
        n_steps_phi = st.slider("Recursion Depth", 1, 20, 5)
        T = st.slider("Max Time", 0.1, 10.0, 5.0)
        noise_strength = st.slider("Noise Strength", 0.0, 0.1, 0.01)
        noise_type = st.selectbox("Noise Type", ["gaussian", "uniform"])

        if st.button("Run φ-Scaled"):
            if num_qubits==1:
                H0 = sigmaz()
                psi_init = state_plus(1)
            else:
                H0 = tensor(sigmaz(), qeye(2)) + 0.2*tensor(qeye(2), sigmax())
                psi_init = state_zero(num_qubits)

            circuit = PhiScaledCircuit(
                base_hamiltonian=H0,
                alpha=alpha,
                beta=beta,
                noise_strength=noise_strength,
                noise_type=noise_type
            )
            tlist = np.linspace(0, T, 200)
            result = circuit.evolve_state(psi_init, n_steps_phi, tlist)
            st.session_state['simulation_results'] = result
            st.success("φ-Scaled experiment completed.")

if __name__=="__main__":
    import streamlit as st
    if 'simulation_results' not in st.session_state:
        st.session_state['simulation_results'] = None
    create_experiment()
