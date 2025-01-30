# app/create_experiment.py
import streamlit as st
from simulations.scripts.evolve_state import (
    run_standard_state_evolution,
    run_phi_scaled_state_evolution
)

def create_experiment():
    st.title("Quantum Experiment Setup")

    mode = st.selectbox("Experiment Mode", ["Standard State Evolve", "Phi-Scaled State Evolve"])

    if mode == "Standard State Evolve":
        num_qubits = st.slider("Number of Qubits", 1, 4, 2)
        state_label = st.selectbox("State Label", ["zero", "one", "plus", "ghz", "w"])
        total_time = st.slider("Total Time", 0.1, 10.0, 5.0)
        n_steps = st.slider("Steps", 1, 100, 50)
        if st.button("Run"):
            result = run_standard_state_evolution(
                num_qubits=num_qubits,
                state_label=state_label,
                total_time=total_time,
                n_steps=n_steps
            )
            st.session_state["simulation_results"] = result
            st.success("Standard state evolution finished.")

    else:
        num_qubits = st.slider("Number of Qubits", 1, 4, 2)
        state_label = st.selectbox("State Label", ["zero", "one", "plus", "ghz", "w"])
        alpha = st.slider("Alpha", 0.1, 2.0, 1.0)
        beta  = st.slider("Beta", 0.0, 1.0, 0.1)
        phi_steps = st.slider("Phi Steps", 1, 20, 5)
        if st.button("Run φ"):
            result = run_phi_scaled_state_evolution(
                num_qubits=num_qubits,
                state_label=state_label,
                phi_steps=phi_steps,
                alpha=alpha,
                beta=beta
            )
            st.session_state["simulation_results"] = result
            st.success("φ-Scaled state evolution completed.")


if __name__=="__main__":
    import streamlit as st
    if 'simulation_results' not in st.session_state:
        st.session_state['simulation_results'] = None
    create_experiment()
