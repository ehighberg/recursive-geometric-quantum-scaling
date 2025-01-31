# app.py
import streamlit as st
from simulations.scripts.evolve_state import (
    run_standard_state_evolution,
    run_phi_scaled_state_evolution
)
from simulations.scripts.evolve_circuit import (
    run_standard_twoqubit_circuit,
    run_phi_scaled_twoqubit_circuit,
    run_fibonacci_braiding_circuit
)

st.set_page_config(
    page_title="Quantum Simulation and Analysis Tool",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔮 Quantum Simulation and Analysis Tool")

mode = st.selectbox("Select a pipeline", [
    "State -> Standard",
    "State -> Phi-Scaled",
    "Circuit -> Standard 2Q",
    "Circuit -> Phi-Scaled 2Q",
    "Circuit -> Fibonacci Braiding"
])

if st.button("Run Simulation"):
    if mode=="State -> Standard":
        num_qubits = st.slider("Number of Qubits", 1, 4, 2)
        state_label = st.selectbox("State Label", ["zero", "one", "plus", "ghz", "w"])
        total_time = st.slider("Total Time", 0.1, 10.0, 5.0)
        n_steps = st.slider("Steps", 1, 100, 50)
        res = run_standard_state_evolution(num_qubits, state_label, total_time, n_steps)
        st.write("Got final state:", res.states[-1])
    elif mode=="State -> Phi-Scaled":
        num_qubits = st.slider("Number of Qubits", 1, 4, 2)
        state_label = st.selectbox("State Label", ["zero", "one", "plus", "ghz", "w"])
        alpha = st.slider("Alpha", 0.1, 2.0, 1.0)
        beta  = st.slider("Beta", 0.0, 1.0, 0.1)
        phi_steps = st.slider("Phi Steps", 1, 20, 5)
        res = run_phi_scaled_state_evolution(num_qubits, state_label, phi_steps, alpha, beta)
        st.write("Got final state:", res.states[-1])
    elif mode=="Circuit -> Standard 2Q":
        res = run_standard_twoqubit_circuit()
        st.write("Got final 2Q state:", res.states[-1])
    elif mode=="Circuit -> Phi-Scaled 2Q":
        res = run_phi_scaled_twoqubit_circuit()
        st.write("Got final 2Q φ-scaled state:", res.states[-1])
    else:
        fib_final= run_fibonacci_braiding_circuit()
        st.write("Fibonacci braiding final state:", fib_final)
