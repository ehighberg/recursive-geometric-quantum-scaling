import streamlit as st
import numpy as np

from qutip import Qobj, tensor, sigmax, sigmay, sigmaz, qeye

# Apply dark theme styling
st.set_page_config(
    page_title="Quantum Simulation and Analysis Tool",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #00B8D4;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #FAFAFA;
    }
    .stSelectbox>div>div>select {
        background-color: #262730;
        color: #FAFAFA;
    }
    .log-container {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        max-height: 200px;
        overflow-y: auto;
    }
    .log-info { color: #00B8D4; }
    .log-warning { color: #FFD740; }
    .log-error { color: #FF5252; }
    .parameter-section {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .analysis-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #00B8D4;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Quantum Simulation and Analysis Tool")
st.markdown("""
This tool allows you to simulate quantum states and circuits under various conditions and analyze their results. 
You can input parameters, run simulations, and view detailed logs and analysis results.
""")

if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def add_log_message(level, message):
    st.session_state.log_messages.append((level, message))
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages = st.session_state.log_messages[-100:]

def display_logs():
    if st.session_state.log_messages:
        with st.expander("Log Messages", expanded=True):
            for level, message in st.session_state.log_messages:
                st.markdown(f'<div class="log-{level}">{message}</div>', unsafe_allow_html=True)

# Sidebar: Noise Configuration
st.sidebar.header("Configuration")

with st.sidebar.expander("🔧 Advanced Configuration", expanded=False):
    st.markdown("### Noise Parameters")
    noise_strength = st.slider("Noise Strength", 0.0, 0.1, 0.01, help="Strength of environmental noise")
    noise_type = st.selectbox("Noise Type", ['gaussian', 'uniform'], help="Type of environmental noise")

# Initialize tabs
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 0

tabs = st.tabs([
    "State and Experiment Creation",
    "Analysis",
])

def on_tab_change():
    """Handles tab state changes."""
    try:
        # Get current tab index
        current_tab = st.session_state['active_tab']
        
        # Validate state requirements for analysis tab
        if current_tab == 1:
            if 'simulation_results' not in st.session_state:
                st.warning("⚠️ Please run an experiment first")
                st.session_state['active_tab'] = 0  # Return to state creation
    except Exception as e:
        add_log_message('error', f"Error handling tab change: {str(e)}")

with tabs[0]:
    st.header("Quantum state creation, quantum circuit setup, experiment definition")
    
    # Available states
    initial_states = {
        'standard': ['fock', 'coherent', 'squeezed', 'cat', 'bell', 'ghz', 'cluster'],
        'entanglement': ['bell', 'ghz', 'cluster']  # Multi-qubit states only
    }

    # Approach: "Classic" vs "φ-Scaled"
    approach = st.selectbox("Choose Simulation Approach", ["Classic", "Phi-Scaled"])

    # Qubit number
    num_qubits = st.number_input("Number of Qubits", min_value=1, max_value=5, value=1, step=1)

    # Let user pick an initial state from your "standard" or "entanglement" category
    # For demonstration, we unify them. Real code might do more specialized expansions.
    state_type = st.selectbox("State Type", ["standard", "entanglement"])

    if state_type == "standard":
        chosen_state = st.selectbox("Standard State", initial_states['standard'])
    else:
        chosen_state = st.selectbox("Entanglement State", initial_states['entanglement'])

    # Minimal handling: We'll map "fock" => |0>, "ghz" => GHZ, etc.
    from simulations.quantum_state import (
        state_zero, state_one, state_plus, state_ghz, state_w
    )

    def get_initial_state(chosen_state, num_qubits):
        if chosen_state in ["fock", "coherent", "squeezed", "cat"]:
            # e.g. treat them all as |0> for demonstration
            return state_zero(num_qubits)
        elif chosen_state == "bell":
            # If 2 qubits => a minimal GHZ-like approach, or you define a real bell state
            if num_qubits == 2:
                # For demonstration: same as GHZ(2)
                from qutip import ket2dm
                # Create a proper Bell state (|00> + |11>)/sqrt(2)
                return (state_zero(2) + state_one(2))/np.sqrt(2)
            else:
                return state_zero(num_qubits)
        elif chosen_state == "ghz":
            # if user picks GHZ => we need >=2 qubits
            if num_qubits<2:
                st.warning("GHZ requires >=2 qubits. Using 2 qubits.")
                num_qubits=2
            return state_ghz(num_qubits)
        elif chosen_state == "cluster":
            # Not implemented => fallback
            return state_zero(num_qubits)
        elif chosen_state == "w":
            return state_w(num_qubits)
        else:
            return state_zero(num_qubits)

    psi_init = get_initial_state(chosen_state, num_qubits)

    if approach == "Classic":
        st.subheader("Classic Circuit Options")
        total_time = st.slider("Total Time (Classic)", 0.1, 10.0, 5.0)
        n_steps = st.slider("Trotter Steps", 1, 100, 50)
        error_prob = st.slider("Error Probability", 0.0, 0.2, 0.05)
        if st.button("Run Classic Simulation"):
            from simulations.quantum_circuit import StandardCircuit
            from qutip import sigmaz, qeye, tensor

            # Simple example Hamiltonian: For single qubit => sigmaz()
            # For multiple => e.g. sigmaz() on qubit 1 + small sigmax on qubit 2
            if num_qubits==1:
                H0 = sigmaz()
            else:
                H0 = tensor(sigmaz(), qeye(2))  # you can expand if >2 qubits

            # Build circuit
            circuit = StandardCircuit(H0, total_time, n_steps,
                                      error_prob=error_prob)

            result = circuit.evolve_closed(psi_init)
            st.session_state['simulation_results'] = result
            st.success("Classic simulation done. Check the Analysis tab.")
    
    else:
        st.subheader("φ-Scaled Circuit Options")
        alpha = st.slider("Alpha", 0.1, 2.0, 1.0)
        beta  = st.slider("Beta", 0.0, 1.0, 0.1)
        fractal_steps = st.slider("Recursion Depth (n_steps)", 1, 20, 5)
        T = st.slider("Max Time (φ-scaled)", 0.1, 10.0, 5.0)

        if st.button("Run φ-Scaled Simulation"):
            from simulations.quantum_circuit import PhiScaledCircuit
            import numpy as np
            tlist = np.linspace(0, T, 200)

            # Example multi-qubit Hamiltonian
            if num_qubits==1:
                H0 = sigmaz()
            else:
                from qutip import sigmaz, sigmax, qeye, tensor
                H0 = tensor(sigmaz(), qeye(2)) + 0.2*tensor(qeye(2), sigmax())

            # Build circuit
            circuit = PhiScaledCircuit(
                base_hamiltonian=H0,
                alpha=alpha,
                beta=beta,
                noise_strength=noise_strength,
                noise_type=noise_type
            )
            result = circuit.evolve_state(psi_init, fractal_steps, tlist)
            st.session_state['simulation_results'] = result
            st.success("φ-Scaled simulation done. Check the Analysis tab.")

with tabs[1]:
    st.header("Analysis Tab")
    st.write("Here you can analyze results. This is a placeholder.")
    if 'simulation_results' not in st.session_state:
        st.warning("No simulation results found.")
    else:
        st.write("🌀 Display or process results from st.session_state['simulation_results']")
        # e.g. show <Z> vs time, etc.
        # If you want to do real analysis or plots:
        # result = st.session_state['simulation_results']
        # ...
        
display_logs()
