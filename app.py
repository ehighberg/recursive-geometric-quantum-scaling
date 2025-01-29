"""The main entry point for the Quantum Simulation and Analysis Tool. It sets up the Streamlit app and handles the overall layout, styling, and user interface."""

import streamlit as st

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
This tool allows you to simulate quantum states and circuits under various conditions and analyze their results. You can input parameters, run simulations, and view detailed logs and analysis results.
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

st.sidebar.header("Configuration")

with st.sidebar.expander("🔧 Advanced Configuration", expanded=False):
    st.markdown("### Noise Parameters")
    noise_strength = st.slider("Noise Strength", 0.001, 0.1, 0.01, help="Strength of environmental noise")
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

    # State creation
    initial_states = {
        'standard': ['fock', 'coherent', 'squeezed', 'cat', 'bell', 'ghz', 'cluster'],
        'entanglement': ['bell', 'ghz', 'cluster']  # Multi-qubit states only
    }
