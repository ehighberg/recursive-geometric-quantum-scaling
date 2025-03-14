# """
# Tests for animated visualization functionality.
# """

# import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend for testing
# from qutip import sigmaz
# from simulations.quantum_state import state_plus, state_zero
# from simulations.quantum_circuit import StandardCircuit
# from analyses.visualization.state_plots import (
#     animate_state_evolution,
#     animate_bloch_sphere
# )
# from analyses.visualization.metric_plots import (
#     animate_metric_evolution,
#     calculate_metrics
# )

# def test_state_evolution_animation():
#     """Test state evolution animation creation"""
#     # Create test states
#     psi = state_plus()
#     H = sigmaz()
#     circuit = StandardCircuit(H, total_time=5.0, n_steps=50)
#     result = circuit.evolve_closed(psi)
    
#     # Create animation
#     times = list(range(len(result.states)))
#     anim = animate_state_evolution(
#         result.states,
#         times,
#         title="Test State Evolution",
#         interval=50
#     )
    
#     # Verify animation properties
#     assert anim is not None
#     assert anim.event_source.interval == 50
#     # Save animation to verify frame count
#     anim.save('test_state_evolution.gif', writer='pillow')

# def test_bloch_sphere_animation():
#     """Test Bloch sphere animation creation"""
#     # Create test states
#     psi = state_plus()
#     H = sigmaz()
#     circuit = StandardCircuit(H, total_time=5.0, n_steps=50)
#     result = circuit.evolve_closed(psi)
    
#     # Create animation
#     anim = animate_bloch_sphere(
#         result.states,
#         title="Test Bloch Sphere",
#         interval=50
#     )
    
#     # Verify animation properties
#     assert anim is not None
#     assert anim.event_source.interval == 50
#     # Save animation to verify frame count
#     anim.save('test_bloch_sphere.gif', writer='pillow')

# def test_metric_evolution_animation():
#     """Test metric evolution animation creation"""
#     # Create test states
#     psi = state_plus()
#     H = sigmaz()
#     circuit = StandardCircuit(H, total_time=5.0, n_steps=50)
#     result = circuit.evolve_closed(psi)
    
#     # Calculate metrics
#     times = list(range(len(result.states)))
#     metrics = calculate_metrics(result.states)
    
#     # Create animation
#     anim = animate_metric_evolution(
#         metrics,
#         times,
#         title="Test Metric Evolution",
#         interval=50
#     )
    
#     # Verify animation properties
#     assert anim is not None
#     assert anim.event_source.interval == 50
#     # Save animation to verify frame count
#     anim.save('test_metric_evolution.gif', writer='pillow')
    
#     # Verify metrics are calculated correctly for single-qubit states
#     assert 'coherence' in metrics
#     assert 'entropy' in metrics
#     assert 'entanglement' not in metrics  # Not calculated for single-qubit states
#     assert len(metrics['coherence']) == len(times)

# def test_animation_with_noise():
#     """Test animations with noisy evolution"""
#     # Create test states with noise
#     psi = state_plus()
#     H = sigmaz()
#     # Add dephasing noise collapse operator
#     c_ops = [np.sqrt(0.1) * sigmaz()]  # Dephasing rate = 0.1
#     circuit = StandardCircuit(H, total_time=5.0, n_steps=50, c_ops=c_ops)
#     result = circuit.evolve_open(psi)
    
#     # Create animations
#     times = list(range(len(result.states)))
#     state_anim = animate_state_evolution(
#         result.states,
#         times,
#         title="Noisy State Evolution",
#         interval=50
#     )
    
#     bloch_anim = animate_bloch_sphere(
#         result.states,
#         title="Noisy Bloch Evolution",
#         interval=50
#     )
    
#     metrics = calculate_metrics(result.states)
#     metric_anim = animate_metric_evolution(
#         metrics,
#         times,
#         title="Noisy Metric Evolution",
#         interval=50
#     )
    
#     # Verify animations
#     assert state_anim is not None
#     assert bloch_anim is not None
#     assert metric_anim is not None
    
#     # Verify noise effects
#     coherence = metrics['coherence']
#     # Check if coherence is decaying or already at minimum
#     if coherence[0] > 0:
#         assert coherence[-1] <= coherence[0]  # Coherence should decay or stay at minimum
#     else:
#         # If initial coherence is already 0, just verify it's not increasing
#         assert coherence[-1] <= 0.01  # Allow small numerical errors

# def test_animation_smoothing():
#     """Test animation smoothing between states"""
#     # Create two distinct states
#     psi1 = state_zero()
#     psi2 = state_plus()
    
#     # Create animation with smoothing
#     times = [0, 1]
#     anim = animate_state_evolution(
#         [psi1, psi2],
#         times,
#         interval=50,
#         smoothing_steps=5
#     )
    
#     # Verify animation properties
#     assert anim is not None
#     assert anim.event_source.interval == 50
#     # Save animation to verify frame count
#     anim.save('test_smoothing.gif', writer='pillow')

# def test_metric_calculation():
#     """Test metric calculation for visualization"""
#     # Create test states
#     psi = state_plus()
#     H = sigmaz()
#     circuit = StandardCircuit(H, total_time=5.0, n_steps=50)
#     result = circuit.evolve_closed(psi)
    
#     # Calculate metrics
#     metrics = calculate_metrics(result.states)
    
#     # Verify metric properties
#     for metric_name, values in metrics.items():
#         assert len(values) == len(result.states)
#         assert all(0 <= v <= 1 for v in values)  # Metrics should be normalized
#         assert not any(np.isnan(v) for v in values)  # No NaN values
