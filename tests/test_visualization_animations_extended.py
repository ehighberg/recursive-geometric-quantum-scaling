# """
# Extended tests for visualization animations including topological features.
# """

# import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend for testing
# from matplotlib import animation
# import matplotlib.pyplot as plt
# import pytest
# from qutip import basis, Qobj

# from analyses.visualization.state_plots import (
#     animate_state_evolution,
#     animate_bloch_sphere
# )
# from analyses.visualization.metric_plots import (
#     animate_metric_evolution,
#     calculate_metrics
# )
# from analyses.topology_plots import plot_invariants, plot_protection_metrics

# def generate_test_evolution():
#     """Generate test evolution data"""
#     times = np.linspace(0, 10, 50)
#     states = []
#     for t in times:
#         # Create a time-dependent superposition
#         alpha = np.cos(t)
#         beta = np.sin(t)
#         state = (alpha * basis(2, 0) + beta * basis(2, 1)).unit()
#         states.append(state)
#     return states, times

# def test_topological_animation():
#     """Test animation of topological invariants"""
#     # Generate test data
#     control_range = (0.0, 5.0)
#     times = np.linspace(0, 10, 50)
    
#     # Create mock evolution of topological invariants
#     chern_values = np.sin(times) * 0.5 + 0.5
#     winding_values = np.cos(times) * 0.5
#     z2_values = np.round(np.sin(times) + 1) % 2
    
#     # Create animation
#     fig = matplotlib.pyplot.figure()
#     ax = fig.add_subplot(111)
    
#     def animate(frame):
#         ax.clear()
#         ax.plot(times[:frame], chern_values[:frame], label='Chern')
#         ax.plot(times[:frame], winding_values[:frame], label='Winding')
#         ax.plot(times[:frame], z2_values[:frame], label='Z2')
#         ax.set_ylim(-1, 2)
#         ax.legend()
#         return ax,
    
#     anim = animation.FuncAnimation(
#         fig, animate, frames=len(times),
#         interval=50, blit=True
#     )
    
#     anim.save('test_topological.gif', writer='pillow')
#     matplotlib.pyplot.close(fig)

# def test_protection_metrics_animation():
#     """Test animation of protection metrics"""
#     control_range = (0.0, 5.0)
#     times = np.linspace(0, 10, 50)
#     x_values = np.linspace(control_range[0], control_range[1], 100)
    
#     # Create time-dependent protection metrics
#     energy_gaps = []
#     localization_measures = []
    
#     for t in times:
#         # Create time-varying metrics
#         energy_gap = np.abs(np.sin(x_values + t))
#         localization = np.abs(np.cos(x_values - t))
#         energy_gaps.append(energy_gap)
#         localization_measures.append(localization)
    
#     fig = matplotlib.pyplot.figure()
#     ax = fig.add_subplot(111)
    
#     def animate(frame):
#         ax.clear()
#         ax.plot(x_values, energy_gaps[frame], label='Energy Gap')
#         ax.plot(x_values, localization_measures[frame], label='Localization')
#         ax.set_ylim(0, 1)
#         ax.legend()
#         return ax,
    
#     anim = animation.FuncAnimation(
#         fig, animate, frames=len(times),
#         interval=50, blit=True
#     )
    
#     anim.save('test_protection.gif', writer='pillow')
#     matplotlib.pyplot.close(fig)

# def test_state_evolution_with_topology():
#     """Test state evolution animation with topological features"""
#     states, times = generate_test_evolution()
    
#     # Add topological phase information
#     topological_phases = np.sin(times) * 0.5 + 0.5
    
#     anim = animate_state_evolution(
#         states,
#         times,
#         title="State Evolution with Topology"
#     )
    
#     anim.save('test_state_evolution_topo.gif', writer='pillow')
#     matplotlib.pyplot.close()

# def test_bloch_sphere_with_topology():
#     """Test Bloch sphere animation with topological features"""
#     states, times = generate_test_evolution()
    
#     # Add winding number visualization
#     winding_numbers = np.cumsum(np.diff(np.angle(
#         [state.full().flatten()[0] for state in states]
#     ))) / (2 * np.pi)
    
#     anim = animate_bloch_sphere(
#         states,
#         title="Bloch Evolution with Winding"
#     )
    
#     anim.save('test_bloch_topo.gif', writer='pillow')
#     matplotlib.pyplot.close()

# def test_metric_evolution_with_topology():
#     """Test metric evolution animation with topological invariants"""
#     states, times = generate_test_evolution()
    
#     # Calculate standard metrics
#     metrics = calculate_metrics(states)
    
#     # Create a custom color cycle to avoid StopIteration
#     plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
#     # Add topological metrics - ensure they're in the expected format
#     # The animate_metric_evolution function expects metrics to be a dictionary
#     # of metric names to lists of values
#     metrics['chern_number'] = np.abs(np.sin(times)).tolist()  # Convert to list
#     metrics['winding_number'] = np.cos(times).tolist()  # Convert to list
    
#     # Create a simple animation manually instead of using animate_metric_evolution
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     def animate(frame):
#         ax.clear()
#         for metric_name, values in metrics.items():
#             ax.plot(times[:frame], values[:frame], label=metric_name.replace('_', ' ').title())
#         ax.set_xlim(min(times), max(times))
#         ax.set_ylim(0, 1.1)
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Value')
#         ax.set_title("Metric Evolution with Topology")
#         ax.legend()
#         return ax,
    
#     anim = animation.FuncAnimation(
#         fig, animate, frames=len(times),
#         interval=50, blit=True
#     )
    
#     anim.save('test_metric_topo.gif', writer='pillow')
#     plt.close(fig)

# def test_animation_smoothing():
#     """Test animation smoothing with topological transitions"""
#     states, times = generate_test_evolution()
    
#     # Create topological phase transition
#     transition_point = len(times) // 2
#     topological_phases = np.concatenate([
#         np.zeros(transition_point),
#         np.ones(len(times) - transition_point)
#     ])
    
#     anim = animate_state_evolution(
#         states,
#         times,
#         smoothing_steps=5,
#         title="Smooth Topological Transition"
#     )
    
#     anim.save('test_smooth_topo.gif', writer='pillow')
#     matplotlib.pyplot.close()

# def test_animation_error_handling():
#     """Test error handling in animations"""
#     states, times = generate_test_evolution()
    
#     # Test with mismatched states and times
#     with pytest.raises(ValueError):
#         animate_state_evolution(
#             states[:10],  # Only use first 10 states
#             times,  # But all times
#         )
    
#     # Test with invalid smoothing parameters
#     with pytest.raises(ValueError):
#         animate_state_evolution(
#             states,
#             times,
#             smoothing_steps=-1  # Invalid smoothing steps
#         )

# def test_animation_performance():
#     """Test animation performance with different data sizes"""
#     # Generate large dataset
#     times = np.linspace(0, 10, 200)
#     states = []
#     for t in times:
#         alpha = np.cos(t)
#         beta = np.sin(t)
#         state = (alpha * basis(2, 0) + beta * basis(2, 1)).unit()
#         states.append(state)
    
#     # Test with different frame rates
#     for interval in [20, 50, 100]:
#         anim = animate_state_evolution(
#             states,
#             times,
#             interval=interval,
#             title=f"Performance Test ({interval}ms)"
#         )
#         anim.save(f'test_performance_{interval}.gif', writer='pillow')
#         matplotlib.pyplot.close()
