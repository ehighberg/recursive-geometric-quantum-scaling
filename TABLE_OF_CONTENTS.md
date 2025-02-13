# Table of Contents

* [README.md](README.md)
* [requirements.txt](requirements.txt)
* [app.py](app.py)
* [constants.py](constants.py)
* [audit_plan.md](audit_plan.md)

## analyses/
* [analyses/__init__.py](analyses/__init__.py)
* [analyses/coherence.py](analyses/coherence.py)
* [analyses/entanglement.py](analyses/entanglement.py)
* [analyses/entropy.py](analyses/entropy.py)

### analyses/visualization/
* [analyses/visualization/__init__.py](analyses/visualization/__init__.py)
* [analyses/visualization/circuit_diagrams.py](analyses/visualization/circuit_diagrams.py)
* [analyses/visualization/metric_plots.py](analyses/visualization/metric_plots.py)
* [analyses/visualization/state_plots.py](analyses/visualization/state_plots.py)
* [analyses/visualization/style_config.py](analyses/visualization/style_config.py)

## app/
* [app/__init__.py](app/__init__.py)
* [app/analyze_results.py](app/analyze_results.py)

## config/
* [config/evolution_config.yaml](config/evolution_config.yaml)

## docs/
### docs/references/
* [docs/references/academic.txt](docs/references/academic.txt)
* [docs/references/libraries.txt](docs/references/libraries.txt)

## simulations/
* [simulations/__init__.py](simulations/__init__.py)
* [simulations/config.py](simulations/config.py)
* [simulations/quantum_circuit.py](simulations/quantum_circuit.py)
* [simulations/quantum_state.py](simulations/quantum_state.py)

### simulations/scripts/
* [simulations/scripts/__init__.py](simulations/scripts/__init__.py)
* [simulations/scripts/evolve_circuit.py](simulations/scripts/evolve_circuit.py)
* [simulations/scripts/evolve_state.py](simulations/scripts/evolve_state.py)
* [simulations/scripts/fibonacci_anyon_braiding.py](simulations/scripts/fibonacci_anyon_braiding.py)
* [simulations/scripts/topological_placeholders.py](simulations/scripts/topological_placeholders.py)

#### simulations/scripts/braid_generators/
* [simulations/scripts/braid_generators/abstract_braid_generator.py](simulations/scripts/braid_generators/abstract_braid_generator.py)
* [simulations/scripts/braid_generators/braid_generator_2d.py](simulations/scripts/braid_generators/braid_generator_2d.py)
* [simulations/scripts/braid_generators/braid_generator_factory.py](simulations/scripts/braid_generators/braid_generator_factory.py)

## tests/
* [tests/__init__.py](tests/__init__.py)
* [tests/test_coherence.py](tests/test_coherence.py)
* [tests/test_entanglement.py](tests/test_entanglement.py)
* [tests/test_entropy.py](tests/test_entropy.py)
* [tests/test_evolve_circuit.py](tests/test_evolve_circuit.py)
* [tests/test_evolve_state.py](tests/test_evolve_state.py)
* [tests/test_fibonacci.py](tests/test_fibonacci.py)
* [tests/test_performance.py](tests/test_performance.py)
* [tests/test_quantum_circuit.py](tests/test_quantum_circuit.py)
* [tests/test_quantum_state.py](tests/test_quantum_state.py)
* [tests/test_visualization.py](tests/test_visualization.py)