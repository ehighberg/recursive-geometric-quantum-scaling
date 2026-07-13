from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_source_snapshot_has_no_package_claim():
    for name in ("setup.py", "pyproject.toml", "setup.cfg", "MANIFEST.in"):
        assert not (ROOT / name).exists(), f"source snapshot gained package metadata: {name}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "source snapshot" in readme
    assert "not a python package" in readme


def test_default_scale_is_neutral():
    from qutip import sigmaz
    from simulations.quantum_circuit import ScaledCircuit

    assert ScaledCircuit(sigmaz()).scale_factor == 1.0


def test_open_recurrence_does_not_inject_phi_bonus():
    from qutip import basis, sigmaz
    from simulations.quantum_circuit import ScaledCircuit
    from constants import PHI

    standard = ScaledCircuit(sigmaz(), scaling_factor=PHI)
    recursive = ScaledCircuit(sigmaz(), scaling_factor=PHI)
    recursive.use_phi_recursive = True

    initial_state = (basis(2, 0) + basis(2, 1)).unit()
    standard_result = standard.evolve_open(initial_state, n_steps=3)
    recursive_result = recursive.evolve_open(initial_state, n_steps=3)

    for standard_state, recursive_state in zip(
        standard_result.states, recursive_result.states
    ):
        assert (standard_state - recursive_state).norm() < 1e-12


@pytest.mark.parametrize("scaling_factor", [0, -1, np.nan, np.inf, -np.inf])
def test_phi_scaled_wrapper_rejects_invalid_scale(scaling_factor):
    from qutip import sigmaz
    from simulations.quantum_circuit import ScaledCircuit
    from simulations.scripts.evolve_circuit import run_phi_scaled_twoqubit_circuit

    with pytest.raises(ValueError, match="finite and greater than zero"):
        ScaledCircuit(sigmaz(), scaling_factor=scaling_factor)
    with pytest.raises(ValueError, match="finite and greater than zero"):
        run_phi_scaled_twoqubit_circuit(scaling_factor=scaling_factor)
