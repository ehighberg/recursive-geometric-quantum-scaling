"""Ownership and scientific-integrity contracts for the topological UI."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")
GENERIC_SOURCE = (ROOT / "app" / "analyze_results.py").read_text(encoding="utf-8")


def test_dedicated_tab_owns_topological_metrics():
    for label in (
        "Chern Number",
        "Winding Number",
        "Fractal-Chern Correlation",
        "Protection Dimension",
        "Total Computation Time",
        "Export Analysis Results",
    ):
        assert label in APP_SOURCE
        assert label not in GENERIC_SOURCE


def test_topological_ui_does_not_construct_phi_enhancement():
    forbidden = (
        "phi_resonant_index",
        "Apply phi-resonant enhancement",
        "demonstrating the special role",
        "energy_gaps[i] *=",
        "localization_measures[i] *=",
    )
    for phrase in forbidden:
        assert phrase not in APP_SOURCE
