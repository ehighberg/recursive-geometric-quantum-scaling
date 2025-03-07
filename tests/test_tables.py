"""
Tests for the tables module.
"""

import unittest
import pandas as pd

from analyses.tables.parameter_tables import (
    generate_parameter_overview_table,
    generate_simulation_parameters_table,
    export_table_to_latex
)
from analyses.tables.phase_tables import (
    generate_phase_diagram_table,
    classify_phase
)
from analyses.tables.performance_tables import (
    generate_performance_table,
    generate_convergence_table
)

class TestParameterTables(unittest.TestCase):
    """Test parameter table generation functions."""
    
    def test_generate_parameter_overview_table(self):
        """Test generating parameter overview table."""
        table = generate_parameter_overview_table()
        self.assertIsInstance(table, pd.DataFrame)
        self.assertGreater(len(table), 0)
        self.assertIn("Symbol", table.columns)
        self.assertIn("Physical Meaning", table.columns)
        self.assertIn("Typical Range/Values", table.columns)
        self.assertIn("Units/Dimensions", table.columns)
    
    def test_generate_simulation_parameters_table(self):
        """Test generating simulation parameters table."""
        # Create a mock result object
        class MockResult:
            def __init__(self):
                self.num_qubits = 2
                self.scaling_factor = 1.618
                self.n_steps = 50
        
        result = MockResult()
        table = generate_simulation_parameters_table(result)
        self.assertIsInstance(table, pd.DataFrame)
        self.assertGreater(len(table), 0)
        self.assertIn("Parameter", table.columns)
        self.assertIn("Value", table.columns)
    
    def test_export_table_to_latex(self):
        """Test exporting table to LaTeX."""
        df = pd.DataFrame({
            "Column1": [1, 2, 3],
            "Column2": ["a", "b", "c"]
        })
        latex = export_table_to_latex(df, "Test Caption", "tab:test")
        self.assertIsInstance(latex, str)
        self.assertIn("\\begin{table}", latex)
        self.assertIn("\\caption{Test Caption}", latex)
        self.assertIn("\\label{tab:test}", latex)

class TestPhaseTables(unittest.TestCase):
    """Test phase table generation functions."""
    
    def test_classify_phase(self):
        """Test phase classification."""
        # Topological phase
        phase = classify_phase(1.0, 0.1, 1, 1.5)
        self.assertEqual(phase, "Topological")
        
        # Critical phase
        phase = classify_phase(1.0, 0.005, 0, 1.5)
        self.assertEqual(phase, "Critical")
        
        # Trivial phase
        phase = classify_phase(1.0, 0.1, 0, 1.5)
        self.assertEqual(phase, "Trivial")
    
    def test_generate_phase_diagram_table(self):
        """Test generating phase diagram table."""
        # Test with default parameters
        table = generate_phase_diagram_table()
        self.assertIsInstance(table, pd.DataFrame)
        self.assertGreater(len(table), 0)
        self.assertIn("f_s Range", table.columns)
        self.assertIn("Phase Type", table.columns)
        
        # Test with custom ranges
        fs_ranges = [(0.1, 0.5), (0.5, 1.0)]
        table = generate_phase_diagram_table(fs_ranges=fs_ranges)
        self.assertEqual(len(table), len(fs_ranges))

class TestPerformanceTables(unittest.TestCase):
    """Test performance table generation functions."""
    
    def test_generate_performance_table(self):
        """Test generating performance table."""
        # Test with default parameters
        table = generate_performance_table()
        self.assertIsInstance(table, pd.DataFrame)
        self.assertGreater(len(table), 0)
        
        # Test with custom parameters
        system_sizes = [1, 2]
        methods = ["Method1", "Method2"]
        table = generate_performance_table(system_sizes=system_sizes, methods=methods)
        self.assertEqual(len(table), len(system_sizes) * len(methods))
    
    def test_generate_convergence_table(self):
        """Test generating convergence table."""
        # Create a mock result object
        class MockResult:
            def __init__(self):
                self.convergence_metrics = {
                    "fidelity": 0.999,
                    "trace_distance": 0.001
                }
        
        result = MockResult()
        table = generate_convergence_table(result)
        self.assertIsInstance(table, pd.DataFrame)
        self.assertGreater(len(table), 0)
        self.assertIn("Metric", table.columns)
        self.assertIn("Value", table.columns)

if __name__ == "__main__":
    unittest.main()
