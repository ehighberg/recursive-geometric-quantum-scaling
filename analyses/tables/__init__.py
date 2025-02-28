"""
Tables package for quantum simulation analysis.
"""

# pylint: disable=import-error
from analyses.tables.parameter_tables import (
    generate_parameter_overview_table,
    generate_simulation_parameters_table
)

from analyses.tables.phase_tables import (
    generate_phase_diagram_table,
    classify_phase
)

from analyses.tables.performance_tables import (
    generate_performance_table,
    measure_performance
)

__all__ = [
    # Parameter tables
    'generate_parameter_overview_table',
    'generate_simulation_parameters_table',
    # Phase tables
    'generate_phase_diagram_table',
    'classify_phase',
    # Performance tables
    'generate_performance_table',
    'measure_performance'
]
