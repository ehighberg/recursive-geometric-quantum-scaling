#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test if the topological invariant functions exist
"""

from analyses.topological_invariants import compute_standard_winding, compute_standard_z2_index, compute_berry_phase_standard

print('Functions exist:')
print(f'compute_standard_winding: {compute_standard_winding.__name__}')
print(f'compute_standard_z2_index: {compute_standard_z2_index.__name__}')
print(f'compute_berry_phase_standard: {compute_berry_phase_standard.__name__}')
