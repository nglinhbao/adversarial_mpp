"""
Adversarial Molecule Generator Package

A toolkit for generating adversarial molecules that maintain 
structural similarity and chemical properties.
"""

from .fragment_substitution import FragmentSubstitution, build_runtime_tables
from .fragment_substitution import calculate_similarity, calculate_properties

__version__ = "1.0.0"
__author__ = "nglinhbao"
__updated__ = "2025-05-27"

__all__ = [
    'FragmentSubstitution',
    'build_runtime_tables',
    'calculate_similarity',
    'calculate_properties'
]