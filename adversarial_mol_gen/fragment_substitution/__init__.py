"""
Runtime-driven fragment-substitution package with structure and property preservation
------------------------------------------------------------------------------------

* No external data files
* No pickle
* Uses RDKit's built-in SMARTS catalogues to fabricate default tables
* Enforces structural similarity to original molecule
* Enforces property similarity to original molecule
* Supports bond substitution, atom addition, and bond addition
"""

from .fragment_utils import build_runtime_tables
from .similarity import calculate_similarity, has_same_scaffold
from .property_utils import calculate_properties, property_similarity
from .core import FragmentSubstitution

__all__ = [
    'FragmentSubstitution',
    'build_runtime_tables',
    'calculate_similarity',
    'calculate_properties',
    'property_similarity',
    'has_same_scaffold'
]