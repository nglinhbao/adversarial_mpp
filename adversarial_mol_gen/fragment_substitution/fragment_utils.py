"""
Helper functions for extracting and analyzing molecular fragments.
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Optional, Tuple, Set, Any

import inspect
import random

from rdkit import Chem
from rdkit.Chem import Fragments


def _smarts_from_fragments() -> Dict[str, str]:
    """
    Return {label: SMARTS} by scraping every `frag*` function's docstring in
    `rdkit.Chem.Fragments`.  Each docstring ends with a SMARTS pattern.
    """
    patterns: Dict[str, str] = {}
    for name, fn in inspect.getmembers(Fragments, inspect.isfunction):
        if name.startswith("frag"):
            doc = fn.__doc__ or ""
            if ":" in doc:
                smarts = doc.split(":")[-1].strip()
                patterns[name] = smarts
    return patterns


def _example_smiles_from_smarts(smarts: str, bases: Sequence[str] = ("C", "N")) -> List[str]:
    """
    Very lightweight method to turn a SMARTS skeleton into a handful of *valid*
    SMILES by replacing the query with simple groups.
    Not chemically exhaustive—just good enough to seed our tables.
    """
    q = Chem.MolFromSmarts(smarts)
    out: List[str] = []
    if q is None:
        return out
    for repl in bases:
        try:
            # Replace the pattern with a simple molecule
            m = Chem.ReplaceSubstructs(
                q, q, Chem.MolFromSmiles(repl), replaceAll=True
            )[0]
            smi = Chem.MolToSmiles(m, isomericSmiles=False)
            out.append(smi)
        except Exception:
            pass
    return list(dict.fromkeys(out))  # dedupe


def build_runtime_tables(
    n_examples: int = 4,
    seed: int = 42,
) -> tuple[
    Dict[str, List[str]],  # fragment_library
    Dict[str, List[str]],  # functional_group_rules
    Dict[str, List[str]],  # fragment_type_map
]:
    """
    Fabricate three lookup tables (fragment library, functional group rules,
    fragment-type map) directly from RDKit resources.
    """
    rng = random.Random(seed)

    # 1) Grab every SMARTS pattern from rdkit.Chem.Fragments
    smarts_map = _smarts_from_fragments()

    # 2) Convert each SMARTS into a few example SMILES strings
    smiles_sets: Dict[str, List[str]] = {
        label: _example_smiles_from_smarts(s, bases=("C", "N", "O", "Cl")) or ["C"]
        for label, s in smarts_map.items()
    }

    # 3) fragment_library — key=label, value=list[SMILES]
    fragment_library = smiles_sets

    # 4) functional_group_rules — choose first example as pattern, rest as replacements
    functional_group_rules = {}
    for examples in smiles_sets.values():
        if len(examples) > 1:
            pattern = examples[0]
            repls = rng.sample(examples[1:], k=min(len(examples) - 1, n_examples))
            functional_group_rules[pattern] = repls

    # 5) fragment_type_map — simple substring lookup: pattern → replacements
    fragment_type_map = {
        examples[0]: (examples[1:n_examples + 1] or ["C", "N", "O"])
        for examples in smiles_sets.values()
    }

    return fragment_library, functional_group_rules, fragment_type_map


def find_substructure_matches(mol: Chem.Mol, smarts_pattern: str) -> List[Tuple[int, ...]]:
    """Find all matches of a SMARTS pattern in a molecule."""
    pattern = Chem.MolFromSmarts(smarts_pattern)
    if pattern:
        return mol.GetSubstructMatches(pattern)
    return []