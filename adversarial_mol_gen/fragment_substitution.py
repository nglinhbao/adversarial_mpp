"""
Runtime-driven fragment-substitution demo
----------------------------------------

* No external data files.
* No pickle.
* Uses RDKit’s built-in SMARTS catalogues to fabricate default tables.
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Iterable, Optional

import inspect
import random

from rdkit import Chem
from rdkit.Chem import Fragments, BRICS


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers to pull SMARTS patterns from rdkit.Chem.Fragments
# ─────────────────────────────────────────────────────────────────────────────

def _smarts_from_fragments() -> Dict[str, str]:
    """
    Return {label: SMARTS} by scraping every `frag*` function’s docstring in
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


# ─────────────────────────────────────────────────────────────────────────────
#  FragmentSubstitution class (workflow unchanged)
# ─────────────────────────────────────────────────────────────────────────────

_GENERIC_REPL = ["C", "N", "O", "S", "F", "Cl"]

class FragmentSubstitution:
    """
    Four-stage adversarial analogue generator.

    Methods are exactly the same as in your original script, but every chemistry
    table is injected at construction time (defaults come from build_runtime_tables()).
    """

    def __init__(
        self,
        fragment_library: Optional[Dict[str, Sequence[str]]] = None,
        functional_group_rules: Optional[Dict[str, Sequence[str]]] = None,
        fragment_type_map: Optional[Dict[str, Sequence[str]]] = None,
        custom_rules: Optional[Dict[str, Dict[str, str]]] = None,
        max_brics_repl_per_type: int = 3,
    ):
        if fragment_library is None or functional_group_rules is None or fragment_type_map is None:
            lib, fg_rules, ftype_map = build_runtime_tables()
            fragment_library = fragment_library or lib
            functional_group_rules = functional_group_rules or fg_rules
            fragment_type_map = fragment_type_map or ftype_map

        self.fragment_library = fragment_library
        self.functional_group_rules = functional_group_rules
        self.fragment_type_map = fragment_type_map
        self.custom_rules = custom_rules or {}
        self.max_brics_repl_per_type = max_brics_repl_per_type

    # ───────────────────────── public API ──────────────────────────
    def generate_substitutions(
        self,
        mol: Chem.Mol,
        important_fragments: Iterable[str],
        n_candidates: int = 10,
    ) -> List[Chem.Mol]:

        important_fragments = list(important_fragments or [])
        mol_smiles = Chem.MolToSmiles(mol)
        print(f"[INFO] Original: {mol_smiles}")

        cands: List[Chem.Mol] = []

        # ------------------ Method 1: targeted fragment swap ----------------
        for frag in important_fragments:
            if not frag or frag not in mol_smiles:
                continue
            repls = next(
                (r for k, r in self.fragment_type_map.items() if k in frag),
                _GENERIC_REPL,
            )
            for r in repls:
                smi = mol_smiles.replace(frag, r, 1)
                m = Chem.MolFromSmiles(smi)
                if m and Chem.MolToSmiles(m) != mol_smiles:
                    try:
                        Chem.SanitizeMol(m)
                        cands.append(m)
                    except Exception:
                        pass

        # --------------- Method 2: functional-group replacement ------------
        for pat, repls in self.functional_group_rules.items():
            if pat not in mol_smiles:
                continue
            for r in repls:
                smi = mol_smiles.replace(pat, r, 1)
                m = Chem.MolFromSmiles(smi)
                if m and Chem.MolToSmiles(m) != mol_smiles:
                    try:
                        Chem.SanitizeMol(m)
                        cands.append(m)
                    except Exception:
                        pass

        # ----------------- Method 3: BRICS recombination -------------------
        try:
            frags = list(BRICS.BRICSDecompose(mol, keepNonLeafNodes=True))
        except Exception:
            frags = []
        if len(frags) > 1:
            prio = [
                i for i, f in enumerate(frags)
                if any(imp and imp in f for imp in important_fragments)
            ] or list(range(min(3, len(frags))))
            for idx in prio:
                for lib in self.fragment_library.values():
                    for repl in lib[: self.max_brics_repl_per_type]:
                        new_frags = frags.copy()
                        new_frags[idx] = repl
                        parts = [Chem.MolFromSmiles(x) for x in new_frags if x]
                        if len(parts) < 2:
                            continue
                        try:
                            m = BRICS.BRICSBuild(parts)
                            if m and Chem.MolToSmiles(m) != mol_smiles:
                                Chem.SanitizeMol(m)
                                cands.append(m)
                        except Exception:
                            pass

        # ------------------- Method 4: atom-level mutation -----------------
        atom_mutation_table = {
            "O": ["N", "S", "F"],
            "N": ["O", "S", "P"],
            "C": ["N", "O", "S"],
            "S": ["O", "N", "P"],
            "F": ["Cl", "Br"],
            "Cl": ["F", "Br"],
            "Br": ["Cl", "I"],
            "I": ["Br", "Cl"],
        }
        priority_atoms = self._priority_atoms(mol, important_fragments)
        pt = Chem.GetPeriodicTable()

        for idx in priority_atoms[:5]:
            orig = mol.GetAtomWithIdx(idx).GetSymbol()
            for new in atom_mutation_table.get(orig, []):
                rw = Chem.RWMol(mol)
                rw.GetAtomWithIdx(idx).SetAtomicNum(pt.GetAtomicNumber(new))
                try:
                    m = rw.GetMol()
                    Chem.SanitizeMol(m)
                    if Chem.MolToSmiles(m) != mol_smiles:
                        cands.append(m)
                except Exception:
                    pass

        # --------------------- final dedup / validity ----------------------
        uniq = self._dedup_valid(cands, mol_smiles)[: n_candidates]
        print(f"[INFO] Generated {len(uniq)} unique valid candidates")
        return uniq

    # ────────────────────── small utilities ──────────────────────
    @staticmethod
    def _priority_atoms(mol: Chem.Mol, imp_frags: List[str]) -> List[int]:
        prio = set()
        for s in imp_frags:
            q = Chem.MolFromSmiles(s) if s else None
            if q:
                for match in mol.GetSubstructMatches(q):
                    prio.update(match)
        atoms = [(i, 1 if i in prio else 0) for i in range(mol.GetNumAtoms())]
        atoms.sort(key=lambda t: t[1], reverse=True)
        return [i for i, _ in atoms]

    @staticmethod
    def _dedup_valid(mols: Iterable[Chem.Mol], orig: str) -> List[Chem.Mol]:
        seen, out = set(), []
        for m in mols:
            if not m:
                continue
            try:
                Chem.SanitizeMol(m)
                smi = Chem.MolToSmiles(m)
                if smi != orig and smi not in seen:
                    out.append(m)
                    seen.add(smi)
            except Exception:
                pass
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Minimal demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    important = ["C(=O)O", "c1ccccc1"]

    fs = FragmentSubstitution()          # uses runtime-generated tables
    new_mols = fs.generate_substitutions(aspirin, important, n_candidates=12)

    print("\nSMILES of generated candidates:")
    for m in new_mols:
        print("  ", Chem.MolToSmiles(m))
