"""
Runtime-driven fragment-substitution with structure preservation
----------------------------------------------------------------

* No external data files
* No pickle
* Uses RDKit's built-in SMARTS catalogues to fabricate default tables
* Enforces structural similarity to original molecule
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Iterable, Optional, Tuple, Set, Any

import inspect
import random
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import Fragments, BRICS, AllChem, DataStructs, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers to pull SMARTS patterns from rdkit.Chem.Fragments
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  New similarity and scaffold-preserving functions
# ─────────────────────────────────────────────────────────────────────────────

def get_molecule_scaffold(mol: Chem.Mol) -> Chem.Mol:
    """Extract the Murcko scaffold from a molecule."""
    return MurckoScaffold.GetScaffoldForMol(mol)


def calculate_similarity(mol1: Chem.Mol, mol2: Chem.Mol, method: str = "morgan") -> float:
    """
    Calculate similarity between two molecules.

    Args:
        mol1: First molecule
        mol2: Second molecule
        method: Fingerprint method ('morgan', 'maccs', 'topological')

    Returns:
        Tanimoto similarity score (0-1)
    """
    if method == "morgan":
        # Use MorganGenerator for fingerprint generation
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp1 = morgan_gen.GetFingerprint(mol1)
        fp2 = morgan_gen.GetFingerprint(mol2)
    elif method == "maccs":
        # Use MACCS keys fingerprint
        from rdkit.Chem import MACCSkeys
        fp1 = MACCSkeys.GenMACCSKeys(mol1)
        fp2 = MACCSkeys.GenMACCSKeys(mol2)
    else:  # default to topological
        # Use RDKit's topological fingerprint
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def has_similar_size(mol1: Chem.Mol, mol2: Chem.Mol, max_diff_ratio: float = 0.3) -> bool:
    """Check if two molecules have similar sizes (within max_diff_ratio)."""
    num_atoms1 = mol1.GetNumAtoms()
    num_atoms2 = mol2.GetNumAtoms()
    
    if num_atoms1 == 0 or num_atoms2 == 0:
        return False
        
    size_ratio = abs(num_atoms1 - num_atoms2) / max(num_atoms1, num_atoms2)
    return size_ratio <= max_diff_ratio


def has_same_scaffold(mol1: Chem.Mol, mol2: Chem.Mol) -> bool:
    """Check if two molecules have the same Murcko scaffold."""
    try:
        scaffold1 = MurckoScaffold.MurckoScaffoldSmiles(Chem.MolToSmiles(mol1))
        scaffold2 = MurckoScaffold.MurckoScaffoldSmiles(Chem.MolToSmiles(mol2))
        return scaffold1 == scaffold2
    except:
        return False


def find_substructure_matches(mol: Chem.Mol, smarts_pattern: str) -> List[Tuple[int, ...]]:
    """Find all matches of a SMARTS pattern in a molecule."""
    pattern = Chem.MolFromSmarts(smarts_pattern)
    if pattern:
        return mol.GetSubstructMatches(pattern)
    return []


def modify_functional_group(
    mol: Chem.Mol, 
    smarts: str, 
    replacements: List[str],
    preserve_scaffold: bool = True
) -> List[Chem.Mol]:
    """
    Modify a functional group in a molecule while preserving its overall structure.
    
    Args:
        mol: Original molecule
        smarts: SMARTS pattern for the functional group
        replacements: List of SMARTS patterns to replace with
        preserve_scaffold: Whether to enforce scaffold preservation
        
    Returns:
        List of modified molecules
    """
    results = []
    pattern = Chem.MolFromSmarts(smarts)
    
    if not pattern:
        return results
        
    # Find all matches of the pattern
    matches = mol.GetSubstructMatches(pattern)
    
    if not matches:
        return results
        
    # Get the scaffold of the original molecule
    scaffold = get_molecule_scaffold(mol) if preserve_scaffold else None
    
    # Try each replacement
    for repl_smarts in replacements:
        repl = Chem.MolFromSmarts(repl_smarts)
        if not repl:
            continue
            
        try:
            # Replace the substructure
            new_mols = Chem.ReplaceSubstructs(mol, pattern, repl)
            
            for new_mol in new_mols:
                try:
                    # Convert to canonical form
                    new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
                    
                    if new_mol:
                        # Validate molecule
                        Chem.SanitizeMol(new_mol)
                        
                        # Check if scaffold is preserved (if required)
                        if preserve_scaffold and scaffold:
                            new_scaffold = get_molecule_scaffold(new_mol)
                            if not new_scaffold or not calculate_similarity(scaffold, new_scaffold, "morgan") > 0.7:
                                continue
                                
                        # Add to results if it's a valid molecule
                        results.append(new_mol)
                except Exception:
                    continue
        except Exception:
            continue
            
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  FragmentSubstitution class (enhanced with similarity constraints)
# ─────────────────────────────────────────────────────────────────────────────

_GENERIC_REPL = ["C", "N", "O", "S", "F", "Cl"]

class FragmentSubstitution:
    """
    Four-stage adversarial analogue generator with structural similarity constraints.
    
    Updated: 2025-05-26 by nglinhbao
    """

    def __init__(
        self,
        fragment_library: Optional[Dict[str, Sequence[str]]] = None,
        functional_group_rules: Optional[Dict[str, Sequence[str]]] = None,
        fragment_type_map: Optional[Dict[str, Sequence[str]]] = None,
        custom_rules: Optional[Dict[str, Dict[str, str]]] = None,
        max_brics_repl_per_type: int = 3,
        min_similarity: float = 0.4,
        preserve_scaffold: bool = True,
        max_size_diff_ratio: float = 0.3
    ):
        """
        Initialize the fragment substitution with similarity constraints.
        
        Args:
            fragment_library: Dictionary mapping fragment types to lists of SMILES
            functional_group_rules: Dictionary mapping patterns to replacements
            fragment_type_map: Dictionary mapping patterns to fragment types
            custom_rules: Custom substitution rules
            max_brics_repl_per_type: Maximum number of BRICS replacements per type
            min_similarity: Minimum similarity threshold (0-1)
            preserve_scaffold: Whether to preserve the molecular scaffold
            max_size_diff_ratio: Maximum allowed size difference ratio (0-1)
        """
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
        
        # Similarity constraints
        self.min_similarity = min_similarity
        self.preserve_scaffold = preserve_scaffold
        self.max_size_diff_ratio = max_size_diff_ratio
        
        print(f"[INFO] Initialized FragmentSubstitution with:")
        print(f"  - min_similarity: {min_similarity}")
        print(f"  - preserve_scaffold: {preserve_scaffold}")
        print(f"  - max_size_diff_ratio: {max_size_diff_ratio}")

    # ───────────────────────── public API ──────────────────────────
    def generate_substitutions(
        self,
        mol: Chem.Mol,
        important_fragments: Iterable[str],
        n_candidates: int = 10,
    ) -> List[Chem.Mol]:
        """
        Generate structurally similar adversarial molecules.
        
        Args:
            mol: Original molecule
            important_fragments: List of important fragments to target
            n_candidates: Maximum number of candidates to generate
            
        Returns:
            List of adversarial molecules that maintain structural similarity
        """
        important_fragments = list(important_fragments or [])
        mol_smiles = Chem.MolToSmiles(mol)
        print(f"[INFO] Original: {mol_smiles}")
        
        # Extract scaffold information
        try:
            scaffold = get_molecule_scaffold(mol) if self.preserve_scaffold else None
            scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else "N/A"
            print(f"[INFO] Scaffold: {scaffold_smiles}")
        except Exception as e:
            print(f"[WARN] Couldn't extract scaffold: {str(e)}")
            scaffold = None

        # Track all candidates and their similarity scores
        cands: List[Tuple[Chem.Mol, float]] = []  # (molecule, similarity)

        # ------------------ Method 1: targeted fragment swap ----------------
        print("[INFO] Method 1: Targeted fragment swap with similarity constraints")
        for frag in important_fragments:
            if not frag or frag not in mol_smiles:
                continue
                
            print(f"  - Processing fragment: {frag}")
            repls = next(
                (r for k, r in self.fragment_type_map.items() if k in frag),
                _GENERIC_REPL,
            )
            
            for r in repls:
                try:
                    smi = mol_smiles.replace(frag, r, 1)
                    m = Chem.MolFromSmiles(smi)
                    
                    if m and Chem.MolToSmiles(m) != mol_smiles:
                        try:
                            Chem.SanitizeMol(m)
                            
                            # Check size similarity
                            if not has_similar_size(mol, m, self.max_size_diff_ratio):
                                continue
                                
                            # Calculate similarity
                            sim = calculate_similarity(mol, m, "morgan")
                            
                            # Check scaffold preservation if required
                            if self.preserve_scaffold and scaffold:
                                if not has_same_scaffold(mol, m):
                                    continue
                            
                            # Add if passes similarity threshold
                            if sim >= self.min_similarity:
                                print(f"    Created valid molecule (sim={sim:.2f}): {frag} → {r}")
                                cands.append((m, sim))
                            
                        except Exception as e:
                            print(f"    Error: {str(e)}")
                except Exception:
                    pass

        # --------------- Method 2: functional-group replacement ------------
        print("[INFO] Method 2: Functional group replacement with structure preservation")
        for pat, repls in self.functional_group_rules.items():
            if pat not in mol_smiles:
                continue
                
            for r in repls:
                try:
                    # Try to replace with more controlled approach
                    pattern = Chem.MolFromSmarts(pat)
                    if pattern and mol.HasSubstructMatch(pattern):
                        modified_mols = modify_functional_group(
                            mol, pat, [r], 
                            preserve_scaffold=self.preserve_scaffold
                        )
                        
                        for m in modified_mols:
                            if m and Chem.MolToSmiles(m) != mol_smiles:
                                # Calculate similarity
                                sim = calculate_similarity(mol, m, "morgan")
                                
                                # Check size similarity
                                if not has_similar_size(mol, m, self.max_size_diff_ratio):
                                    continue
                                
                                # Add if passes similarity threshold
                                if sim >= self.min_similarity:
                                    print(f"    Created valid molecule (sim={sim:.2f}): {pat} → {r}")
                                    cands.append((m, sim))
                    else:
                        # Fallback to simple replacement
                        smi = mol_smiles.replace(pat, r, 1)
                        m = Chem.MolFromSmiles(smi)
                        
                        if m and Chem.MolToSmiles(m) != mol_smiles:
                            Chem.SanitizeMol(m)
                            
                            # Calculate similarity
                            sim = calculate_similarity(mol, m, "morgan")
                            
                            # Check size similarity
                            if not has_similar_size(mol, m, self.max_size_diff_ratio):
                                continue
                                
                            # Check scaffold preservation if required
                            if self.preserve_scaffold and scaffold:
                                if not has_same_scaffold(mol, m):
                                    continue
                            
                            # Add if passes similarity threshold
                            if sim >= self.min_similarity:
                                print(f"    Created valid molecule (sim={sim:.2f}): {pat} → {r}")
                                cands.append((m, sim))
                except Exception:
                    pass

        # ----------------- Method 3: BRICS recombination -------------------
        print("[INFO] Method 3: BRICS recombination with scaffold preservation")
        try:
            frags = list(BRICS.BRICSDecompose(mol, keepNonLeafNodes=True))
        except Exception:
            frags = []
            
        if len(frags) > 1:
            # Find BRICS fragments that contain important fragments
            prio = [
                i for i, f in enumerate(frags)
                if any(imp and imp in f for imp in important_fragments)
            ] or list(range(min(3, len(frags))))
            
            # Map to identify fragments that are part of the scaffold
            scaffold_frags = set()
            if scaffold:
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                for i, frag in enumerate(frags):
                    if frag in scaffold_smiles:
                        scaffold_frags.add(i)
            
            for idx in prio:
                # Skip modification of scaffold fragments if preserving scaffold
                if self.preserve_scaffold and idx in scaffold_frags:
                    continue
                    
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
                                
                                # Calculate similarity
                                sim = calculate_similarity(mol, m, "morgan")
                                
                                # Check size similarity
                                if not has_similar_size(mol, m, self.max_size_diff_ratio):
                                    continue
                                    
                                # Check scaffold preservation if required
                                if self.preserve_scaffold and scaffold:
                                    m_scaffold = get_molecule_scaffold(m)
                                    if not m_scaffold:
                                        continue
                                    scaffold_sim = calculate_similarity(scaffold, m_scaffold, "morgan")
                                    if scaffold_sim < 0.7:
                                        continue
                                
                                # Add if passes similarity threshold
                                if sim >= self.min_similarity:
                                    print(f"    Created valid molecule via BRICS (sim={sim:.2f})")
                                    cands.append((m, sim))
                        except Exception:
                            pass

        # ------------------- Method 4: atom-level mutation -----------------
        print("[INFO] Method 4: Atom-level mutation preserving structure")
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
        
        # Find atoms that should not be modified if preserving scaffold
        protected_atoms = set()
        if self.preserve_scaffold and scaffold:
            scaffold_matches = mol.GetSubstructMatches(scaffold)
            for match in scaffold_matches:
                for atom_idx in match:
                    # Allow terminal atoms of scaffold to be modified
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetDegree() > 1:  # Not a terminal atom
                        protected_atoms.add(atom_idx)
        
        priority_atoms = self._priority_atoms(mol, important_fragments)
        pt = Chem.GetPeriodicTable()

        for idx in priority_atoms:
            # Skip protected atoms when preserving scaffold
            if self.preserve_scaffold and idx in protected_atoms:
                continue
                
            orig = mol.GetAtomWithIdx(idx).GetSymbol()
            for new in atom_mutation_table.get(orig, []):
                rw = Chem.RWMol(mol)
                rw.GetAtomWithIdx(idx).SetAtomicNum(pt.GetAtomicNumber(new))
                try:
                    m = rw.GetMol()
                    Chem.SanitizeMol(m)
                    
                    if Chem.MolToSmiles(m) != mol_smiles:
                        # Calculate similarity
                        sim = calculate_similarity(mol, m, "morgan")
                        
                        # Check size similarity - always true for atom mutations
                        
                        # Add if passes similarity threshold - atom changes are always similar
                        # but we still check for consistency
                        if sim >= self.min_similarity:
                            print(f"    Created valid molecule by changing atom {idx} from {orig} to {new} (sim={sim:.2f})")
                            cands.append((m, sim))
                except Exception:
                    pass

        # --------------------- final dedup / sort by similarity ----------------------
        # Sort by uniqueness (to remove duplicates) and similarity (prioritize similar molecules)
        uniq_mols = self._dedup_valid([m for m, _ in cands], mol_smiles)
        
        # Re-calculate similarities for the unique molecules
        scored_mols = []
        for m in uniq_mols:
            sim = calculate_similarity(mol, m, "morgan")
            scored_mols.append((m, sim))
            
        # Sort by similarity (descending)
        scored_mols.sort(key=lambda x: x[1], reverse=True)
        
        # Get molecules only
        final_mols = [m for m, _ in scored_mols[:n_candidates]]
        
        print(f"[INFO] Generated {len(final_mols)} unique valid candidates with similarity ≥ {self.min_similarity}")
        
        if final_mols:
            print("[INFO] Top candidates:")
            for i, (m, sim) in enumerate(scored_mols[:min(5, len(scored_mols))]):
                print(f"  {i+1}. {Chem.MolToSmiles(m)} (similarity: {sim:.2f})")
                
        return final_mols

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