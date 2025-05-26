from __future__ import annotations
from typing import Dict, List, Sequence, Iterable, Tuple, Optional

import inspect
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import (AllChem, Draw, rdFMCS, Fragments,
                        BRICS, rdmolops)


# ─────────────────────────────────────────────────────────────────────────────
#   Helpers to auto-harvest SMARTS patterns from rdkit.Chem.Fragments
# ─────────────────────────────────────────────────────────────────────────────
def _smarts_from_fragments() -> Dict[str, str]:
    """
    Build {name: SMARTS} by scraping each `frag*` function’s docstring.
    RDKit embeds the SMARTS pattern at the end of every docstring.
    """
    patt = {}
    for name, fn in inspect.getmembers(Fragments, inspect.isfunction):
        if name.startswith("frag"):
            doc = fn.__doc__ or ""
            if ":" in doc:
                patt[name] = doc.split(":")[-1].strip()
    return patt


def _default_functional_groups() -> Dict[str, Chem.Mol]:
    """
    Use rdkit.Chem.Fragments SMARTS as the functional-group dictionary.
    Returns {label: MolFromSmarts}.
    """
    return {lbl: Chem.MolFromSmarts(sma)
            for lbl, sma in _smarts_from_fragments().items()
            if sma and Chem.MolFromSmarts(sma) is not None}


# ─────────────────────────────────────────────────────────────────────────────
#   Main class – workflow unchanged, logic data-driven
# ─────────────────────────────────────────────────────────────────────────────

class FragmentAnalysis:
    """
    Identify “important” fragments wrt. a black-box model.
    All hard-coded lists replaced by *parameters* with RDKit-based defaults.
    """

    def __init__(
        self,
        black_box_model,
        functional_groups: Optional[Dict[str, Chem.Mol]] = None,
        morgan_top_k: int = 3,
        perturb_delta: float = 0.01,
    ):
        """
        Parameters
        ----------
        black_box_model
            Callable with .predict(mol) → float.
        functional_groups
            {name: MolFromSmarts}.  If None, auto-generated from
            `rdkit.Chem.Fragments`.
        morgan_top_k
            How many top-contributing atoms to extract per fingerprint run.
        perturb_delta
            Absolute delta threshold for the perturbation test.
        """
        self.black_box_model = black_box_model
        self.functional_groups = functional_groups or _default_functional_groups()
        self.morgan_top_k = morgan_top_k
        self.perturb_delta = perturb_delta

        # meta
        self.version = "1.2.0-runtime"
        self.last_updated = "2025-05-26"

    # ────────────────────────── public interface ───────────────────────────
    def identify_important_fragments(
        self,
        mol_orig: Chem.Mol,
        mol_adv: Optional[Chem.Mol] = None,
    ) -> List[str]:
        """
        Return a list of *SMILES* fragments deemed important.
        """
        print("[INFO] Analyzing:", Chem.MolToSmiles(mol_orig))

        if mol_adv:
            return self._compare_molecules(mol_orig, mol_adv)

        frags = (
            self._perturbation_analysis(mol_orig) +
            self._functional_group_analysis(mol_orig) +
            self._morgan_fingerprint_analysis(mol_orig) +
            self._direct_fragment_extraction(mol_orig)
        )

        # dedupe / validate
        out = []
        seen = set()
        for f in frags:
            if not f or len(f) < 2:
                continue
            try:
                m = Chem.MolFromSmiles(f)
                if m and f not in seen:
                    seen.add(f)
                    out.append(f)
            except Exception:
                pass

        if not out:
            print("[INFO] No valid fragments, using fallback.")
            out = self._fallback_fragments(mol_orig)

        print("[INFO] Important fragments:", out)
        return out

    # ───────────────────────── internal helpers ────────────────────────────
    def _compare_molecules(self, m1: Chem.Mol, m2: Chem.Mol) -> List[str]:
        mcs = rdFMCS.FindMCS([m1, m2], completeRingsOnly=True)
        core = Chem.MolFromSmarts(mcs.smartsString)
        diff_smiles = []
        for mol in (m1, m2):
            diff_atoms = set(range(mol.GetNumAtoms())) - set(mol.GetSubstructMatch(core))
            for idx in diff_atoms:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, 2, idx)
                submol = Chem.PathToSubmol(mol, env) if env else None
                if submol and submol.GetNumAtoms() > 0:
                    diff_smiles.append(Chem.MolToSmiles(submol))
        return diff_smiles

    # ---------------- perturbation ----------------------------------------
    def _perturbation_analysis(self, mol_orig: Chem.Mol) -> List[str]:
        orig_pred = self.black_box_model.predict(mol_orig)
        brics_frags = list(BRICS.BRICSDecompose(mol_orig))

        imp = []
        for frag in brics_frags:
            frag_mol = Chem.MolFromSmiles(frag)
            if frag_mol and mol_orig.HasSubstructMatch(frag_mol):
                mod = Chem.MolFromSmiles(
                    Chem.MolToSmiles(mol_orig).replace(frag, "C", 1)
                )
                if not mod:
                    continue
                new_pred = self.black_box_model.predict(mod)
                if abs(new_pred - orig_pred) > self.perturb_delta:
                    imp.append(frag)
        return imp

    # ---------------- functional groups -----------------------------------
    def _functional_group_analysis(self, mol: Chem.Mol) -> List[str]:
        imp = []
        for name, smarts_mol in self.functional_groups.items():
            if mol.HasSubstructMatch(smarts_mol):
                for match in mol.GetSubstructMatches(smarts_mol):
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, 2, match[0])
                    sub = Chem.PathToSubmol(mol, env) if env else None
                    if sub and sub.GetNumAtoms() > 0:
                        imp.append(Chem.MolToSmiles(sub))
        return imp

    # ---------------- morgan fingerprint ----------------------------------
    def _morgan_fingerprint_analysis(self, mol: Chem.Mol) -> List[str]:
        info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, bitInfo=info)
        contrib = defaultdict(int)
        for _, atom_lists in info.items():
            for aid, _ in atom_lists:
                contrib[aid] += 1
        top_atoms = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[: self.morgan_top_k]
        imp = []
        for aid, _ in top_atoms:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, 2, aid)
            sub = Chem.PathToSubmol(mol, env) if env else None
            if sub and sub.GetNumAtoms() > 0:
                imp.append(Chem.MolToSmiles(sub))
        return imp

    # ---------------- direct extraction ------------------------------------
    def _direct_fragment_extraction(self, mol: Chem.Mol) -> List[str]:
        out = []
        # generic checks – use rdkit.Chem.Fragments counters
        if Fragments.fr_COO(mol):          out.append("C(=O)O")
        if Fragments.fr_ester(mol):        out.append("C(=O)OC")
        if Fragments.fr_ketone(mol):       out.append("C(=O)C")
        if Fragments.fr_benzene(mol):      out.append("c1ccccc1")
        # plus simple substring heuristics
        smi = Chem.MolToSmiles(mol)
        if "CC(=O)O" in smi:               out.append("CC(=O)O")
        if "c1ccccc1C(=O)O" in smi:        out.append("c1ccccc1C(=O)O")
        return out

    # ---------------- fallback ---------------------------------------------
    def _fallback_fragments(self, mol: Chem.Mol) -> List[str]:
        out = []
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings():
            atoms = set(ring_info.AtomRings()[0])
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, 1, next(iter(atoms)))
            sub = Chem.PathToSubmol(mol, env) if env else None
            if sub: out.append(Chem.MolToSmiles(sub))
        smi = Chem.MolToSmiles(mol)
        if "C(=O)" in smi:   out.append("C(=O)")
        if "c1" in smi:      out.append("c1ccccc1")
        return out

    # ---------------- visual ------------------------------------------------
    @staticmethod
    def visualize_fragment_importance(mol: Chem.Mol, fragments: Sequence[str]):
        highlight = set()
        for frag in fragments:
            q = Chem.MolFromSmiles(frag)
            if q:
                for match in mol.GetSubstructMatches(q):
                    highlight.update(match)
        return Draw.MolToImage(mol, highlightAtoms=list(highlight))
