"""
Real-data version of the dummy-fingerprint demo.
Dataset  : MoleculeNet BBBP (binary BBB permeability)
Features : 2048-bit ECFP-like Morgan fingerprints (radius 2)
Models   : RandomForest, LogisticRegression
Author   : <your-name> – 2025-05-26
"""
import os, sys, pickle, io, tempfile
from pathlib import Path

import pandas as pd  # new
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# keep your project‐local import
from adversarial_mol_gen.model_query import BlackBoxModel


# ---------- 1.  DATA LOADING --------------------------------------------------
DATASET_URL   = (
    "https://raw.githubusercontent.com/"
    "GLambard/Molecules_Dataset_Collection/master/latest/BBBP.csv"
)
SMILES_COLUMN = "smiles"
LABEL_COLUMN  = "p_np"      # 1 = permeable, 0 = non-permeable


# ---------- 1.  DATA LOADING (robust) ----------------------------------------
def load_bbbp_dataset(url: str = DATASET_URL):
    """
    Download BBBP CSV, clean SMILES column, and return X, y, mols.
    """
    print("Downloading BBBP …")
    df = pd.read_csv(url, dtype={SMILES_COLUMN: str})   # force strings

    # Basic sanitisation ------------------------------------------------------
    df = df[[SMILES_COLUMN, LABEL_COLUMN]].dropna(subset=[SMILES_COLUMN])
    df[SMILES_COLUMN] = df[SMILES_COLUMN].str.strip()

    # Drop any rows whose SMILES field is now empty
    before, df = len(df), df[df[SMILES_COLUMN] != ""]
    print(f"Kept {len(df):,}/{before:,} rows with non-empty SMILES")

    # Convert to RDKit molecules and fingerprints ----------------------------
    mols, labels = [], []
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    for smi, label in zip(df[SMILES_COLUMN], df[LABEL_COLUMN]):
        mol = Chem.MolFromSmiles(smi)
        if mol:                                 # skip unparsable SMILES
            mols.append(mol)
            labels.append(int(label))

    print(f"Parsed {len(mols):,} valid molecules (dropped {len(df)-len(mols)})")
    X = np.vstack([np.array(gen.GetFingerprint(m)) for m in mols])
    y = np.asarray(labels, dtype=int)
    return X, y, mols



# ---------- 2.  TRAIN + SERIALISE  -------------------------------------------
def train_and_save(model, X_train, y_train, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.fit(X_train, y_train)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model


def main():
    X, y, mols = load_bbbp_dataset()

    # 80/20 split keeps evaluation honest
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest …")
    rf = train_and_save(
        RandomForestClassifier(n_estimators=200, random_state=42),
        X_train, y_train, Path("models/random_forest.pkl")
    )

    print("Training Logistic Regression …")
    lr = train_and_save(
        LogisticRegression(max_iter=4000, solver="liblinear", random_state=42),
        X_train, y_train, Path("models/logistic_regression.pkl")
    )

    # quick sanity check
    for name, mdl in [("RF", rf), ("LR", lr)]:
        acc = mdl.score(X_test, y_test)
        print(f"{name} test accuracy: {acc: .3f}")

    # ---------- 3.  BLACK-BOX WRAPPER TEST  ----------------------------------
    bb_rf = BlackBoxModel("models/random_forest.pkl")
    bb_lr = BlackBoxModel("models/logistic_regression.pkl")
    aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    print(f"RF black-box prob (aspirin): {bb_rf.predict(aspirin):.4f}")
    print(f"LR black-box prob (aspirin): {bb_lr.predict(aspirin):.4f}")


if __name__ == "__main__":
    main()
