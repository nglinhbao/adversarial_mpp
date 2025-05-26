from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs
import numpy as np
import matplotlib.pyplot as plt

def mol_to_smiles(mol):
    """Convert RDKit molecule to SMILES string."""
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def smiles_to_mol(smiles):
    """Convert SMILES string to RDKit molecule."""
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)

def calculate_similarity(mol1, mol2):
    """Calculate Tanimoto similarity between two molecules."""
    if mol1 is None or mol2 is None:
        return 0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def visualize_molecules(molecules, labels=None, filename=None):
    """Visualize a list of molecules."""
    if labels is None:
        labels = [f'Molecule {i+1}' for i in range(len(molecules))]
    
    img = Draw.MolsToGridImage(molecules, molsPerRow=3, subImgSize=(300, 300), legends=labels)
    
    if filename:
        img.save(filename)
    
    return img

def get_property_change(mol_orig, mol_adv, black_box_model):
    """Get property change between original and adversarial molecules."""
    pred_orig = black_box_model.predict(mol_orig)
    pred_adv = black_box_model.predict(mol_adv)
    
    return {
        'original': pred_orig,
        'adversarial': pred_adv,
        'change': pred_adv - pred_orig,
        'change_pct': (pred_adv - pred_orig) / pred_orig * 100 if pred_orig != 0 else float('inf')
    }

def plot_property_changes(property_changes, filename=None):
    """Plot property changes for multiple adversarial examples."""
    labels = list(property_changes.keys())
    orig_values = [property_changes[k]['original'] for k in labels]
    adv_values = [property_changes[k]['adversarial'] for k in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, orig_values, width, label='Original')
    ax.bar(x + width/2, adv_values, width, label='Adversarial')
    
    ax.set_ylabel('Property Value')
    ax.set_title('Property Changes')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    if filename:
        plt.savefig(filename)
    
    return fig