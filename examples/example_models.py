from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import pickle
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adversarial_mol_gen.model_query import BlackBoxModel

def create_dummy_data(n_samples=1000):
    """Create dummy data for training a model."""
    # Generate random molecules
    molecules = []
    for _ in range(n_samples):
        # Generate simple molecules with 3-10 atoms
        n_atoms = np.random.randint(3, 11)
        smiles = 'C' * n_atoms  # Simple alkanes
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molecules.append(mol)
    
    # Use MorganGenerator instead of deprecated function
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X = np.zeros((len(molecules), 2048))
    for i, mol in enumerate(molecules):
        fp = generator.GetFingerprint(mol)
        X[i] = np.array(fp)

    # Generate random labels
    y = np.random.randint(0, 2, len(molecules))
    
    return X, y, molecules

def train_random_forest(X, y, output_path='models/random_forest.pkl'):
    """Train a random forest model and save it."""
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model

def train_logistic_regression(X, y, output_path='models/logistic_regression.pkl'):
    """Train a logistic regression model and save it."""
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model

def main():
    """Train example models."""
    print("Generating dummy data...")
    X, y, molecules = create_dummy_data(n_samples=1000)
    
    print("Training random forest model...")
    rf_model = train_random_forest(X, y)
    
    print("Training logistic regression model...")
    lr_model = train_logistic_regression(X, y)
    
    print("Testing models...")
    # Create a BlackBoxModel instance
    bb_model_rf = BlackBoxModel('models/random_forest.pkl')
    bb_model_lr = BlackBoxModel('models/logistic_regression.pkl')
    
    # Test prediction on a sample molecule
    test_mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
    pred_rf = bb_model_rf.predict(test_mol)
    pred_lr = bb_model_lr.predict(test_mol)
    
    print(f"Random Forest prediction for aspirin: {pred_rf:.4f}")
    print(f"Logistic Regression prediction for aspirin: {pred_lr:.4f}")
    
    print("Done!")

if __name__ == '__main__':
    main()
    