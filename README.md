# Adversarial Molecule Generator

A framework for generating chemically valid adversarial molecules that can fool black-box Molecular Property Prediction (MPP) models, combining latent space perturbation with fragment-level substitution.

## Features

- Encodes molecules into a latent space using VAE or flow models
- Performs latent space perturbation to find potential adversarial regions
- Identifies important fragments for adversarial mutation
- Performs chemically valid fragment substitutions
- Queries black-box models to find successful adversarial examples
- Guarantees chemical validity of generated molecules

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from adversarial_mol_gen.main import AdversarialMoleculeGenerator
from rdkit import Chem

# Initialize the generator
generator = AdversarialMoleculeGenerator()

# Generate adversarial examples for aspirin
mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
adversarial_mols = generator.generate(mol)

# Print the results
for i, adv_mol in enumerate(adversarial_mols):
    print(f"Adversarial molecule {i+1}: {Chem.MolToSmiles(adv_mol)}")
```

## Method Overview

1. **Input Encoding**: Map molecule to latent vector using pretrained VAE or flow model
2. **Latent Space Perturbation**: Find adversarial direction in latent space
3. **Fragment Importance Analysis**: Identify key fragments for modification
4. **Fragment Substitution**: Generate chemically valid modifications
5. **Black-Box Model Query**: Evaluate candidates against target model
6. **Selection**: Choose optimal adversarial examples

## References

- CRAG (Zhang et al.): Adversarial search in molecular latent space
- Atomwise's Chimeric Molecules: Recombination of real-world fragments
- BRICS and CReM: Rule-based molecular mutation
- NAG-R (Zhou et al.): Efficient search without gradient access