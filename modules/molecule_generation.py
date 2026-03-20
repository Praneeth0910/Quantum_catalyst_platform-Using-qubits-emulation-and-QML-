from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Optional
import random

def generate_3d_molecule(smiles: str):
    """
    Generates a 3D RDKit molecule from SMILES.
    Returns the molecule object with 3D coordinates or None if failed.
    Handles fallback for small molecules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDG()
    success = AllChem.EmbedMolecule(mol, params)
    if success != 0:
        # Fallback: Compute 2D coords if 3D fails
        AllChem.Compute2DCoords(mol)
        return mol
    AllChem.UFFOptimizeMolecule(mol)
    return mol


def _is_valid_smiles(smiles: str) -> bool:
    """Return True if RDKit can parse SMILES."""
    return Chem.MolFromSmiles(smiles) is not None


def mutate_catalyst_smiles(base_smiles: str, seed: Optional[int] = None) -> Optional[str]:
    """
    Generate a simple catalyst mutation from a base catalyst SMILES.

    Mutations are chemistry-inspired but lightweight for fast exploration.
    """
    if seed is not None:
        random.seed(seed)

    mutation_map = {
        "[Fe]": ["[Fe]=O", "[Co]", "[Ni]"],
        "[Pt]": ["[Pd]", "[Rh]", "[Ru]"],
        "[Ni]": ["[Ni]=O", "[Co]", "[Cu]"],
        "[Cu]": ["[Cu]=O", "[Ag]", "[Au]"],
        "[Pd]": ["[Pt]", "[Rh]", "[Ni]"],
        "[Ru]": ["[Fe]", "[Rh]", "[Co]"],
    }

    candidates = mutation_map.get(base_smiles, [base_smiles])
    candidate = random.choice(candidates)
    return candidate if _is_valid_smiles(candidate) else None


def generate_catalyst_candidates(
    reaction_type: str,
    num_candidates: int = 5,
    seed: Optional[int] = 42
) -> List[str]:
    """
    Generate candidate catalyst SMILES for a target reaction.

    Uses a seeded random exploration strategy over known catalyst families,
    plus lightweight mutation for novelty.
    """
    if seed is not None:
        random.seed(seed)

    reaction_pools = {
        "H2_O2": ["[Pt]", "[Pd]", "[Rh]", "[Ru]", "[Ni]=O"],
        "N2_H2": ["[Fe]", "[Ru]", "[Co]", "[Fe]=O", "[Ni]"],
        "CO2_reduction": ["[Cu]", "[Ag]", "[Au]", "[Pd]", "[Ni]"],
    }

    base_pool = reaction_pools.get(reaction_type, ["[Pt]", "[Pd]", "[Fe]", "[Ni]", "[Cu]"])
    generated = []

    for _ in range(max(1, num_candidates * 3)):
        base = random.choice(base_pool)
        mutated = mutate_catalyst_smiles(base)
        chosen = mutated if mutated is not None else base

        if chosen not in generated and _is_valid_smiles(chosen):
            generated.append(chosen)

        if len(generated) >= num_candidates:
            break

    # Fallback to base pool if mutation did not produce enough unique candidates.
    for base in base_pool:
        if len(generated) >= num_candidates:
            break
        if base not in generated and _is_valid_smiles(base):
            generated.append(base)

    return generated[:num_candidates]
