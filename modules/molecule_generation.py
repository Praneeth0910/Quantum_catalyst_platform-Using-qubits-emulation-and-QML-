from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Optional, Set
import random


METAL_SWAP_OPTIONS = {
    "Pt": ["Pd", "Ni", "Cu", "Rh", "Ru"],
    "Pd": ["Pt", "Ni", "Cu", "Rh"],
    "Ni": ["Co", "Fe", "Cu", "Pd"],
    "Cu": ["Ag", "Au", "Ni", "Pd"],
    "Fe": ["Co", "Ni", "Ru"],
    "Ru": ["Rh", "Fe", "Co"],
    "Rh": ["Ru", "Pd", "Pt"],
}

LIGAND_VARIANTS = ["=O", "O", "C"]
DOPING_ATOMS = ["N", "O", "S"]

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
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    success = AllChem.EmbedMolecule(mol, params)
    if success != 0:
        # Fallback: Compute 2D coords if 3D fails
        AllChem.Compute2DCoords(mol)
        return mol
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        # Keep embedded geometry even if force field optimization fails.
        pass
    return mol


def _is_valid_smiles(smiles: str) -> bool:
    """Return True if RDKit can parse SMILES."""
    return Chem.MolFromSmiles(smiles) is not None


def _sanitize_and_canonicalize(smiles: str) -> Optional[str]:
    """Return canonical sanitized SMILES or None for invalid molecules."""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _metal_symbols_in_molecule(mol: Chem.Mol) -> List[str]:
    """Return unique metal symbols from a molecule in insertion order."""
    seen = set()
    metals = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in METAL_SWAP_OPTIONS and symbol not in seen:
            seen.add(symbol)
            metals.append(symbol)
    return metals


def mutate_catalyst(base_smiles: str, num_variations: int = 5) -> List[str]:
    """
    Generate mutated catalyst candidates from a base SMILES string.

    Mutation strategies:
    1. Metal swapping
    2. Ligand addition (=O, -OH equivalent via O, methyl via C)
    3. Peripheral-atom doping

    Returns only unique, sanitized canonical SMILES strings.
    """
    base_mol = Chem.MolFromSmiles(base_smiles)
    if base_mol is None:
        return []

    base_canonical = _sanitize_and_canonicalize(base_smiles)
    valid_mutants: List[str] = []
    seen: Set[str] = set()

    def add_candidate(candidate_smiles: str):
        canonical = _sanitize_and_canonicalize(candidate_smiles)
        if canonical is None:
            return
        if base_canonical is not None and canonical == base_canonical:
            return
        if canonical in seen:
            return
        seen.add(canonical)
        valid_mutants.append(canonical)

    # 1) Metal-swapping mutations using substructure replacement.
    metals = _metal_symbols_in_molecule(base_mol)
    for metal in metals:
        query = Chem.MolFromSmarts(f"[{metal}]")
        if query is None:
            continue
        for replacement_symbol in METAL_SWAP_OPTIONS.get(metal, []):
            replacement = Chem.MolFromSmiles(f"[{replacement_symbol}]")
            if replacement is None:
                continue
            replaced = Chem.ReplaceSubstructs(base_mol, query, replacement, replaceAll=False)
            for mol in replaced:
                add_candidate(Chem.MolToSmiles(mol, canonical=True))

    # 2) Ligand addition for single-metal centers.
    if base_mol.GetNumAtoms() == 1 and base_mol.GetAtomWithIdx(0).GetSymbol() in METAL_SWAP_OPTIONS:
        metal = base_mol.GetAtomWithIdx(0).GetSymbol()
        for ligand in LIGAND_VARIANTS:
            add_candidate(f"[{metal}]{ligand}")

    # 3) Doping: replace first peripheral non-metal with common hetero dopants.
    atom_symbols = [atom.GetSymbol() for atom in base_mol.GetAtoms()]
    for idx, symbol in enumerate(atom_symbols):
        if symbol in METAL_SWAP_OPTIONS or symbol == "H":
            continue
        for dopant in DOPING_ATOMS:
            if dopant == symbol:
                continue
            editable = Chem.RWMol(base_mol)
            editable.GetAtomWithIdx(idx).SetAtomicNum(Chem.Atom(dopant).GetAtomicNum())
            add_candidate(Chem.MolToSmiles(editable.GetMol(), canonical=True))
        break

    if not valid_mutants:
        return []

    random.shuffle(valid_mutants)
    return valid_mutants[:max(0, num_variations)]


def mutate_catalyst_smiles(base_smiles: str, seed: Optional[int] = None) -> Optional[str]:
    """
    Generate a simple catalyst mutation from a base catalyst SMILES.

    Mutations are chemistry-inspired but lightweight for fast exploration.
    """
    if seed is not None:
        random.seed(seed)

    candidates = mutate_catalyst(base_smiles, num_variations=5)
    if not candidates:
        return None
    return random.choice(candidates)


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
        variants = mutate_catalyst(base, num_variations=2)
        if variants:
            for chosen in variants:
                if chosen not in generated and _is_valid_smiles(chosen):
                    generated.append(chosen)
                    if len(generated) >= num_candidates:
                        break
        elif base not in generated and _is_valid_smiles(base):
            generated.append(base)

        if len(generated) >= num_candidates:
            break

    # Fallback to base pool if mutation did not produce enough unique candidates.
    for base in base_pool:
        if len(generated) >= num_candidates:
            break
        if base not in generated and _is_valid_smiles(base):
            generated.append(base)

    return generated[:num_candidates]
