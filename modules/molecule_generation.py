from rdkit import Chem
from rdkit.Chem import AllChem

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
