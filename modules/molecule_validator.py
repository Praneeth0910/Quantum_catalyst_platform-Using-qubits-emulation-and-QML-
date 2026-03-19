from rdkit import Chem

def molecule_to_smiles(molecule_input: str) -> str:
    """
    Converts a formula, element, or common name to SMILES.
    """
    # Expanded mapping to handle common catalyst formulas and names
    mapping = {
        "Fe": "[Fe]",
        "Pt": "[Pt]",
        "NiO": "[Ni]=O",
        "PbNO3": "[Pb]([O-])[N+](=O)[O-]",
        "C": "C",
        "H2O": "O",
        "CO2": "C(=O)=O",
        "NH3": "N",
        "CH4": "C"
    }
    
    # 1. Check our mapping first for easy identification
    if molecule_input in mapping:
        return mapping[molecule_input]
    
    # 2. If not in mapping, try to parse the input directly as a SMILES string
    mol = Chem.MolFromSmiles(molecule_input)
    if mol:
        return molecule_input
        
    return None

def validate_molecule(smiles: str) -> bool:
    """
    Validates that the molecule is real and consists of 6 or fewer atoms.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
            
        # Add Hydrogens to the molecule to get the true atom count
        # For example, H2O has 1 Oxygen + 2 Hydrogens = 3 atoms
        mol_with_hs = Chem.AddHs(mol)
        atom_count = mol_with_hs.GetNumAtoms()
        
        # Enforce the constraint of 6 atoms for simulation stability
        if atom_count > 6:
            return False
            
        return True
    except Exception:
        return False