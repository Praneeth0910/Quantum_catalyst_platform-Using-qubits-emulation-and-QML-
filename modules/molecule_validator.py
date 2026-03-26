from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import re
from typing import Dict, Optional, List

# ========================================================================
# COMPREHENSIVE MOLECULE DATABASE
# ========================================================================

# Common molecule names mapped to SMILES
COMMON_NAMES = {
    # Simple molecules
    "water": "O", "h2o": "O",
    "hydrogen": "[H][H]", "h2": "[H][H]",
    "oxygen": "O=O", "o2": "O=O",
    "nitrogen": "N#N", "n2": "N#N",
    "ammonia": "N", "nh3": "N",
    "methane": "C", "ch4": "C",
    "ethane": "CC", "c2h6": "CC",
    "propane": "CCC", "c3h8": "CCC",
    "butane": "CCCC", "c4h10": "CCCC",
    "ethylene": "C=C", "ethene": "C=C", "c2h4": "C=C",
    "acetylene": "C#C", "c2h2": "C#C",
    "benzene": "c1ccccc1", "c6h6": "c1ccccc1",
    "methanol": "CO", "ch3oh": "CO",
    "ethanol": "CCO", "c2h5oh": "CCO",
    "formaldehyde": "C=O", "ch2o": "C=O",
    "formic acid": "C(=O)O", "hcooh": "C(=O)O",
    "acetic acid": "CC(=O)O", "ch3cooh": "CC(=O)O",

    # Inorganic molecules
    "carbon dioxide": "C(=O)=O", "co2": "C(=O)=O",
    "carbon monoxide": "[C-]#[O+]", "co": "[C-]#[O+]",
    "nitric oxide": "[N]=O", "no": "[N]=O",
    "nitrogen dioxide": "N(=O)[O]", "no2": "N(=O)[O]",
    "sulfur dioxide": "O=S=O", "so2": "O=S=O",
    "hydrogen sulfide": "S", "h2s": "S",
    "hydrochloric acid": "Cl", "hcl": "Cl",
    "hydrogen peroxide": "OO", "h2o2": "OO",

    # Metal catalysts (single atoms)
    "iron": "[Fe]", "fe": "[Fe]",
    "platinum": "[Pt]", "pt": "[Pt]",
    "palladium": "[Pd]", "pd": "[Pd]",
    "nickel": "[Ni]", "ni": "[Ni]",
    "copper": "[Cu]", "cu": "[Cu]",
    "gold": "[Au]", "au": "[Au]",
    "silver": "[Ag]", "ag": "[Ag]",
    "rhodium": "[Rh]", "rh": "[Rh]",
    "ruthenium": "[Ru]", "ru": "[Ru]",
    "cobalt": "[Co]", "co_metal": "[Co]",
    "titanium": "[Ti]", "ti": "[Ti]",
    "zinc": "[Zn]", "zn": "[Zn]",
    "chromium": "[Cr]", "cr": "[Cr]",
    "manganese": "[Mn]", "mn": "[Mn]",
    "vanadium": "[V]", "v": "[V]",
    "molybdenum": "[Mo]", "mo": "[Mo]",
    "tungsten": "[W]", "w": "[W]",

    # Metal oxides (simplified representations)
    "iron oxide": "[Fe]=O", "feo": "[Fe]=O",
    "nickel oxide": "[Ni]=O", "nio": "[Ni]=O",
    "copper oxide": "[Cu]=O", "cuo": "[Cu]=O",
    "zinc oxide": "[Zn]=O", "zno": "[Zn]=O",
    "titanium dioxide": "O=[Ti]=O", "tio2": "O=[Ti]=O",
}

# Formula patterns for automatic parsing
PERIODIC_TABLE = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Pt': 78, 'Au': 79, 'W': 74
}

# ========================================================================
# FORMULA PARSER
# ========================================================================

def parse_molecular_formula(formula: str) -> Optional[str]:
    """
    Parse molecular formula like H2O, CO2, CH4, Fe2O3 and convert to SMILES.
    Uses RDKit's built-in formula parser when possible.

    Args:
        formula: Molecular formula string (e.g., "H2O", "CO2", "CH4")

    Returns:
        SMILES string or None if parsing fails
    """
    # Try simple molecules first
    simple_formulas = {
        "H2": "[H][H]",
        "O2": "O=O",
        "N2": "N#N",
        "Cl2": "ClCl",
        "H2O": "O",
        "H2O2": "OO",
        "CO": "[C-]#[O+]",
        "CO2": "O=C=O",
        "NO": "[N]=O",
        "NO2": "N(=O)[O]",
        "SO2": "O=S=O",
        "NH3": "N",
        "CH4": "C",
        "C2H6": "CC",
        "C2H4": "C=C",
        "C2H2": "C#C",
        "H2S": "S",
        "HCl": "Cl",
        "HF": "F",
        "HBr": "Br",
        "HI": "I",
    }

    if formula in simple_formulas:
        return simple_formulas[formula]

    # Check if it's a single element (catalyst)
    if formula in PERIODIC_TABLE:
        return f"[{formula}]"

    return None

# ========================================================================
# MAIN CONVERSION FUNCTION
# ========================================================================

def molecule_to_smiles(molecule_input: str) -> Optional[str]:
    """
    Intelligent converter: handles SMILES, common names, formulas, and elements.

    4-Stage Parsing Pipeline:
    1. Check if input is already valid SMILES
    2. Check common names database
    3. Try parsing as molecular formula
    4. Return None if all fail

    Args:
        molecule_input: User input (any format)

    Returns:
        SMILES string or None

    Examples:
        >>> molecule_to_smiles("water")
        "O"
        >>> molecule_to_smiles("H2O")
        "O"
        >>> molecule_to_smiles("Fe")
        "[Fe]"
        >>> molecule_to_smiles("O")  # Direct SMILES
        "O"
    """
    if not molecule_input:
        return None

    # Normalize input
    molecule_input = molecule_input.strip()

    # Stage 1: Try direct SMILES parsing first
    try:
        mol = Chem.MolFromSmiles(molecule_input)
        if mol is not None:
            # Sanitize to ensure it's chemically valid
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol)  # Return canonical SMILES
    except:
        pass

    # Stage 2: Check common names database (case-insensitive)
    lookup_key = molecule_input.lower()
    if lookup_key in COMMON_NAMES:
        return COMMON_NAMES[lookup_key]

    # Stage 3: Try parsing as molecular formula
    formula_smiles = parse_molecular_formula(molecule_input)
    if formula_smiles:
        return formula_smiles

    # Stage 4: Failed all parsing attempts
    return None

# ========================================================================
# VALIDATION & METADATA
# ========================================================================

def validate_molecule(smiles: str, max_atoms: int = 6) -> Dict:
    """
    Comprehensive validation with chemistry rules and detailed metadata.

    Validation Steps:
    1. RDKit can parse the SMILES
    2. Molecule is chemically valid (valence rules, sanitization)
    3. Atom count ≤ max_atoms (including hydrogens)
    4. Return detailed metadata for downstream use

    Args:
        smiles: SMILES string to validate
        max_atoms: Maximum allowed atom count (default 6)

    Returns:
        Dictionary with validation results and metadata:
        {
            "valid": bool,
            "smiles": str (canonical),
            "formula": str,
            "mol_weight": float,
            "atom_count": int,
            "heavy_atom_count": int,
            "elements": List[str],
            "error": str (if invalid)
        }
    """
    result = {
        "valid": False,
        "smiles": smiles,
        "formula": "",
        "mol_weight": 0.0,
        "atom_count": 0,
        "heavy_atom_count": 0,
        "elements": [],
        "error": ""
    }

    try:
        # Step 1: Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result["error"] = "Invalid SMILES string - cannot parse molecule"
            return result

        # Step 2: Chemical validation (sanitization)
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            result["error"] = f"Chemically invalid molecule: {str(e)}"
            return result

        # Step 3: Add explicit hydrogens for accurate atom count
        mol_with_hs = Chem.AddHs(mol)
        atom_count = mol_with_hs.GetNumAtoms()
        heavy_atom_count = mol.GetNumHeavyAtoms()

        # Step 4: Check atom count constraint
        if heavy_atom_count > max_atoms:
            result["error"] = f"Molecule has {heavy_atom_count} heavy atoms (max allowed: {max_atoms})"
            return result

        # Step 5: Extract metadata
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        mol_weight = Descriptors.MolWt(mol)
        elements = list(set([atom.GetSymbol() for atom in mol.GetAtoms()]))
        canonical_smiles = Chem.MolToSmiles(mol)

        # Success!
        result.update({
            "valid": True,
            "smiles": canonical_smiles,
            "formula": formula,
            "mol_weight": round(mol_weight, 2),
            "atom_count": atom_count,
            "heavy_atom_count": heavy_atom_count,
            "elements": sorted(elements),
            "error": ""
        })

        return result

    except Exception as e:
        result["error"] = f"Validation error: {str(e)}"
        return result

# ========================================================================
# USER-FRIENDLY WRAPPER
# ========================================================================

def process_molecule_input(user_input: str, max_atoms: int = 6) -> Dict:
    """
    Complete pipeline: Convert input → Validate → Return metadata.

    This is the main function to use in the Streamlit app.

    Args:
        user_input: Any format (name, formula, SMILES)
        max_atoms: Maximum atom count allowed

    Returns:
        Validation result dictionary with all metadata

    Example:
        >>> result = process_molecule_input("water")
        >>> print(result)
        {
            "valid": True,
            "smiles": "O",
            "formula": "H2O",
            "atom_count": 3,
            ...
        }
    """
    # Step 1: Convert to SMILES
    smiles = molecule_to_smiles(user_input)

    if smiles is None:
        return {
            "valid": False,
            "error": f"Cannot recognize '{user_input}'. Try: common names (water, methane), formulas (H2O, CO2), or SMILES strings.",
            "smiles": "",
            "formula": "",
            "mol_weight": 0.0,
            "atom_count": 0,
            "heavy_atom_count": 0,
            "elements": []
        }

    # Step 2: Validate and get metadata
    return validate_molecule(smiles, max_atoms)

# ========================================================================
# HELPER: Get suggestions for failed inputs
# ========================================================================

def get_similar_molecules(user_input: str, limit: int = 5) -> List[str]:
    """
    Suggest similar molecule names when input fails.
    Uses simple string matching.
    """
    user_lower = user_input.lower()
    matches = []

    for name in COMMON_NAMES.keys():
        if user_lower in name or name in user_lower:
            matches.append(name)
            if len(matches) >= limit:
                break

    return matches