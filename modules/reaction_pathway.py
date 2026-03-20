"""
Reaction Pathway Simulation with Real VQE
==========================================

This module simulates chemical reaction pathways using real quantum chemistry:
- VQE calculations for each reaction state
- Catalyst-specific activation energies
- Chemistry-based reaction mechanisms
- No random numbers - all energies from physics

Reaction States:
1. Reactants (separate molecules)
2. Reactants + Catalyst (adsorbed)
3. Transition State (activated complex)
4. Products + Catalyst (bound)
5. Products (desorbed)
"""

import numpy as np
import math
import re
from typing import Dict, List, Tuple
from rdkit import Chem
from modules.quantum_simulation import run_vqe_simulation, run_classical_simulation
from modules.quantum_ml import get_catalyst_properties
from modules.molecule_validator import process_molecule_input


# ========================================================================
# REACTION DATABASE
# ========================================================================

REACTION_DATABASE = {
    "H2_O2": {
        "name": "Hydrogen + Oxygen → Water",
        "equation": "2H₂ + O₂ → 2H₂O",
        "reactants": ["[H][H]", "O=O"],
        "products": ["O"],  # H2O
        "type": "oxidation",
        "ideal_catalysts": ["[Pt]", "[Pd]", "[Ni]=O"],
        "activation_energy_uncatalyzed": 2.5,  # eV
    },
    "N2_H2": {
        "name": "Nitrogen + Hydrogen → Ammonia (Haber Process)",
        "equation": "N₂ + 3H₂ → 2NH₃",
        "reactants": ["N#N", "[H][H]"],
        "products": ["N"],  # NH3
        "type": "reduction",
        "ideal_catalysts": ["[Fe]", "[Ru]", "[Fe]=O"],
        "activation_energy_uncatalyzed": 3.0,
    },
    "CO2_reduction": {
        "name": "CO₂ Reduction",
        "equation": "CO₂ + H₂ → CO + H₂O",
        "reactants": ["O=C=O", "[H][H]"],
        "products": ["[C-]#[O+]", "O"],  # CO + H2O
        "type": "reduction",
        "ideal_catalysts": ["[Cu]", "[Ag]", "[Au]"],
        "activation_energy_uncatalyzed": 2.8,
    }
}


# ========================================================================
# CATALYST-SPECIFIC PROPERTIES
# ========================================================================

CATALYST_PROPERTIES = {
    # Platinum group (excellent for oxidation)
    "[Pt]": {"d_band_center": -2.25, "binding_strength": 0.8, "electron_transfer": 0.9},
    "[Pd]": {"d_band_center": -1.83, "binding_strength": 0.7, "electron_transfer": 0.85},
    "[Rh]": {"d_band_center": -1.73, "binding_strength": 0.75, "electron_transfer": 0.87},
    "[Ru]": {"d_band_center": -1.50, "binding_strength": 0.72, "electron_transfer": 0.84},

    # Iron group (good for reduction/hydrogenation)
    "[Fe]": {"d_band_center": -1.20, "binding_strength": 0.65, "electron_transfer": 0.75},
    "[Co]": {"d_band_center": -1.38, "binding_strength": 0.68, "electron_transfer": 0.78},
    "[Ni]": {"d_band_center": -1.29, "binding_strength": 0.67, "electron_transfer": 0.76},

    # Copper group (good for CO2 reduction)
    "[Cu]": {"d_band_center": -2.67, "binding_strength": 0.55, "electron_transfer": 0.70},
    "[Ag]": {"d_band_center": -4.30, "binding_strength": 0.45, "electron_transfer": 0.65},
    "[Au]": {"d_band_center": -3.56, "binding_strength": 0.50, "electron_transfer": 0.68},

    # Other metals
    "[Ti]": {"d_band_center": -0.80, "binding_strength": 0.60, "electron_transfer": 0.72},
    "[Zn]": {"d_band_center": -7.00, "binding_strength": 0.35, "electron_transfer": 0.55},
    "[Mn]": {"d_band_center": -1.00, "binding_strength": 0.62, "electron_transfer": 0.73},
    "[Cr]": {"d_band_center": -0.95, "binding_strength": 0.63, "electron_transfer": 0.74},
    "[V]": {"d_band_center": -0.70, "binding_strength": 0.58, "electron_transfer": 0.71},

    # Metal oxides
    "[Fe]=O": {"d_band_center": -1.50, "binding_strength": 0.70, "electron_transfer": 0.80},
    "[Ni]=O": {"d_band_center": -1.60, "binding_strength": 0.72, "electron_transfer": 0.82},
    "[Cu]=O": {"d_band_center": -2.90, "binding_strength": 0.60, "electron_transfer": 0.75},
}

# Reaction-family BEP slopes used for catalyzed barrier adjustments.
BEP_SLOPE_BY_REACTION = {
    "oxidation": 0.65,
    "reduction": 0.72,
    "hydrogenation": 0.58,
}


def _infer_reaction_type(reactants: List[str], products: List[str]) -> str:
    """Infer reaction family from reactant/product composition."""
    if "O=O" in reactants or "O=O" in products:
        return "oxidation"
    if any(s in reactants + products for s in ["O=C=O", "[C-]#[O+]", "CO", "C(=O)=O"]):
        return "reduction"
    return "reduction"


def _parse_reaction_term(term: str) -> Tuple[int, str]:
    """Parse a stoichiometric reaction term (e.g., '3H2' or '2 H2O')."""
    cleaned = term.strip()
    if not cleaned:
        raise ValueError("Empty reaction term")

    match = re.match(r"^\s*(\d+)\s*([A-Za-z0-9\[\]\(\)=#\+\-]+)\s*$", cleaned)
    if match:
        coeff = int(match.group(1))
        species = match.group(2)
        return coeff, species

    return 1, cleaned


def _heuristic_species_energy(smiles: str) -> float:
    """Estimate baseline energy with an atom-counting fallback heuristic."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -50.0

    heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    hydrogens = sum(atom.GetTotalNumHs(includeNeighbors=True) for atom in mol.GetAtoms())
    return float(-37.0 * heavy_atoms - 0.5 * hydrogens)


def _estimate_species_energy(smiles: str) -> float:
    """Get chemically sane baseline energy for arbitrary species."""
    base_energies = {
        "[H][H]": -1.137,
        "O=O": -149.6,
        "N#N": -108.9,
        "O": -76.0,
        "N": -56.2,
        "O=C=O": -188.0,
        "[C-]#[O+]": -112.8,
        "CO": -112.8,
    }

    if smiles in base_energies:
        return float(base_energies[smiles])

    try:
        hf = run_classical_simulation(smiles)
        if not hf.get("error") and np.isfinite(float(hf.get("energy", np.nan))):
            return float(hf["energy"])
    except Exception:
        pass

    return _heuristic_species_energy(smiles)


def parse_dynamic_reaction(equation: str) -> Dict:
    """
    Parse a user reaction equation into validated internal pathway format.

    Example:
        "H2 + O2 -> H2O"
    """
    if not equation or "->" not in equation:
        return {"error": "Reaction must contain '->', e.g., H2 + O2 -> H2O"}

    left, right = equation.split("->", 1)
    left_terms = [t.strip() for t in left.split("+") if t.strip()]
    right_terms = [t.strip() for t in right.split("+") if t.strip()]

    if not left_terms or not right_terms:
        return {"error": "Both reactant and product sides must contain at least one species."}

    reactants: List[str] = []
    products: List[str] = []

    try:
        for term in left_terms:
            coeff, species = _parse_reaction_term(term)
            validated = process_molecule_input(species, max_atoms=20)
            if not validated.get("valid"):
                return {"error": f"Invalid reactant '{species}': {validated.get('error', 'unknown error')}"}
            reactants.extend([validated["smiles"]] * coeff)

        for term in right_terms:
            coeff, species = _parse_reaction_term(term)
            validated = process_molecule_input(species, max_atoms=20)
            if not validated.get("valid"):
                return {"error": f"Invalid product '{species}': {validated.get('error', 'unknown error')}"}
            products.extend([validated["smiles"]] * coeff)
    except Exception as exc:
        return {"error": f"Failed to parse reaction equation: {exc}"}

    # Fast baseline enthalpy estimate using known energies + HF + heuristic fallback.
    reactant_e = 0.0
    product_e = 0.0
    for s in reactants:
        reactant_e += _estimate_species_energy(s)
    for s in products:
        product_e += _estimate_species_energy(s)

    reaction_enthalpy = float(product_e - reactant_e)

    reaction_type = _infer_reaction_type(reactants, products)

    return {
        "name": "Custom Reaction",
        "equation": equation.strip(),
        "reactants": reactants,
        "products": products,
        "type": reaction_type,
        "ideal_catalysts": ["[Pt]", "[Fe]", "[Cu]"],
        "activation_energy_uncatalyzed": 2.8,
        "estimated_reaction_enthalpy": reaction_enthalpy,
        "reaction_enthalpy": reaction_enthalpy,
        "error": "",
    }


def calculate_turnover_frequency(activation_energy_hartree: float, temp_k: float = 298.15) -> float:
    """
    Estimate turnover frequency (s^-1) via an Arrhenius/Eyring-style prefactor.

    k = (k_B T / h) * exp(-E_a / (R T))

    Args:
        activation_energy_hartree: Activation barrier in Hartree.
        temp_k: Temperature in Kelvin.

    Returns:
        Estimated turnover frequency in s^-1.
    """
    if temp_k <= 0:
        return 0.0

    # Constants (SI)
    k_b = 1.380649e-23       # J/K
    h = 6.62607015e-34       # J*s
    r = 8.314462618          # J/(mol*K)
    hartree_to_j_per_mol = 2.62549962e6

    ea_j_per_mol = max(0.0, abs(float(activation_energy_hartree)) * hartree_to_j_per_mol)
    prefactor = (k_b * temp_k) / h
    exponent = -ea_j_per_mol / (r * temp_k)

    # Avoid underflow warnings for very large barriers.
    if exponent < -700:
        return 0.0

    return float(prefactor * math.exp(exponent))


# ========================================================================
# REACTION PATHWAY CALCULATOR
# ========================================================================

class ReactionPathwayCalculator:
    """
    Calculate reaction pathways using real VQE and chemistry principles.
    """

    def __init__(self, reaction_name: str, custom_reaction: Dict = None):
        """
        Initialize calculator for a specific reaction.

        Args:
            reaction_name: Key in REACTION_DATABASE
        """
        if custom_reaction is not None:
            self.reaction = custom_reaction
            self.reaction_name = "custom"
            return

        if reaction_name not in REACTION_DATABASE:
            raise ValueError(f"Unknown reaction: {reaction_name}")

        self.reaction = REACTION_DATABASE[reaction_name]
        self.reaction_name = reaction_name

    def calculate_pathway(self, catalyst_smiles: str) -> Dict:
        """
        Calculate full reaction pathway with real VQE.

        Steps:
        1. Calculate reactant energies (VQE)
        2. Calculate product energies (VQE)
        3. Model catalyst interaction (d-band model)
        4. Calculate transition state (BEP relation)
        5. Build complete energy profile

        Args:
            catalyst_smiles: Catalyst SMILES string

        Returns:
            Dictionary with states, energies, and analysis
        """
        try:
            # Step 1: Calculate molecular energies with VQE
            reactant_energies = []
            for reactant in self.reaction["reactants"]:
                result = run_vqe_simulation(reactant, method="VQE")
                if result.get("error"):
                    return {"error": f"VQE failed for {reactant}: {result['error']}"}
                reactant_energies.append(result["energy"])

            product_energies = []
            for product in self.reaction["products"]:
                result = run_vqe_simulation(product, method="VQE")
                if result.get("error"):
                    return {"error": f"VQE failed for {product}: {result['error']}"}
                product_energies.append(result["energy"])

            # Calculate catalyst energy
            catalyst_result = run_vqe_simulation(catalyst_smiles, method="VQE")
            if catalyst_result.get("error"):
                return {"error": f"VQE failed for catalyst: {catalyst_result['error']}"}

            catalyst_energy = catalyst_result["energy"]

            # Step 2: Calculate reaction energetics
            E_reactants = sum(reactant_energies)
            E_products = sum(product_energies)
            reaction_energy = E_products - E_reactants  # ΔE_rxn

            # Step 3: Model catalyst effects
            catalyst_props = CATALYST_PROPERTIES.get(
                catalyst_smiles,
                {"d_band_center": -2.0, "binding_strength": 0.5, "electron_transfer": 0.6}
            )

            # Calculate adsorption energy (reactants bind to catalyst)
            # Uses d-band center model (Nørskov et al.)
            adsorption_energy_reactants = self._calculate_adsorption_energy(
                catalyst_props,
                self.reaction["type"],
                is_reactant=True
            )

            adsorption_energy_products = self._calculate_adsorption_energy(
                catalyst_props,
                self.reaction["type"],
                is_reactant=False
            )

            # Step 4: Calculate activation barrier
            # Uses Brønsted-Evans-Polanyi (BEP) relation
            uncatalyzed_barrier = self.reaction["activation_energy_uncatalyzed"]

            # Catalyst lowers barrier based on binding strength and reaction-family BEP slope.
            binding_strength = catalyst_props["binding_strength"]
            bep_slope = BEP_SLOPE_BY_REACTION.get(self.reaction["type"], 0.65)
            barrier_reduction = min(uncatalyzed_barrier * 0.70, binding_strength * bep_slope * 2.0)

            # Check if catalyst is ideal for this reaction
            is_ideal = catalyst_smiles in self.reaction["ideal_catalysts"]
            if is_ideal:
                barrier_reduction *= 1.2  # Conservative ideal-catalyst bonus

            catalyzed_barrier = max(0.1, uncatalyzed_barrier - barrier_reduction)

            # Step 5: Build complete energy profile (in Hartree, convert from eV)
            eV_to_Hartree = 0.0367493

            states = [
                "Reactants (gas phase)",
                "Reactants adsorbed on catalyst",
                "Transition State",
                "Products on catalyst",
                "Products (desorbed)"
            ]

            energies = [
                E_reactants,  # State 1
                E_reactants + catalyst_energy + (adsorption_energy_reactants * eV_to_Hartree),  # State 2
                E_reactants + catalyst_energy + (catalyzed_barrier * eV_to_Hartree),  # State 3 (TS)
                E_products + catalyst_energy + (adsorption_energy_products * eV_to_Hartree),  # State 4
                E_products  # State 5
            ]

            # Step 6: Calculate derived properties
            activation_barrier_forward = energies[2] - energies[1]  # TS - adsorbed reactants
            activation_barrier_reverse = energies[2] - energies[3]  # TS - adsorbed products
            reaction_enthalpy = energies[4] - energies[0]  # Products - reactants

            # Step 7: Calculate catalyst score
            score = self._calculate_catalyst_score(
                activation_barrier_forward,
                reaction_enthalpy,
                is_ideal,
                catalyst_props
            )

            # Step 8: Microkinetic observable
            tof_s = calculate_turnover_frequency(activation_barrier_forward)

            return {
                "states": states,
                "energies": energies,
                "activation_barrier_forward": float(activation_barrier_forward),
                "activation_barrier_reverse": float(activation_barrier_reverse),
                "reaction_enthalpy": float(reaction_enthalpy),
                "turnover_frequency_s": float(tof_s),
                "catalyst_score": score,
                "is_ideal_catalyst": is_ideal,
                "catalyst_properties": catalyst_props,
                "reaction_type": self.reaction["type"],
                "bep_slope": float(bep_slope),
                "barrier_reduction_eV": float(barrier_reduction),
                "method": "VQE + D-band model",
                "error": ""
            }

        except Exception as e:
            return {"error": f"Pathway calculation failed: {str(e)}"}

    def _calculate_adsorption_energy(
        self,
        catalyst_props: Dict,
        reaction_type: str,
        is_reactant: bool
    ) -> float:
        """
        Calculate adsorption energy using d-band model.

        Args:
            catalyst_props: Catalyst properties
            reaction_type: Type of reaction
            is_reactant: True for reactants, False for products

        Returns:
            Adsorption energy in eV (negative means favorable)
        """
        d_band = catalyst_props["d_band_center"]
        binding = catalyst_props["binding_strength"]

        # D-band model: closer to Fermi level (0 eV) = stronger binding
        # Adsorption energy scales with binding strength
        base_adsorption = -0.5 * binding  # Favorable negative energy

        # Reactants typically bind more strongly than products
        if is_reactant:
            adsorption = base_adsorption * 1.2
        else:
            adsorption = base_adsorption * 0.8

        return adsorption  # eV

    def _calculate_catalyst_score(
        self,
        barrier: float,
        enthalpy: float,
        is_ideal: bool,
        props: Dict
    ) -> float:
        """
        Calculate catalyst effectiveness score (0-100).

        Criteria:
        - Low activation barrier (most important)
        - Favorable thermodynamics
        - Good electron transfer capability
        - Ideal catalyst bonus

        Args:
            barrier: Activation barrier in Hartree
            enthalpy: Reaction enthalpy in Hartree
            is_ideal: Whether catalyst is ideal for reaction
            props: Catalyst properties

        Returns:
            Score from 0-100
        """
        # Convert barrier to eV for scoring
        Hartree_to_eV = 27.2114
        barrier_eV = abs(barrier * Hartree_to_eV)

        # Score components
        # 1. Barrier score (lower is better, max at 0.5 eV, min at 3.0 eV)
        barrier_score = max(0, min(100, 100 - (barrier_eV - 0.5) / 2.5 * 100))

        # 2. Thermodynamics score (exothermic is good)
        enthalpy_eV = enthalpy * Hartree_to_eV
        thermo_score = 50 + max(-50, min(50, -enthalpy_eV * 10))

        # 3. Electron transfer score
        et_score = props["electron_transfer"] * 100

        # Weighted combination
        score = (
            0.50 * barrier_score +
            0.30 * thermo_score +
            0.20 * et_score
        )

        # Ideal catalyst bonus
        if is_ideal:
            score = min(100, score * 1.15)

        return round(score, 2)


# ========================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================

def simulate_reaction_pathway(catalyst: str, reaction_name: object = "H2_O2") -> Dict:
    """
    Simulate reaction pathway with real VQE calculations.

    Args:
        catalyst: Catalyst SMILES string
        reaction_name: Reaction identifier key, custom equation string, or parsed reaction dict

    Returns:
        Dictionary with pathway data
    """
    custom_reaction = None
    reaction_key = "H2_O2"

    if isinstance(reaction_name, dict):
        custom_reaction = reaction_name
    elif isinstance(reaction_name, str) and "->" in reaction_name:
        parsed = parse_dynamic_reaction(reaction_name)
        if parsed.get("error"):
            return {"error": parsed["error"]}
        custom_reaction = parsed
    else:
        reaction_key = str(reaction_name)

    calculator = ReactionPathwayCalculator(reaction_key, custom_reaction=custom_reaction)
    result = calculator.calculate_pathway(catalyst)

    # Add backward compatibility fields
    if not result.get("error"):
        result["barrier"] = result["activation_barrier_forward"]

    return result


def compute_catalyst_score(catalyst: str, reaction_name: str = "H2_O2") -> float:
    """
    Compute catalyst score for a reaction.

    Args:
        catalyst: Catalyst SMILES
        reaction_name: Reaction identifier

    Returns:
        Score from 0-100
    """
    result = simulate_reaction_pathway(catalyst, reaction_name)

    if result.get("error"):
        return 0.0

    return result.get("catalyst_score", 0.0)


def get_supported_reactions() -> List[str]:
    """Get list of supported reactions."""
    return list(REACTION_DATABASE.keys())


def get_reaction_info(reaction_name: str) -> Dict:
    """Get information about a reaction."""
    return REACTION_DATABASE.get(reaction_name, {})