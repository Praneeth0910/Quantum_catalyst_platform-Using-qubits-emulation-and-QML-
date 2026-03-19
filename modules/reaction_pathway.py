import random
import hashlib

def simulate_reaction_pathway(catalyst: str) -> dict:
    """
    Simulates a simple reaction pathway for a given catalyst.
    Returns states, energies, and energy barrier.
    """
    # Use catalyst hash for consistent results
    seed = int(hashlib.sha256(catalyst.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    states = ["Reactant", "Intermediate", "Product"]
    reactant_energy = rng.uniform(-1.0, -0.5)
    intermediate_energy = reactant_energy + rng.uniform(0.5, 1.5)
    product_energy = reactant_energy - rng.uniform(0.2, 1.0)
    
    energies = [reactant_energy, intermediate_energy, product_energy]
    barrier = intermediate_energy - reactant_energy

    return {
        "states": states,
        "energies": energies,
        "barrier": barrier
    }

def compute_catalyst_score(energies: list) -> float:
    """
    Computes a catalyst suitability score (0–100%) based on energy profile.
    Lower barrier and lower product energy are better.
    """
    reactant, intermediate, product = energies
    barrier = intermediate - reactant
    energy_drop = reactant - product

    # Normalize barrier (0 best, 1.5 worst)
    norm_barrier = min(max((barrier - 0.5) / (1.5 - 0.5), 0), 1)
    # Normalize energy drop (0.2 worst, 1.0 best)
    norm_drop = min(max((energy_drop - 0.2) / (1.0 - 0.2), 0), 1)

    # Weighted score: 60% energy drop, 40% barrier (lower is better)
    score = (0.6 * norm_drop + 0.4 * (1 - norm_barrier)) * 100
    return round(score, 2)