"""
Quantum Simulation Engine with VQE
===================================

This module provides quantum chemistry simulations using:
1. Real VQE (Variational Quantum Eigensolver)
2. Custom Hamiltonian database (no PySCF dependency issues)
3. Scientifically accurate results for small molecules

Features:
- Works without PySCF installation
- Uses pre-computed molecular Hamiltonians
- Runs actual VQE optimization (not fake!)
- Proper convergence tracking
"""

import numpy as np
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP, COBYLA
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from modules.hamiltonian_database import get_hamiltonian_db, smiles_to_xyz
from typing import Dict, Optional


def run_vqe_simulation(smiles: str, method: str = "VQE") -> Dict:
    """
    Run VQE simulation on a molecule using pre-computed Hamiltonian.

    This function performs REAL quantum simulation without needing PySCF:
    1. Retrieves molecular Hamiltonian from database
    2. Constructs parameterized quantum circuit (ansatz)
    3. Runs VQE optimization to find ground state energy
    4. Returns energy, convergence data, and metadata

    Args:
        smiles: Canonical SMILES string of the molecule
        method: Simulation method ("VQE", "VQE-COBYLA", or "HF")

    Returns:
        Dictionary containing:
        - energy: Ground state energy in Hartree
        - iterations: Number of optimization iterations
        - convergence: List of energies during optimization
        - num_qubits: Number of qubits used
        - method: Method used for simulation
        - error: Error message if simulation failed
    """
    try:
        # Step 1: Get Hamiltonian from database
        db = get_hamiltonian_db()

        if not db.has_molecule(smiles):
            return {
                "error": f"Molecule '{smiles}' not in database. Supported molecules: {len(db.get_supported_molecules())}",
                "energy": 0,
                "convergence": [0],
                "iterations": 0,
                "num_qubits": 0,
                "method": method
            }

        hamiltonian, nuclear_repulsion, reference_energy, num_qubits = db.get_hamiltonian(smiles)

        # Step 2: Choose simulation method
        if method == "HF":
            # Hartree-Fock approximation (classical reference)
            return {
                "energy": float(reference_energy),
                "iterations": 0,
                "convergence": [reference_energy],
                "num_qubits": num_qubits,
                "method": "Hartree-Fock (Classical)",
                "error": ""
            }

        # Step 3: Set up VQE components
        # Choose ansatz (quantum circuit template)
        if num_qubits <= 2:
            ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2)
        else:
            ansatz = EfficientSU2(num_qubits=num_qubits, reps=2)

        # Choose optimizer
        if method == "VQE-COBYLA":
            optimizer = COBYLA(maxiter=100)
        else:
            optimizer = SLSQP(maxiter=100)

        # Set up quantum estimator (uses statevector simulation)
        estimator = StatevectorEstimator()

        # Step 4: Run VQE optimization
        convergence = []

        def callback(eval_count, parameters, mean, std=None):
            """Track convergence during optimization."""
            convergence.append(float(mean))

        vqe = VQE(estimator, ansatz, optimizer, callback=callback)
        result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

        # Step 5: Extract results
        vqe_energy = result.eigenvalue.real
        total_energy = vqe_energy  # Hamiltonian already includes nuclear repulsion

        return {
            "energy": float(total_energy),
            "iterations": len(convergence),
            "convergence": convergence,
            "num_qubits": num_qubits,
            "method": f"VQE (Optimizer: {optimizer.__class__.__name__})",
            "optimal_parameters": result.optimal_parameters.tolist() if hasattr(result.optimal_parameters, 'tolist') else [],
            "error": ""
        }

    except Exception as e:
        return {
            "error": f"VQE simulation error: {str(e)}",
            "energy": 0,
            "convergence": [0],
            "iterations": 0,
            "num_qubits": 0,
            "method": method
        }


def run_classical_simulation(smiles: str) -> Dict:
    """
    Run classical Hartree-Fock simulation for comparison.

    Args:
        smiles: Canonical SMILES string

    Returns:
        Dictionary with HF results
    """
    return run_vqe_simulation(smiles, method="HF")


def compare_methods(smiles: str) -> Dict:
    """
    Compare quantum (VQE) vs classical (HF) methods.

    Args:
        smiles: Canonical SMILES string

    Returns:
        Dictionary with results from both methods
    """
    vqe_result = run_vqe_simulation(smiles, method="VQE")
    hf_result = run_classical_simulation(smiles)

    if vqe_result["error"] or hf_result["error"]:
        return {
            "error": vqe_result["error"] or hf_result["error"],
            "vqe": vqe_result,
            "hf": hf_result,
            "advantage": 0
        }

    # Calculate quantum advantage
    energy_diff = abs(hf_result["energy"] - vqe_result["energy"])

    return {
        "vqe": vqe_result,
        "hf": hf_result,
        "energy_difference": energy_diff,
        "quantum_advantage": energy_diff > 0.001,  # Threshold for meaningful difference
        "percent_improvement": (energy_diff / abs(hf_result["energy"])) * 100 if hf_result["energy"] != 0 else 0,
        "error": ""
    }


def get_supported_molecules() -> list:
    """Get list of all molecules supported by the simulation engine."""
    db = get_hamiltonian_db()
    return db.get_supported_molecules()