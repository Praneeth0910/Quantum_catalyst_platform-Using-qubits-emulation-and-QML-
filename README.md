# Quantum Catalyst Platform

Interactive Streamlit application for quantum-powered catalyst discovery, reaction analysis, and learning workflows.

## What This Project Does

This platform combines:

- Quantum chemistry simulation with VQE
- Quantum machine learning workflows (QSVM, VQC, QGAN-inspired candidate generation)
- Classical baselines for side-by-side comparison
- Molecule validation and 3D visualization
- Reaction pathway analysis and catalyst scoring

Primary goal: explore catalyst performance for reactions such as hydrogen oxidation, ammonia synthesis, and CO2 reduction.

## Main Features

- Feature 1: AI Discovery
	- Select a target reaction
	- Generate catalyst candidates
	- Rank candidates with quantum/classical signals
	- Run deeper VQE-based analysis

- Feature 2: Learning Game
	- Enter your catalyst guess
	- Receive QSVM-based score and feedback
	- Compare against reaction-specific ideal catalysts

- Quantum vs Classical Comparison
	- Compare VQE with HF/DFT-style baselines
	- Compare quantum ML outputs against classical ML methods
	- Inspect advantage metrics and charts

- Molecule Explorer
	- Accept molecule input as name, formula, or SMILES
	- Validate chemistry constraints
	- Visualize molecules in 3D
	- Run analysis pipelines from one place

- Results and Export
	- Keep session history
	- Export outcomes for reporting

## Tech Stack

- Python
- Streamlit
- RDKit
- Py3Dmol
- Qiskit / Qiskit Nature ecosystem
- Matplotlib
- Pandas / NumPy

## Project Structure

```text
Quantum_catalyst_platform/
|-- app.py
|-- run_app.bat
|-- requirements.txt
|-- modules/
|   |-- animation.py
|   |-- classical_baselines.py
|   |-- export_utils.py
|   |-- hamiltonian_database.py
|   |-- molecule_generation.py
|   |-- molecule_validator.py
|   |-- quantum_ml.py
|   |-- quantum_simulation.py
|   |-- reaction_pathway.py
|   |-- visualization.py
|-- test_complete_system.py
|-- test_quantum.py
|-- test_validator.py
```

## Setup

### 1. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If your environment is missing additional packages used by the app, install them as needed (for example pandas or numpy).

## Run the App

Option A: Streamlit directly

```bash
streamlit run app.py
```

Option B: Python module form

```bash
python -m streamlit run app.py
```

Option C: Windows batch launcher

```bat
run_app.bat
```

Note: the batch launcher expects a Conda environment named quantum.

## Run Tests

Quick checks:

```bash
python test_validator.py
python test_quantum.py
python test_complete_system.py
```

## Typical Workflow

1. Start the app.
2. Use AI Discovery to generate and rank catalyst candidates.
3. Use Learning Game to test catalyst intuition.
4. Compare quantum vs classical methods.
5. Export results for reporting.

## Troubleshooting

- Streamlit not found:
	- Install with pip install streamlit
- Import errors (RDKit/Qiskit/etc.):
	- Reinstall requirements and verify active environment
- Slow VQE runs:
	- Start with small molecules (for example H2)
- Molecule rejected:
	- Try formula/SMILES form and use smaller molecules

## Additional Documentation

- HOW_TO_RUN.md: detailed execution guide
- IMPLEMENTATION_SUMMARY.md: module-level implementation notes
- DAY1-2_COMPLETE.md: milestone summary
