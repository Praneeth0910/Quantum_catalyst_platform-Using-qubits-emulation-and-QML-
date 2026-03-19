"""
Test script for the new molecule validator
"""
from modules.molecule_validator import process_molecule_input, get_similar_molecules

print("=" * 60)
print("TESTING NEW MOLECULE VALIDATOR")
print("=" * 60)

# Test cases covering all input formats
test_cases = [
    # Common names
    ("water", True),
    ("hydrogen", True),
    ("methane", True),
    ("iron", True),
    ("platinum", True),

    # Molecular formulas
    ("H2O", True),
    ("CO2", True),
    ("H2", True),
    ("O2", True),
    ("NH3", True),

    # SMILES strings
    ("O", True),  # Water (3 atoms: O + 2H)
    ("C", True),  # Methane (5 atoms: C + 4H)
    ("[Fe]", True),  # Iron (1 atom)
    ("C=C", True),  # Ethylene (6 atoms: 2C + 4H)

    # Invalid inputs
    ("CC", False),  # Ethane (8 atoms: 2C + 6H - exceeds limit)
    ("CCCCCCCCCC", False),  # Too many atoms
    ("XYZ123", False),  # Nonsense
    ("benzene", False),  # Too many atoms (12 total)
]

print("\nRunning validation tests...\n")

for i, (input_str, should_pass) in enumerate(test_cases, 1):
    result = process_molecule_input(input_str, max_atoms=6)
    status = "[PASS]" if result["valid"] == should_pass else "[FAIL]"

    print(f"{status} Test {i}: '{input_str}'")
    print(f"   Expected: {'VALID' if should_pass else 'INVALID'}")
    print(f"   Got: {'VALID' if result['valid'] else 'INVALID'}")

    if result["valid"]:
        print(f"   Formula: {result['formula']}")
        print(f"   SMILES: {result['smiles']}")
        print(f"   Atoms: {result['atom_count']} (Heavy: {result['heavy_atom_count']})")
        print(f"   Weight: {result['mol_weight']} g/mol")
        print(f"   Elements: {', '.join(result['elements'])}")
    else:
        print(f"   Error: {result['error']}")
    print()

print("=" * 60)
print("Testing suggestion system...")
print("=" * 60)

# Test suggestions for failed inputs
test_queries = ["wat", "meth", "iro"]
for query in test_queries:
    suggestions = get_similar_molecules(query, limit=3)
    print(f"\nQuery: '{query}'")
    print(f"Suggestions: {', '.join(suggestions) if suggestions else 'None'}")

print("\n" + "=" * 60)
print("VALIDATOR TEST COMPLETE")
print("=" * 60)
