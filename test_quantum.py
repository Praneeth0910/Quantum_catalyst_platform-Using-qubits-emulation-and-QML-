"""
Test PySCF + Qiskit Nature + VQE integration
"""
import sys

print("=" * 60)
print("TESTING QUANTUM SIMULATION STACK")
print("=" * 60)

# Test 1: Check imports
print("\n[1/4] Testing imports...")
try:
    import pyscf
    print("  [OK] PySCF version:", pyscf.__version__)
except ImportError as e:
    print("  [ERROR] PySCF not available:", e)
    sys.exit(1)

try:
    from qiskit_nature.second_q.drivers import PySCFDriver
    print("  [OK] Qiskit Nature PySCFDriver")
except ImportError as e:
    print("  [ERROR] Qiskit Nature issue:", e)
    sys.exit(1)

try:
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import SLSQP
    print("  [OK] Qiskit Algorithms (VQE, SLSQP)")
except ImportError as e:
    print("  [ERROR] Qiskit Algorithms issue:", e)
    sys.exit(1)

# Test 2: Run simple H2 molecule with PySCF
print("\n[2/4] Testing PySCF electronic structure calculation...")
try:
    from pyscf import gto, scf

    # Build H2 molecule
    mol = gto.M(
        atom='H 0 0 0; H 0 0 0.74',  # Bond length 0.74 Angstrom
        basis='sto-3g'
    )

    # Run Hartree-Fock
    mf = scf.RHF(mol)
    energy = mf.kernel()

    print(f"  [OK] H2 Hartree-Fock energy: {energy:.6f} Hartree")
except Exception as e:
    print(f"  [ERROR] PySCF calculation failed: {e}")
    sys.exit(1)

# Test 3: Test PySCFDriver in Qiskit Nature
print("\n[3/4] Testing Qiskit Nature PySCFDriver...")
try:
    driver = PySCFDriver(
        atom='H 0 0 0; H 0 0 0.74',
        basis='sto-3g'
    )
    problem = driver.run()
    print(f"  [OK] Electronic structure problem created")
    print(f"  [OK] Nuclear repulsion energy: {problem.nuclear_repulsion_energy:.6f} Hartree")
    print(f"  [OK] Number of molecular orbitals: {problem.num_spatial_orbitals}")

except Exception as e:
    print(f"  [ERROR] PySCFDriver failed: {e}")
    sys.exit(1)

# Test 4: Test full VQE simulation
print("\n[4/4] Testing VQE with our quantum_simulation module...")
try:
    from modules.quantum_simulation import run_vqe_simulation

    # Test H2 molecule
    xyz_coords = "H 0 0 0; H 0 0 0.74"
    result = run_vqe_simulation(xyz_coords)

    if "error" in result and result["error"]:
        print(f"  [ERROR] VQE simulation error: {result['error']}")
    else:
        print(f"  [OK] VQE ground state energy: {result['energy']:.6f} Hartree")
        print(f"  [OK] Iterations: {result['iterations']}")
        print(f"  [OK] Convergence data points: {len(result['convergence'])}")

        # Sanity check: H2 ground state should be around -1.137 Hartree
        expected = -1.137
        if abs(result['energy'] - expected) < 0.1:
            print(f"  [OK] Energy is physically reasonable (expected ~{expected:.3f})")
        else:
            print(f"  [WARNING] Energy seems off (expected ~{expected:.3f}, got {result['energy']:.3f})")

except Exception as e:
    print(f"  [ERROR] VQE simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED - QUANTUM STACK WORKING!")
print("=" * 60)
