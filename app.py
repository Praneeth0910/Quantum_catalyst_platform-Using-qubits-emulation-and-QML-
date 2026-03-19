import streamlit as st
import matplotlib.pyplot as plt
from modules.molecule_validator import molecule_to_smiles, validate_molecule
from modules.molecule_generation import generate_3d_molecule
from modules.visualization import show_molecule_3d, mol_to_xyz
from modules.quantum_simulation import run_vqe_simulation
from modules.reaction_pathway import simulate_reaction_pathway, compute_catalyst_score
from modules.animation import show_reaction_animation

# Page Configuration
st.set_page_config(page_title="Quantum Catalyst Platform", layout="wide")
st.title("⚛️ Quantum Catalyst Discovery Platform")
st.markdown("---")

# Sidebar - User Inputs
st.sidebar.header("Configuration")
target_reaction = st.sidebar.selectbox(
    "Select Target Reaction:",
    ["H2 + O2 → H2O (Fuel Cell)", "CO2 Reduction", "N2 Fixation"]
)

molecule_input = st.sidebar.text_input(
    "Enter Catalyst Formula (e.g., Fe, NiO, Pt):",
    value="NiO"
)

analyze_btn = st.sidebar.button("Run Quantum Analysis")

# Main Dashboard Layout
if analyze_btn:
    smiles = molecule_to_smiles(molecule_input)
    
    if smiles and validate_molecule(smiles):
        col_left, col_right = st.columns([1, 1])
        
        # --- LEFT COLUMN: Structure & Quantum Engine ---
        with col_left:
            st.subheader("🔬 Molecular Structure & QML Engine")
            mol = generate_3d_molecule(smiles)
            if mol:
                show_molecule_3d(mol)
                xyz_data = mol_to_xyz(mol)

                with st.spinner("Solving Schrödinger Equation via VQE..."):
                    # REAL Quantum Simulation
                    q_result = run_vqe_simulation(xyz_data)

                if "error" in q_result:
                    st.error(f"Quantum Engine Error: {q_result['error']}")
                else:
                    st.metric("Ground State Energy", f"{q_result['energy']:.4f} Hartree")
                    
                    # QML Convergence Graph
                    fig_conv, ax_conv = plt.subplots()
                    ax_conv.plot(q_result["convergence"], marker='o', color='#007bff')
                    ax_conv.set_xlabel("VQE Iteration")
                    ax_conv.set_ylabel("Energy (Hartree)")
                    ax_conv.set_title("QML Convergence Profile")
                    st.pyplot(fig_conv)
            else:
                st.error("3D generation failed.")

        # --- RIGHT COLUMN: Pathway & Animation ---
        with col_right:
            st.subheader("🛤️ Reaction Pathway & Scoring")
            path_data = simulate_reaction_pathway(molecule_input)
            
            # Reaction Energy Profile
            fig_path, ax_path = plt.subplots(figsize=(8, 4))
            ax_path.plot(path_data["states"], path_data["energies"], marker='s', color='red')
            ax_path.set_title(f"Pathway: {target_reaction}")
            st.pyplot(fig_path)

            # Catalyst Score
            score = compute_catalyst_score(path_data["energies"])
            st.metric("Catalyst Suitability Score", f"{score}%")
            
            # Animation Placeholder
            st.subheader("🔄 Reaction Animation")
            show_reaction_animation(path_data["states"], path_data["energies"])

        st.markdown("---")
    else:
        st.error("❌ Invalid molecule or formula too complex (Max 6 atoms).")
else:
    st.info("👈 Enter a catalyst and click 'Run Quantum Analysis' to start.")