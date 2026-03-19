"""
Quantum Catalyst Discovery Platform - Main Application
======================================================

A comprehensive platform for quantum-powered catalyst discovery using:
- Real VQE simulations
- Quantum Machine Learning (QSVM, VQC, QGAN)
- Classical algorithm comparisons
- Interactive educational tools

Author: Built with Claude Code
Date: March 2026
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Import our modules
from modules.molecule_validator import process_molecule_input
from modules.molecule_generation import generate_3d_molecule
from modules.visualization import show_molecule_3d, mol_to_xyz
from modules.quantum_simulation import run_vqe_simulation, compare_methods, get_supported_molecules
from modules.reaction_pathway import (
    simulate_reaction_pathway,
    compute_catalyst_score,
    get_supported_reactions,
    get_reaction_info,
    REACTION_DATABASE
)
from modules.quantum_ml import (
    discover_catalysts,
    score_user_catalyst,
    QuantumCatalystScorer
)
from modules.classical_baselines import (
    compare_quantum_vs_classical_chemistry,
    compare_quantum_vs_classical_ml,
    run_full_comparison
)

# ========================================================================
# PAGE CONFIGURATION
# ========================================================================

st.set_page_config(
    page_title="Quantum Catalyst Platform",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visuals
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ========================================================================
# SESSION STATE INITIALIZATION
# ========================================================================

if 'results_history' not in st.session_state:
    st.session_state.results_history = []

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# ========================================================================
# SIDEBAR NAVIGATION
# ========================================================================

st.sidebar.title("⚛️ Quantum Catalyst Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔬 Feature 1: AI Discovery",
        "🎮 Feature 2: Learning Game",
        "📊 Quantum vs Classical",
        "🧪 Molecule Explorer",
        "📈 Results & Export"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About this Platform:**

This platform demonstrates quantum computing's advantage in catalyst discovery using:
- Real VQE simulations
- Quantum Machine Learning
- Chemistry-based models

Built with Qiskit, RDKit, and Streamlit.
""")

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def plot_energy_landscape(states, energies, title="Energy Landscape"):
    """Create beautiful energy landscape plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(states))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    # Plot lines
    ax.plot(x_pos, energies, 'o-', linewidth=2, markersize=10, color='#2c3e50')

    # Fill area
    ax.fill_between(x_pos, energies, min(energies), alpha=0.3, color='#3498db')

    # Annotate energies
    for i, (state, energy) in enumerate(zip(states, energies)):
        ax.annotate(f'{energy:.4f} Ha',
                   xy=(i, energy),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels(states, rotation=45, ha='right')
    ax.set_ylabel('Energy (Hartree)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    return fig

def plot_convergence(convergence_data, title="VQE Convergence"):
    """Plot VQE convergence."""
    fig, ax = plt.subplots(figsize=(10, 5))

    iterations = range(1, len(convergence_data) + 1)
    ax.plot(iterations, convergence_data, 'o-', linewidth=2, markersize=6, color='#e74c3c')

    # Mark minimum
    min_idx = np.argmin(convergence_data)
    min_energy = convergence_data[min_idx]
    ax.plot(min_idx + 1, min_energy, 'g*', markersize=20, label=f'Minimum: {min_energy:.6f} Ha')

    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy (Hartree)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig

def plot_comparison_bar(data_dict, title="Comparison", ylabel="Score"):
    """Create comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(data_dict.keys())
    values = list(data_dict.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(methods)]

    bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig

def save_result_to_history(result_data):
    """Save analysis result to session history."""
    result_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.results_history.append(result_data)

# ========================================================================
# PAGE: HOME
# ========================================================================

if page == "🏠 Home":
    st.markdown('<p class="main-header">⚛️ Quantum Catalyst Discovery Platform</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Future of Catalyst Discovery!

    This platform harnesses **quantum computing** and **machine learning** to revolutionize
    how we discover and optimize catalysts for chemical reactions.
    """)

    # Key Features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🔬 Feature 1: AI-Powered Discovery
        - **QGAN**: Generate novel catalyst candidates
        - **VQC**: Classify catalyst effectiveness
        - **VQE**: Validate with quantum simulations

        Perfect for researchers discovering new materials!
        """)

        if st.button("🚀 Try AI Discovery", key="home_discovery"):
            st.session_state.current_page = "🔬 Feature 1: AI Discovery"
            st.rerun()

    with col2:
        st.markdown("""
        #### 🎮 Feature 2: Interactive Learning
        - **QSVM**: Score your catalyst choices
        - **Real-time feedback**: Learn chemistry principles
        - **Gamified experience**: Challenge yourself!

        Perfect for students and educators!
        """)

        if st.button("🎯 Try Learning Game", key="home_learning"):
            st.session_state.current_page = "🎮 Feature 2: Learning Game"
            st.rerun()

    st.markdown("---")

    # Platform Capabilities
    st.markdown("### 🎯 Platform Capabilities")

    cap_col1, cap_col2, cap_col3 = st.columns(3)

    with cap_col1:
        st.metric("Molecules Supported", f"{len(get_supported_molecules())}")
        st.metric("Reactions Available", f"{len(get_supported_reactions())}")

    with cap_col2:
        st.metric("Quantum Algorithms", "4+")
        st.caption("VQE, QSVM, VQC, QGAN")

    with cap_col3:
        st.metric("Classical Baselines", "6+")
        st.caption("HF, DFT, RF, SVM, GB")

    st.markdown("---")

    # Quick Start Guide
    with st.expander("📖 Quick Start Guide"):
        st.markdown("""
        ### How to Use This Platform

        1. **AI Discovery Mode**:
           - Select a target reaction
           - Let the QGAN generate catalyst candidates
           - Review quantum simulation results
           - Compare with classical methods

        2. **Learning Game Mode**:
           - Choose a reaction
           - Guess the best catalyst
           - Get scored by quantum ML
           - Learn from detailed feedback

        3. **Comparison Mode**:
           - Explore quantum vs classical algorithms
           - See side-by-side energy calculations
           - Understand quantum advantage

        4. **Molecule Explorer**:
           - Validate any molecule
           - Run VQE simulations
           - View 3D structures
           - Analyze properties
        """)

    # Technical Details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### Technology Stack

        **Quantum Computing:**
        - **Qiskit**: IBM's quantum framework
        - **VQE**: Variational Quantum Eigensolver
        - **QSVM**: Quantum Support Vector Machine
        - **Real quantum circuits** (not simulated data!)

        **Chemistry:**
        - **RDKit**: Molecular validation and properties
        - **Custom Hamiltonians**: Pre-computed for 26+ molecules
        - **D-band model**: Catalyst activity prediction
        - **BEP relation**: Activation energy estimation

        **Machine Learning:**
        - **Qiskit ML**: Quantum machine learning algorithms
        - **Scikit-learn**: Classical ML baselines
        - **Feature engineering**: 8D molecular descriptors

        **Visualization:**
        - **Streamlit**: Interactive web interface
        - **Matplotlib**: Scientific plotting
        - **Py3Dmol**: 3D molecular visualization
        """)

# ========================================================================
# PAGE: FEATURE 1 - AI DISCOVERY
# ========================================================================

elif page == "🔬 Feature 1: AI Discovery":
    st.markdown('<p class="main-header">🔬 AI-Powered Catalyst Discovery</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Discover Novel Catalysts with Quantum AI

    This feature uses **QGAN** to generate catalyst candidates, **VQC** to classify them,
    and **VQE** to validate their effectiveness through quantum simulations.
    """)

    # User inputs
    st.markdown("### ⚙️ Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        reaction_options = {
            "H2_O2": "H₂ + O₂ → H₂O (Fuel Cell / Water Formation)",
            "N2_H2": "N₂ + 3H₂ → 2NH₃ (Haber Process / Ammonia)",
            "CO2_reduction": "CO₂ + H₂ → CO + H₂O (Carbon Capture)"
        }

        selected_reaction = st.selectbox(
            "Select Target Reaction:",
            options=list(reaction_options.keys()),
            format_func=lambda x: reaction_options[x]
        )

        reaction_info = get_reaction_info(selected_reaction)
        st.info(f"**Reaction:** {reaction_info['equation']}")

    with col2:
        num_candidates = st.slider("Number of Candidates:", 3, 10, 5)

    discover_btn = st.button("🚀 Discover Catalysts", type="primary", use_container_width=True)

    if discover_btn:
        with st.spinner("🧬 Quantum AI is generating candidates..."):
            # Discover catalysts
            candidates = discover_catalysts(selected_reaction, num_candidates)

            if candidates:
                st.success(f"✅ Generated {len(candidates)} catalyst candidates!")

                # Display candidates
                st.markdown("### 📋 Catalyst Candidates")

                # Create DataFrame
                df = pd.DataFrame([
                    {
                        "Rank": i+1,
                        "Catalyst": cand['smiles'],
                        "Metal": cand.get('metal_type', 'N/A'),
                        "Score": f"{cand['catalyst_score']:.2f}",
                        "Classification": cand['classification'],
                        "Generation Score": f"{cand['generation_score']:.2f}"
                    }
                    for i, cand in enumerate(candidates)
                ])

                st.dataframe(df, use_container_width=True)

                # Visualize top 3
                st.markdown("### 🏆 Top 3 Detailed Analysis")

                for i, cand in enumerate(candidates[:3]):
                    with st.expander(f"#{i+1}: {cand['smiles']} - Score: {cand['catalyst_score']:.2f}/100"):
                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.markdown("**Catalyst Information:**")
                            st.write(f"- **SMILES:** `{cand['smiles']}`")
                            st.write(f"- **Metal:** {cand.get('metal_type', 'N/A')}")
                            st.write(f"- **Classification:** {cand['classification']}")
                            st.write(f"- **Feedback:** {cand['feedback']}")

                        with col_b:
                            st.markdown("**Run Full Simulation:**")
                            if st.button(f"Run VQE + Pathway", key=f"sim_{i}"):
                                with st.spinner("Running quantum simulation..."):
                                    # Run VQE
                                    vqe_result = run_vqe_simulation(cand['smiles'])

                                    if not vqe_result.get('error'):
                                        st.metric("Ground State Energy", f"{vqe_result['energy']:.6f} Ha")

                                        # Plot convergence
                                        fig_conv = plot_convergence(
                                            vqe_result['convergence'],
                                            f"VQE Convergence: {cand['smiles']}"
                                        )
                                        st.pyplot(fig_conv)
                                        plt.close()

                                        # Run pathway
                                        pathway = simulate_reaction_pathway(cand['smiles'], selected_reaction)

                                        if not pathway.get('error'):
                                            fig_path = plot_energy_landscape(
                                                pathway['states'],
                                                pathway['energies'],
                                                f"Reaction Pathway: {cand['smiles']}"
                                            )
                                            st.pyplot(fig_path)
                                            plt.close()
                                    else:
                                        st.error(f"Error: {vqe_result['error']}")

                # Comparison chart
                st.markdown("### 📊 Candidate Comparison")
                score_dict = {f"#{i+1} {c['smiles']}": c['catalyst_score'] for i, c in enumerate(candidates[:5])}
                fig_comp = plot_comparison_bar(score_dict, "Catalyst Scores", "Score (0-100)")
                st.pyplot(fig_comp)
                plt.close()

                # Save to history
                save_result_to_history({
                    'type': 'AI Discovery',
                    'reaction': selected_reaction,
                    'candidates': candidates
                })

            else:
                st.error("❌ Failed to generate candidates. Please try again.")

# ========================================================================
# PAGE: FEATURE 2 - LEARNING GAME
# ========================================================================

elif page == "🎮 Feature 2: Learning Game":
    st.markdown('<p class="main-header">🎮 Interactive Catalyst Learning Game</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Test Your Chemistry Knowledge!

    Choose a catalyst for the reaction and see how it compares to the ideal choice.
    Get scored by **quantum machine learning** algorithms!
    """)

    # Game configuration
    st.markdown("### 🎯 Challenge Configuration")

    col1, col2 = st.columns(2)

    with col1:
        game_reaction = st.selectbox(
            "Select Reaction:",
            options=list(REACTION_DATABASE.keys()),
            format_func=lambda x: REACTION_DATABASE[x]['name']
        )

        reaction_data = REACTION_DATABASE[game_reaction]
        st.info(f"**Equation:** {reaction_data['equation']}")

        # Show ideal catalysts hint
        if st.checkbox("Show Hint"):
            st.warning(f"💡 Ideal catalysts: {', '.join(reaction_data['ideal_catalysts'])}")

    with col2:
        user_catalyst = st.text_input(
            "Your Catalyst Guess:",
            placeholder="e.g., Pt, Fe, NiO, iron, platinum"
        )

        st.caption("Enter: element name, formula, or SMILES")

    submit_guess = st.button("🎯 Submit & Get Scored", type="primary", use_container_width=True)

    if submit_guess and user_catalyst:
        # Validate input
        validation = process_molecule_input(user_catalyst, max_atoms=6)

        if validation['valid']:
            user_smiles = validation['smiles']

            st.success(f"✅ Valid catalyst: **{validation['formula']}** ({validation['atom_count']} atoms)")

            with st.spinner("🧠 Quantum ML is evaluating your choice..."):
                # Get ideal catalyst for comparison
                ideal_catalyst = reaction_data['ideal_catalysts'][0]

                # Score user's choice
                scoring_result = score_user_catalyst(user_smiles, ideal_catalyst, game_reaction)

                if not scoring_result.get('error'):
                    st.markdown("---")
                    st.markdown("### 📊 Your Results")

                    # Overall score with colored background
                    score = scoring_result['overall_score']
                    if score >= 80:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"## 🏆 Excellent! Score: {score:.1f}/100")
                        st.markdown("</div>", unsafe_allow_html=True)
                    elif score >= 60:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown(f"## 👍 Good! Score: {score:.1f}/100")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.markdown(f"## 💡 Keep Learning! Score: {score:.1f}/100")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Detailed breakdown
                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        st.metric("QSVM Score", f"{scoring_result['qsvm_score']:.2f}/100")

                    with col_b:
                        st.metric("Feature Similarity", f"{scoring_result['feature_similarity']:.2f}%")

                    with col_c:
                        st.metric("Classification", scoring_result['classification'].upper())

                    # Feedback
                    st.markdown("### 💬 Feedback")
                    st.info(scoring_result['qsvm_feedback'])

                    # VQC Category
                    st.markdown(f"**Catalyst Category:** {scoring_result['vqc_category']}")

                    # Run full analysis
                    st.markdown("---")
                    st.markdown("### 🔬 Deep Dive Analysis")

                    if st.button("Run Full Quantum Simulation"):
                        with st.spinner("Running comprehensive analysis..."):
                            # VQE simulation
                            vqe_result = run_vqe_simulation(user_smiles)

                            col_vqe1, col_vqe2 = st.columns(2)

                            with col_vqe1:
                                if not vqe_result.get('error'):
                                    st.metric("Ground State Energy", f"{vqe_result['energy']:.6f} Ha")
                                    st.metric("VQE Iterations", vqe_result['iterations'])
                                    st.metric("Qubits Used", vqe_result['num_qubits'])
                                else:
                                    st.error(f"VQE Error: {vqe_result['error']}")

                            with col_vqe2:
                                if not vqe_result.get('error'):
                                    fig_conv = plot_convergence(vqe_result['convergence'])
                                    st.pyplot(fig_conv)
                                    plt.close()

                            # Reaction pathway
                            pathway = simulate_reaction_pathway(user_smiles, game_reaction)

                            if not pathway.get('error'):
                                st.markdown("### 🛤️ Reaction Energy Pathway")

                                col_path1, col_path2 = st.columns([2, 1])

                                with col_path1:
                                    fig_path = plot_energy_landscape(
                                        pathway['states'],
                                        pathway['energies'],
                                        "Reaction Pathway"
                                    )
                                    st.pyplot(fig_path)
                                    plt.close()

                                with col_path2:
                                    st.metric("Activation Barrier",
                                            f"{pathway['activation_barrier_forward']*27.211:.2f} eV")
                                    st.metric("Reaction Enthalpy",
                                            f"{pathway['reaction_enthalpy']*27.211:.2f} eV")
                                    st.metric("Pathway Score", f"{pathway['catalyst_score']:.2f}/100")

                                    if pathway['is_ideal_catalyst']:
                                        st.success("✨ This IS an ideal catalyst!")
                                    else:
                                        st.info("Try one of the ideal catalysts for comparison!")

                    # Save to history
                    save_result_to_history({
                        'type': 'Learning Game',
                        'reaction': game_reaction,
                        'user_catalyst': user_catalyst,
                        'score': score
                    })

                else:
                    st.error(f"❌ Scoring error: {scoring_result.get('error')}")

        else:
            st.error(f"❌ Invalid catalyst: {validation['error']}")

            # Show suggestions
            from modules.molecule_validator import get_similar_molecules
            suggestions = get_similar_molecules(user_catalyst, limit=5)
            if suggestions:
                st.info(f"💡 Did you mean: {', '.join(suggestions)}?")

# ========================================================================
# PAGE: QUANTUM VS CLASSICAL
# ========================================================================

elif page == "📊 Quantum vs Classical":
    st.markdown('<p class="main-header">📊 Quantum vs Classical Comparison</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Demonstrate Quantum Advantage

    Compare quantum algorithms (VQE, QSVM) against classical methods (HF, DFT, ML)
    to see the power of quantum computing in action!
    """)

    # Comparison type
    comparison_type = st.radio(
        "Comparison Type:",
        ["Chemistry Methods (VQE vs HF/DFT)", "Machine Learning (QSVM vs Classical ML)", "Full Comparison"]
    )

    # Molecule/Catalyst input
    test_molecule = st.text_input(
        "Enter Molecule/Catalyst:",
        value="[Pt]",
        placeholder="e.g., H2, H2O, Pt, Fe"
    )

    if comparison_type in ["Machine Learning (QSVM vs Classical ML)", "Full Comparison"]:
        ml_reaction = st.selectbox(
            "Reaction for ML Comparison:",
            options=list(REACTION_DATABASE.keys()),
            format_func=lambda x: REACTION_DATABASE[x]['name']
        )

    run_comparison = st.button("⚡ Run Comparison", type="primary", use_container_width=True)

    if run_comparison and test_molecule:
        # Validate molecule
        validation = process_molecule_input(test_molecule, max_atoms=6)

        if validation['valid']:
            smiles = validation['smiles']
            st.success(f"✅ Analyzing: **{validation['formula']}**")

            # Chemistry Comparison
            if comparison_type in ["Chemistry Methods (VQE vs HF/DFT)", "Full Comparison"]:
                st.markdown("---")
                st.markdown("### ⚛️ Quantum Chemistry Comparison")

                with st.spinner("Running quantum and classical simulations..."):
                    chem_comp = compare_quantum_vs_classical_chemistry(smiles)

                    if not chem_comp.get('error'):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("#### VQE (Quantum)")
                            st.metric("Energy", f"{chem_comp['vqe']['energy']:.6f} Ha")
                            st.metric("Iterations", chem_comp['vqe']['iterations'])
                            st.caption(chem_comp['vqe']['method'])

                        with col2:
                            st.markdown("#### Hartree-Fock")
                            st.metric("Energy", f"{chem_comp['hf']['energy']:.6f} Ha")
                            st.caption(chem_comp['hf']['description'])

                        with col3:
                            st.markdown("#### DFT (B3LYP)")
                            st.metric("Energy", f"{chem_comp['dft']['energy']:.6f} Ha")
                            st.caption(chem_comp['dft']['description'])

                        # Comparison visualization
                        st.markdown("#### Energy Comparison")
                        energy_dict = {
                            "VQE\n(Quantum)": chem_comp['vqe']['energy'],
                            "HF\n(Classical)": chem_comp['hf']['energy'],
                            "DFT\n(Classical)": chem_comp['dft']['energy']
                        }
                        fig_energy = plot_comparison_bar(energy_dict, "Energy Comparison", "Energy (Hartree)")
                        st.pyplot(fig_energy)
                        plt.close()

                        # Quantum advantage analysis
                        st.markdown("#### 🎯 Quantum Advantage Analysis")
                        comparison = chem_comp['comparison']

                        adv_col1, adv_col2 = st.columns(2)

                        with adv_col1:
                            st.markdown("**VQE vs HF:**")
                            st.metric("Energy Difference",
                                    f"{comparison['vs_hf']['energy_difference']:.6f} Ha")
                            st.metric("Improvement",
                                    f"{comparison['vs_hf']['percent_improvement']:.4f}%")
                            if comparison['vs_hf']['quantum_is_better']:
                                st.success("✅ VQE is more accurate!")
                            else:
                                st.info("HF approximation sufficient")

                        with adv_col2:
                            st.markdown("**VQE vs DFT:**")
                            st.metric("Energy Difference",
                                    f"{comparison['vs_dft']['energy_difference']:.6f} Ha")
                            st.metric("Improvement",
                                    f"{comparison['vs_dft']['percent_improvement']:.4f}%")
                            if comparison['vs_dft']['quantum_is_better']:
                                st.success("✅ VQE is more accurate!")
                            else:
                                st.info("DFT approximation sufficient")

                        # Summary
                        summary = chem_comp['summary']
                        if summary['quantum_advantage_demonstrated']:
                            st.success(f"🏆 **Quantum Advantage Demonstrated!** Most accurate method: {summary['most_accurate']}")
                        else:
                            st.info(f"Most accurate: {summary['most_accurate']}")

                        # VQE Convergence
                        if st.checkbox("Show VQE Convergence Details"):
                            fig_conv = plot_convergence(chem_comp['vqe']['convergence'])
                            st.pyplot(fig_conv)
                            plt.close()

                    else:
                        st.error(f"Error: {chem_comp.get('error')}")

            # ML Comparison
            if comparison_type in ["Machine Learning (QSVM vs Classical ML)", "Full Comparison"]:
                st.markdown("---")
                st.markdown("### 🤖 Machine Learning Comparison")

                with st.spinner("Running quantum and classical ML..."):
                    ml_comp = compare_quantum_vs_classical_ml(smiles, ml_reaction)

                    if not ml_comp.get('error'):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### Quantum ML (QSVM)")
                            qsvm = ml_comp['quantum_ml']
                            st.metric("Score", f"{qsvm['score']:.2f}/100")
                            st.metric("Classification", qsvm['classification'].upper())
                            st.metric("Confidence", f"{qsvm['confidence']:.1f}%")
                            st.caption(qsvm['method'])

                        with col2:
                            st.markdown("#### Classical ML (Average)")
                            avg_score = ml_comp['comparison']['avg_classical_score']
                            st.metric("Average Score", f"{avg_score:.2f}/100")

                            st.markdown("**Individual Methods:**")
                            classical = ml_comp['classical_ml']
                            st.write(f"- RF: {classical['random_forest']['score']:.2f}")
                            st.write(f"- SVM: {classical['svm']['score']:.2f}")
                            st.write(f"- GB: {classical['gradient_boosting']['score']:.2f}")

                        # ML Comparison visualization
                        st.markdown("#### Scoring Comparison")
                        ml_dict = {
                            "QSVM\n(Quantum)": qsvm['score'],
                            "Random\nForest": classical['random_forest']['score'],
                            "SVM\n(Classical)": classical['svm']['score'],
                            "Gradient\nBoosting": classical['gradient_boosting']['score']
                        }
                        fig_ml = plot_comparison_bar(ml_dict, "ML Method Comparison", "Score (0-100)")
                        st.pyplot(fig_ml)
                        plt.close()

                        # Quantum ML advantage
                        advantage = ml_comp['comparison']['quantum_advantage']
                        if advantage > 5:
                            st.success(f"🎯 **Quantum ML Advantage: {advantage:.2f} points!**")
                        else:
                            st.info(f"Methods are comparable (difference: {advantage:.2f} points)")

                    else:
                        st.error(f"Error: {ml_comp.get('error')}")

            # Save comparison
            save_result_to_history({
                'type': 'Comparison',
                'molecule': test_molecule,
                'comparison_type': comparison_type
            })

        else:
            st.error(f"❌ Invalid molecule: {validation['error']}")

# ========================================================================
# PAGE: MOLECULE EXPLORER
# ========================================================================

elif page == "🧪 Molecule Explorer":
    st.markdown('<p class="main-header">🧪 Molecule Explorer</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Explore Molecular Properties & Quantum Simulations

    Validate any molecule, visualize its structure, run VQE simulations,
    and analyze its properties.
    """)

    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        explore_input = st.text_input(
            "Enter Molecule:",
            placeholder="e.g., water, H2O, methane, Pt, benzene",
            help="Accepts: common names, formulas, or SMILES"
        )

    with col2:
        max_atoms = st.number_input("Max Atoms:", 1, 20, 6)

    explore_btn = st.button("🔍 Analyze Molecule", type="primary", use_container_width=True)

    # Show supported molecules
    with st.expander("📚 View All Supported Molecules"):
        supported = get_supported_molecules()
        st.write(f"Total: {len(supported)} molecules")

        # Group by type
        diatomic = [s for s in supported if len(s) <= 7 and '[' in s and 'H' in s]
        metals = [s for s in supported if '[' in s and len(s) <= 5]
        organic = [s for s in supported if 'C' in s or 'O' in s and len(s) > 1]

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Diatomic:**")
            st.write(", ".join(diatomic[:10]))
        with col_b:
            st.markdown("**Metals:**")
            st.write(", ".join(metals[:10]))
        with col_c:
            st.markdown("**Organic:**")
            st.write(", ".join(organic[:10]))

    if explore_btn and explore_input:
        # Validate
        validation = process_molecule_input(explore_input, max_atoms=max_atoms)

        if validation['valid']:
            smiles = validation['smiles']

            st.success("✅ Valid Molecule!")

            # Display properties
            st.markdown("### 📋 Molecular Properties")

            prop_col1, prop_col2, prop_col3, prop_col4 = st.columns(4)

            with prop_col1:
                st.metric("Formula", validation['formula'])
            with prop_col2:
                st.metric("Atoms", validation['atom_count'])
            with prop_col3:
                st.metric("Molecular Weight", f"{validation['mol_weight']:.2f}")
            with prop_col4:
                st.metric("Heavy Atoms", validation['heavy_atom_count'])

            st.write(f"**Elements:** {', '.join(validation['elements'])}")
            st.write(f"**SMILES:** `{smiles}`")

            # 3D Visualization
            st.markdown("---")
            st.markdown("### 🔬 3D Structure")

            try:
                mol_3d = generate_3d_molecule(smiles)
                if mol_3d:
                    show_molecule_3d(mol_3d)
                else:
                    st.warning("3D generation not available for this molecule")
            except:
                st.warning("3D visualization not available for this molecule")

            # Quantum Simulation
            st.markdown("---")
            st.markdown("### ⚛️ Quantum Simulation")

            if st.button("Run VQE Simulation"):
                with st.spinner("Running VQE..."):
                    vqe_result = run_vqe_simulation(smiles, method="VQE")

                    if not vqe_result.get('error'):
                        sim_col1, sim_col2 = st.columns([1, 2])

                        with sim_col1:
                            st.metric("Ground State Energy", f"{vqe_result['energy']:.6f} Ha")
                            st.metric("Iterations", vqe_result['iterations'])
                            st.metric("Qubits", vqe_result['num_qubits'])
                            st.caption(vqe_result['method'])

                            # Compare with HF
                            if st.checkbox("Compare with HF"):
                                comp = compare_methods(smiles)
                                if not comp.get('error'):
                                    st.metric("HF Energy", f"{comp['hf']['energy']:.6f} Ha")
                                    st.metric("Difference", f"{comp['energy_difference']:.6f} Ha")

                        with sim_col2:
                            fig_conv = plot_convergence(
                                vqe_result['convergence'],
                                f"VQE Convergence: {validation['formula']}"
                            )
                            st.pyplot(fig_conv)
                            plt.close()
                    else:
                        st.error(f"VQE Error: {vqe_result['error']}")

            # Test in reactions
            st.markdown("---")
            st.markdown("### 🧬 Test as Catalyst")

            test_reaction = st.selectbox(
                "Test in Reaction:",
                options=list(REACTION_DATABASE.keys()),
                format_func=lambda x: REACTION_DATABASE[x]['name']
            )

            if st.button("Run Catalyst Test"):
                with st.spinner("Testing catalyst..."):
                    pathway = simulate_reaction_pathway(smiles, test_reaction)

                    if not pathway.get('error'):
                        cat_col1, cat_col2 = st.columns([2, 1])

                        with cat_col1:
                            fig_path = plot_energy_landscape(
                                pathway['states'],
                                pathway['energies'],
                                f"Pathway: {validation['formula']}"
                            )
                            st.pyplot(fig_path)
                            plt.close()

                        with cat_col2:
                            st.metric("Catalyst Score", f"{pathway['catalyst_score']:.2f}/100")
                            st.metric("Activation Barrier",
                                    f"{pathway['activation_barrier_forward']*27.211:.2f} eV")
                            st.metric("Reaction Enthalpy",
                                    f"{pathway['reaction_enthalpy']*27.211:.2f} eV")

                            if pathway['is_ideal_catalyst']:
                                st.success("⭐ Ideal Catalyst!")

                            st.write(f"**Type:** {pathway['reaction_type']}")
                            st.write(f"**Method:** {pathway['method']}")
                    else:
                        st.error(f"Error: {pathway['error']}")

        else:
            st.error(f"❌ {validation['error']}")

            # Suggestions
            from modules.molecule_validator import get_similar_molecules
            suggestions = get_similar_molecules(explore_input)
            if suggestions:
                st.info(f"💡 Did you mean: {', '.join(suggestions[:5])}?")

# ========================================================================
# PAGE: RESULTS & EXPORT
# ========================================================================

elif page == "📈 Results & Export":
    st.markdown('<p class="main-header">📈 Results & Export</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Session History & Data Export

    Review your session history and export results for further analysis.
    """)

    if st.session_state.results_history:
        st.success(f"📊 {len(st.session_state.results_history)} results in session")

        # Display history
        st.markdown("### 📋 Session History")

        for i, result in enumerate(reversed(st.session_state.results_history)):
            with st.expander(f"#{len(st.session_state.results_history)-i}: {result['type']} - {result['timestamp']}"):
                st.json(result, expanded=False)

        # Export options
        st.markdown("---")
        st.markdown("### 💾 Export Options")

        export_col1, export_col2, export_col3 = st.columns(3)

        with export_col1:
            if st.button("📄 Export JSON"):
                json_data = json.dumps(st.session_state.results_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"quantum_catalyst_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with export_col2:
            if st.button("📊 Export CSV"):
                # Flatten data for CSV
                csv_data = []
                for result in st.session_state.results_history:
                    csv_data.append({
                        'Timestamp': result['timestamp'],
                        'Type': result['type'],
                        'Details': str(result)
                    })
                df = pd.DataFrame(csv_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"quantum_catalyst_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        with export_col3:
            if st.button("🗑️ Clear History"):
                st.session_state.results_history = []
                st.rerun()

    else:
        st.info("No results yet. Run some analyses first!")

# ========================================================================
# FOOTER
# ========================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Quantum Catalyst Discovery Platform</strong> | Built with Qiskit, RDKit, and Streamlit</p>
    <p>Powered by Real Quantum Computing & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
