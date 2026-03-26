"""
Quantum Catalyst Discovery Platform - Main Application
======================================================

A comprehensive platform for quantum-powered catalyst discovery using:
- Real VQE simulations
- Quantum Machine Learning (QSVM, Rule-Based Chemical Classifier, Stochastic Quantum Latent-Space Generator)
- Classical algorithm comparisons
- Interactive educational tools

Author: Baratam Pranneth Gupta
Date: March 2026
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from datetime import datetime
from PIL import Image
import os
#caching wrappers
@st.cache_data(ttl=3600)
def cached_vqe_simulation(smiles, apply_noise=False):
    return run_vqe_simulation(smiles, apply_noise=apply_noise)

@st.cache_data(ttl=3600)
def cached_pathway_simulation(smiles, reaction_input):
    return simulate_reaction_pathway(smiles, reaction_input)

@st.cache_data(ttl=3600)
def cached_discover_catalysts(reaction_key, num_cands):
    return discover_catalysts(reaction_key, num_cands)

# Import our modules
from modules.molecule_validator import process_molecule_input
from modules.molecule_generation import generate_3d_molecule
from modules.visualization import show_molecule_3d, mol_to_xyz
from modules.quantum_simulation import run_vqe_simulation, compare_methods, get_supported_molecules, HAS_PYSCF
from modules.reaction_pathway import (
    simulate_reaction_pathway,
    compute_catalyst_score,
    get_supported_reactions,
    get_reaction_info,
    REACTION_DATABASE,
    parse_dynamic_reaction,
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
from modules.export_utils import export_discovery_batch_to_csv

# ========================================================================
# PAGE CONFIGURATION
# ========================================================================

st.set_page_config(
    page_title="Quantum Catalyst Platform",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme with glassmorphism
st.markdown("""
<style>
    /* Root theme configuration */
    :root {
        --primary-bg: #0a0e27;
        --secondary-bg: #0f1438;
        --accent-cyan: #00d4ff;
        --accent-purple: #a855f7;
        --text-primary: #e8e8f0;
        --text-secondary: #a8aac0;
        --glass-bg: rgba(139, 89, 198, 0.08);
        --border-color: rgba(255, 255, 255, 0.1);
        --glow-color: rgba(0, 212, 255, 0.2);
    }

    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a0a3d 100%);
        color: var(--text-primary);
    }

    /* Metric cards - glassmorphic style */
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(8px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
        border-left: 4px solid var(--accent-cyan);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-left: 4px solid var(--accent-purple);
        box-shadow: 0 12px 48px rgba(168, 85, 247, 0.15);
    }

    /* Success box */
    .success-box {
        background: rgba(34, 197, 94, 0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #22c55e;
        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.1);
        color: #86efac;
    }

    /* Warning box */
    .warning-box {
        background: rgba(245, 158, 11, 0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #f59e0b;
        box-shadow: 0 8px 32px rgba(245, 158, 11, 0.1);
        color: #fcd34d;
    }

    /* Error box */
    .error-box {
        background: rgba(239, 68, 68, 0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #ef4444;
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.1);
        color: #fca5a5;
    }

    /* Main header */
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
        letter-spacing: 0.5px;
    }

    /* Sub header */
    .sub-header {
        font-size: 1.75rem;
        color: var(--accent-cyan);
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 700;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
        padding-bottom: 0.5rem;
    }

    /* Lab branding in sidebar */
    .lab-branding {
        text-align: center;
        padding: 1.5rem 1rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border-color);
    }

    .lab-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--accent-cyan);
        margin-top: 0.75rem;
        letter-spacing: 0.5px;
    }

    /* Footer styling */
    .footer-container {
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid var(--border-color);
        text-align: center;
    }

    .footer-text {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .footer-name {
        color: var(--accent-cyan);
        font-weight: 600;
        margin: 0.5rem 0;
    }

    .footer-subtitle {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }

    /* Enhanced spacing for sections */
    .section-divider {
        margin: 2.5rem 0;
        border: none;
        border-top: 1px solid var(--border-color);
    }

    /* Improve column spacing */
    .space-large {
        margin: 2rem 0;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }

        .sub-header {
            font-size: 1.35rem;
        }
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

if 'discovered_catalysts' not in st.session_state:
    st.session_state.discovered_catalysts = []

# ========================================================================
# SIDEBAR NAVIGATION
# ========================================================================

st.sidebar.title("⚛️ Quantum Catalyst Platform")

# Display logo and lab branding
try:
    logo_path = "logo-singularity.png"
    if os.path.exists(logo_path):
        st.sidebar.markdown('<div class="lab-branding">', unsafe_allow_html=True)
        st.sidebar.image(logo_path, width=150, use_container_width=False)
        st.sidebar.markdown('<div class="lab-name">The Singularity Advanced Research Lab</div>', unsafe_allow_html=True)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
except Exception as e:
    st.sidebar.warning(f"Logo not found: {e}")

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

if HAS_PYSCF:
    st.sidebar.success("🟢 Dynamic PySCF Engine: Active")
else:
    st.sidebar.warning("🟡 Dynamic Engine Offline (Using Static Fallback)")

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def plot_energy_landscape(states, classical_energies, quantum_energies=None, title="Reaction Landscape"):
    """Create comparative classical vs quantum reaction landscape plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(states))

    # Plot Classical Chemical Pathway
    ax.plot(
        x_pos,
        classical_energies,
        'o-',
        linewidth=2,
        markersize=10,
        color='#e74c3c',
        label='Classical Chemical Profile'
    )

    # Plot Quantum Reaction Pathway (if provided)
    if quantum_energies is not None and len(quantum_energies) == len(classical_energies):
        ax.plot(
            x_pos,
            quantum_energies,
            '*--',
            linewidth=2.5,
            markersize=12,
            color='#3498db',
            label='Quantum Profile (Tunneling Corrected)'
        )

        # Highlight the tunneling advantage at the Transition State (Index 2)
        if len(classical_energies) > 2:
            ts_diff = classical_energies[2] - quantum_energies[2]
            ax.annotate(
                f'Quantum Tunneling\n-{ts_diff:.3f} Ha',
                xy=(2, quantum_energies[2]),
                xytext=(2.2, classical_energies[2]),
                arrowprops=dict(facecolor='green', shrink=0.05),
                fontsize=10,
                color='green',
                fontweight='bold'
            )

    # Annotate classical energies
    for i, energy in enumerate(classical_energies):
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
    ax.legend()

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


def show_simulation_provenance(sim_result: dict):
    """Display Hamiltonian provenance for VQE/HF results when available."""
    generation_mode = sim_result.get("generation_mode", "Static Database")
    st.metric("Hamiltonian Source", generation_mode)

    active_electrons = int(sim_result.get("active_electrons", 0) or 0)
    frozen_orbitals = int(sim_result.get("frozen_orbitals", 0) or 0)

    if generation_mode == "Dynamic":
        active_orbitals = max(1, int(sim_result.get("num_qubits", 0) // 2))
        st.caption(f"Active Space: [{active_electrons}e, {active_orbitals}o]")
        st.caption(f"Core Orbitals Frozen: {frozen_orbitals}")

    noise_model = sim_result.get("noise_model", "None")
    if noise_model and noise_model != "None":
        st.caption(f"Noise Model: {noise_model}")

    source = sim_result.get("hamiltonian_source")
    if source == "approximate_fallback":
        st.warning("Using approximate fallback Hamiltonian for an unsupported molecule. Interpret results as exploratory.")
    elif source == "dynamic_pyscf":
        st.success("Dynamic PySCF Hamiltonian generated successfully.")
    elif source == "database":
        st.caption("Hamiltonian source: curated static database")


def _map_custom_reaction_to_qml_key(parsed_reaction: dict) -> str:
    """Map parsed custom reactions to the closest trained QSVM reaction key."""
    if not parsed_reaction:
        return "H2_O2"

    reactants = parsed_reaction.get("reactants", [])
    if "N#N" in reactants:
        return "N2_H2"

    reaction_type = parsed_reaction.get("type", "")
    if reaction_type == "oxidation":
        return "H2_O2"
    return "CO2_reduction"


def catalyst_input_widget(label: str, key_prefix: str, placeholder: str = "e.g., Pt, Fe, NiO") -> str:
    """Render catalyst input as discovered-candidate select + custom entry."""
    discovered = st.session_state.get("discovered_catalysts", [])
    if discovered:
        mode = st.selectbox(
            f"{label} Input Mode:",
            options=["Select Discovered Catalyst", "Type Custom SMILES"],
            key=f"{key_prefix}_mode",
        )
        if mode == "Select Discovered Catalyst":
            return st.selectbox(
                label,
                options=discovered,
                key=f"{key_prefix}_selected",
            )

    return st.text_input(
        label,
        placeholder=placeholder,
        key=f"{key_prefix}_custom",
    )

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
        - **Stochastic Quantum Latent-Space Generator**: Generate novel catalyst candidates
        - **Rule-Based Chemical Classifier**: Classify catalyst effectiveness
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
        st.caption("VQE, QSVM, Rule-Based Chemical Classifier, Stochastic Quantum Latent-Space Generator")

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
           - Let the Stochastic Quantum Latent-Space Generator generate catalyst candidates
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
        - **Feature engineering**: 16D descriptors (physicochemical + Coulomb-like + fingerprint)

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

    This feature uses **Stochastic Quantum Latent-Space Generator** to generate catalyst candidates, **Rule-Based Chemical Classifier** to classify them,
    and **VQE** to validate their effectiveness through quantum simulations.
    """)

    # User inputs
    st.markdown("### ⚙️ Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        reaction_options = {
            "H2_O2": "H₂ + O₂ → H₂O (Fuel Cell / Water Formation)",
            "N2_H2": "N₂ + 3H₂ → 2NH₃ (Haber Process / Ammonia)",
            "CO2_reduction": "CO₂ + H₂ → CO + H₂O (Carbon Capture)",
            "CUSTOM": "Custom Reaction (Enter Equation)",
        }

        selected_reaction = st.selectbox(
            "Select Target Reaction:",
            options=list(reaction_options.keys()),
            format_func=lambda x: reaction_options[x]
        )

        custom_reaction_equation = ""
        parsed_custom_reaction = None

        if selected_reaction == "CUSTOM":
            custom_reaction_equation = st.text_input(
                "Custom Reaction Equation:",
                placeholder="e.g., CO + H2O -> CO2 + H2",
                key="custom_discovery_equation",
            )
            if custom_reaction_equation:
                parsed_custom_reaction = parse_dynamic_reaction(custom_reaction_equation)
                if parsed_custom_reaction.get("error"):
                    st.error(parsed_custom_reaction["error"])
                else:
                    st.info(f"**Reaction:** {parsed_custom_reaction['equation']}")
        else:
            reaction_info = get_reaction_info(selected_reaction)
            st.info(f"**Reaction:** {reaction_info['equation']}")

    with col2:
        num_candidates = st.slider("Number of Candidates:", 3, 10, 5)

    discover_btn = st.button("🚀 Discover Catalysts", type="primary", use_container_width=True)

    if discover_btn:
        st.info("Generating novel catalytic structures via RDKit mutation engine...")
        with st.spinner("🧬 Sampling Statevector probabilities and generating stochastic catalyst mutations..."):
            if selected_reaction == "CUSTOM":
                parsed_custom_reaction = parse_dynamic_reaction(custom_reaction_equation) if custom_reaction_equation else {"error": "Please enter a custom reaction equation."}
                if parsed_custom_reaction.get("error"):
                    st.error(parsed_custom_reaction["error"])
                    st.stop()

                discovery_reaction_key = _map_custom_reaction_to_qml_key(parsed_custom_reaction)
                pathway_reaction_input = parsed_custom_reaction
            else:
                discovery_reaction_key = selected_reaction
                pathway_reaction_input = selected_reaction

            # Discover catalysts
            candidates = cached_discover_catalysts(discovery_reaction_key, num_candidates)

            if candidates:
                st.success(f"✅ Generated {len(candidates)} catalyst candidates!")

                # Persist discovered catalysts across tabs for pipeline flow.
                discovered = st.session_state.discovered_catalysts
                for cand in candidates:
                    smiles = cand.get('smiles')
                    if smiles and smiles not in discovered:
                        discovered.append(smiles)

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
                            apply_noise = st.toggle(
                                "🎛️ Simulate Quantum Hardware Noise (NISQ)",
                                key=f"noise_discovery_{i}",
                                value=False,
                            )
                            if st.button(f"Run VQE + Pathway", key=f"sim_{i}"):
                                with st.spinner("Running quantum simulation..."):
                                    # Run VQE
                                    vqe_result = cached_vqe_simulation(cand['smiles'], apply_noise=apply_noise)

                                    if not vqe_result.get('error'):
                                        st.metric("Ground State Energy", f"{vqe_result['energy']:.6f} Ha")
                                        show_simulation_provenance(vqe_result)

                                        # Plot convergence
                                        fig_conv = plot_convergence(
                                            vqe_result['convergence'],
                                            f"VQE Convergence: {cand['smiles']}"
                                        )
                                        st.pyplot(fig_conv)
                                        plt.close()

                                        # Run pathway
                                        pathway = cached_pathway_simulation(cand['smiles'], pathway_reaction_input)

                                        if not pathway.get('error'):
                                            fig_path = plot_energy_landscape(
                                                states=pathway['states'],
                                                classical_energies=pathway['energies'],
                                                quantum_energies=pathway.get('quantum_energies'),
                                                title=f"Reaction Pathway: {cand['smiles']}"
                                            )
                                            st.pyplot(fig_path)
                                            plt.close()

                                            st.metric(
                                                "Predicted Turnover Frequency (TOF)",
                                                f"{pathway.get('turnover_frequency_s', 0.0):.2e} s^-1"
                                            )
                                    else:
                                        st.error(f"Error: {vqe_result['error']}")

                # Comparison chart
                st.markdown("### 📊 Candidate Comparison")
                score_dict = {f"#{i+1} {c['smiles']}": c['catalyst_score'] for i, c in enumerate(candidates[:5])}
                fig_comp = plot_comparison_bar(score_dict, "Catalyst Scores", "Score (0-100)")
                st.pyplot(fig_comp)
                plt.close()

                csv_report = export_discovery_batch_to_csv(candidates)
                st.download_button(
                    label="📥 Download Catalyst Discovery Report (CSV)",
                    data=csv_report,
                    file_name="quantum_discovery_report.csv",
                    mime="text/csv",
                )

                # Save to history
                save_result_to_history({
                    'type': 'AI Discovery',
                    'reaction': custom_reaction_equation if selected_reaction == "CUSTOM" else selected_reaction,
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
        game_options = list(REACTION_DATABASE.keys()) + ["CUSTOM"]
        game_reaction = st.selectbox(
            "Select Reaction:",
            options=game_options,
            format_func=lambda x: REACTION_DATABASE[x]['name'] if x in REACTION_DATABASE else "Custom Reaction (Enter Equation)"
        )

        custom_game_equation = ""
        parsed_game_reaction = None

        if game_reaction == "CUSTOM":
            custom_game_equation = st.text_input(
                "Custom Reaction Equation:",
                placeholder="e.g., CO + H2O -> CO2 + H2",
                key="custom_game_equation",
            )
            if custom_game_equation:
                parsed_game_reaction = parse_dynamic_reaction(custom_game_equation)
                if parsed_game_reaction.get("error"):
                    st.error(parsed_game_reaction["error"])
                else:
                    st.info(f"**Equation:** {parsed_game_reaction['equation']}")
        else:
            reaction_data = REACTION_DATABASE[game_reaction]
            st.info(f"**Equation:** {reaction_data['equation']}")

        # Show ideal catalysts hint
        if st.checkbox("Show Hint") and game_reaction != "CUSTOM":
            st.warning(f"💡 Ideal catalysts: {', '.join(reaction_data['ideal_catalysts'])}")

    with col2:
        user_catalyst = catalyst_input_widget(
            "Your Catalyst Guess:",
            key_prefix="learning",
            placeholder="e.g., Pt, Fe, NiO, iron, platinum",
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
                if game_reaction == "CUSTOM":
                    parsed_game_reaction = parse_dynamic_reaction(custom_game_equation) if custom_game_equation else {"error": "Please enter a custom reaction equation."}
                    if parsed_game_reaction.get("error"):
                        st.error(parsed_game_reaction["error"])
                        st.stop()

                    ideal_catalyst = parsed_game_reaction.get('ideal_catalysts', ['[Pt]'])[0]
                    scoring_reaction_key = _map_custom_reaction_to_qml_key(parsed_game_reaction)
                    pathway_reaction_input = parsed_game_reaction
                else:
                    ideal_catalyst = reaction_data['ideal_catalysts'][0]
                    scoring_reaction_key = game_reaction
                    pathway_reaction_input = game_reaction

                # Score user's choice
                scoring_result = score_user_catalyst(user_smiles, ideal_catalyst, scoring_reaction_key)

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

                    # Rule-Based Chemical Classifier Category
                    st.markdown(f"**Catalyst Category:** {scoring_result['vqc_category']}")

                    # Run full analysis
                    st.markdown("---")
                    st.markdown("### 🔬 Deep Dive Analysis")

                    if st.button("Run Full Quantum Simulation"):
                        with st.spinner("Running comprehensive analysis..."):
                            # VQE simulation
                            vqe_result = cached_vqe_simulation(user_smiles)

                            col_vqe1, col_vqe2 = st.columns(2)

                            with col_vqe1:
                                if not vqe_result.get('error'):
                                    st.metric("Ground State Energy", f"{vqe_result['energy']:.6f} Ha")
                                    st.metric("VQE Iterations", vqe_result['iterations'])
                                    st.metric("Qubits Used", vqe_result['num_qubits'])
                                    show_simulation_provenance(vqe_result)
                                else:
                                    st.error(f"VQE Error: {vqe_result['error']}")

                            with col_vqe2:
                                if not vqe_result.get('error'):
                                    fig_conv = plot_convergence(vqe_result['convergence'])
                                    st.pyplot(fig_conv)
                                    plt.close()

                            # Reaction pathway
                            pathway = cached_pathway_simulation(user_smiles, pathway_reaction_input)

                            if not pathway.get('error'):
                                st.markdown("### 🛤️ Reaction Energy Pathway")

                                col_path1, col_path2 = st.columns([2, 1])

                                with col_path1:
                                    fig_path = plot_energy_landscape(
                                        states=pathway['states'],
                                        classical_energies=pathway['energies'],
                                        quantum_energies=pathway.get('quantum_energies'),
                                        title="Reaction Pathway"
                                    )
                                    st.pyplot(fig_path)
                                    plt.close()

                                with col_path2:
                                    st.metric("Activation Barrier",
                                            f"{pathway['activation_barrier_forward']*27.211:.2f} eV")
                                    st.metric("Reaction Enthalpy",
                                            f"{pathway['reaction_enthalpy']*27.211:.2f} eV")
                                    st.metric("Pathway Score", f"{pathway['catalyst_score']:.2f}/100")
                                    st.metric(
                                        "Predicted Turnover Frequency (TOF)",
                                        f"{pathway.get('turnover_frequency_s', 0.0):.2e} s^-1"
                                    )

                                    if pathway['is_ideal_catalyst']:
                                        st.success("✨ This IS an ideal catalyst!")
                                    else:
                                        st.info("Try one of the ideal catalysts for comparison!")

                    # Save to history
                    save_result_to_history({
                        'type': 'Learning Game',
                        'reaction': custom_game_equation if game_reaction == "CUSTOM" else game_reaction,
                        'user_catalyst': user_catalyst,
                        'score': score
                    })

                else:
                    st.error(f"❌ Scoring error: {scoring_result.get('error')}")
                    if "feature" in str(scoring_result.get('error', '')).lower():
                        st.info("Try a simpler catalyst input (single metal or small oxide), for example [Pt], [Fe], or [Ni]=O.")

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
    test_molecule = catalyst_input_widget(
        "Enter Molecule/Catalyst:",
        key_prefix="comparison",
        placeholder="e.g., H2, H2O, Pt, Fe",
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
                            show_simulation_provenance(chem_comp['vqe'])

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
                        correlation_energy = chem_comp['correlation_energy']

                        st.metric(
                            label="Electron Correlation Energy Captured",
                            value=f"{correlation_energy:.4f} Ha"
                        )
                        st.info(
                            "Hartree-Fock (Classical) ignores electron-electron correlation. "
                            "The VQE (Quantum) captures this correlation, which is critical for "
                            "accurate catalyst binding energies."
                        )

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

            explorer_noise = st.toggle(
                "🎛️ Simulate Quantum Hardware Noise (NISQ)",
                key="noise_explorer",
                value=False,
            )

            if st.button("Run VQE Simulation"):
                with st.spinner("Running VQE..."):
                    vqe_result = cached_vqe_simulation(smiles, apply_noise=explorer_noise)

                    if not vqe_result.get('error'):
                        sim_col1, sim_col2 = st.columns([1, 2])

                        with sim_col1:
                            st.metric("Ground State Energy", f"{vqe_result['energy']:.6f} Ha")
                            st.metric("Iterations", vqe_result['iterations'])
                            st.metric("Qubits", vqe_result['num_qubits'])
                            st.caption(vqe_result['method'])
                            show_simulation_provenance(vqe_result)

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
                    pathway = cached_pathway_simulation(smiles, test_reaction)

                    if not pathway.get('error'):
                        cat_col1, cat_col2 = st.columns([2, 1])

                        with cat_col1:
                            fig_path = plot_energy_landscape(
                                states=pathway['states'],
                                classical_energies=pathway['energies'],
                                quantum_energies=pathway.get('quantum_energies'),
                                title=f"Pathway: {validation['formula']}"
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

st.markdown('<div class="footer-container">', unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div class="footer-text">
    <div class="footer-name">Baratam Praneeth Gupta</div>
    <div class="footer-subtitle">Member of Anu Tattva (Part of Singularity)</div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
