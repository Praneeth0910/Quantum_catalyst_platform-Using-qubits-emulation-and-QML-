import streamlit as st
import time

def show_reaction_animation(states, energies):
    """
    Simulates a visual bond-forming/breaking progress bar based on energy states.
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, state in enumerate(states):
        # Update progress based on state
        percent = int((i + 1) / len(states) * 100)
        progress_bar.progress(percent)
        
        # Display simplified chemical logic
        if state == "Reactant":
            status_text.text("🔹 Step 1: Reactants adsorbing to catalyst surface...")
        elif state == "Intermediate":
            status_text.text("⚡ Step 2: Transition state - Breaking/Forming bonds...")
        else:
            status_text.text("✅ Step 3: Product formed and desorbing.")
        
        time.sleep(1) # Simulated animation speed