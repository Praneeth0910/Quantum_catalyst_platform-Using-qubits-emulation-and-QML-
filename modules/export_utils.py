"""
Export and Reporting Module
============================

Generate PDF reports and export data in various formats for analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import json
import pandas as pd
from typing import Dict, List


def create_pdf_report(results: Dict, filename: str = None) -> str:
    """
    Generate comprehensive PDF report from analysis results.

    Args:
        results: Analysis results dictionary
        filename: Output PDF filename (auto-generated if None)

    Returns:
        Path to generated PDF file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_catalyst_report_{timestamp}.pdf"

    with PdfPages(filename) as pdf:
        # Page 1: Title and Summary
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Quantum Catalyst Analysis Report', fontsize=24, fontweight='bold')

        ax = fig.add_subplot(111)
        ax.axis('off')

        # Report metadata
        report_text = f"""
        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        Platform: Quantum Catalyst Discovery Platform

        Analysis Type: {results.get('type', 'Unknown')}

        ─────────────────────────────────────────────────

        EXECUTIVE SUMMARY

        This report contains quantum computing and machine learning
        analysis results for catalyst discovery and evaluation.

        All energies calculated using real Variational Quantum
        Eigensolver (VQE) algorithms running on Qiskit.

        ─────────────────────────────────────────────────
        """

        ax.text(0.1, 0.8, report_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', family='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Add result-specific pages
        result_type = results.get('type', '')

        if result_type == 'AI Discovery' and 'candidates' in results:
            _add_discovery_pages(pdf, results)

        elif result_type == 'Learning Game':
            _add_learning_game_pages(pdf, results)

        elif result_type == 'Comparison':
            _add_comparison_pages(pdf, results)

        # Final page: Methodology
        _add_methodology_page(pdf)

    return filename


def export_discovery_batch_to_csv(candidates: list) -> str:
    """
    Export discovery candidates to CSV string.

    Fields:
    - SMILES
    - QSVM Score
    - Classification
    """
    rows = []
    for cand in candidates or []:
        rows.append({
            "SMILES": cand.get("smiles", ""),
            "QSVM Score": cand.get("catalyst_score", ""),
            "Classification": cand.get("classification", ""),
        })

    df = pd.DataFrame(rows, columns=["SMILES", "QSVM Score", "Classification"])
    return df.to_csv(index=False)


def _add_discovery_pages(pdf: PdfPages, results: Dict):
    """Add AI discovery result pages to PDF."""
    candidates = results.get('candidates', [])

    # Candidates summary page
    fig, ax = plt.subplots(figsize=(8.5, 11))

    ax.text(0.5, 0.95, f'AI Catalyst Discovery Results',
            ha='center', fontsize=18, fontweight='bold',
            transform=ax.transAxes)

    ax.text(0.5, 0.90, f'Reaction: {results.get("reaction", "Unknown")}',
            ha='center', fontsize=14, transform=ax.transAxes)

    # Table of candidates
    table_data = []
    for i, cand in enumerate(candidates[:10]):
        table_data.append([
            f"#{i+1}",
            cand['smiles'],
            cand.get('metal_type', 'N/A'),
            f"{cand['catalyst_score']:.2f}",
            cand['classification']
        ])

    table = ax.table(cellText=table_data,
                    colLabels=['Rank', 'Catalyst', 'Metal', 'Score', 'Class'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0.1, 0.3, 0.8, 0.5])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Score comparison chart
    fig, ax = plt.subplots(figsize=(8.5, 6))

    scores = [c['catalyst_score'] for c in candidates[:10]]
    labels = [f"#{i+1}\n{c['smiles']}" for i, c in enumerate(candidates[:10])]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(scores)))

    bars = ax.bar(range(len(scores)), scores, color=colors, edgecolor='black', linewidth=1.5)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{scores[i]:.1f}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Catalyst Score (0-100)', fontsize=12, fontweight='bold')
    ax.set_title('Generated Catalyst Candidates - Score Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def _add_learning_game_pages(pdf: PdfPages, results: Dict):
    """Add learning game result pages to PDF."""
    fig, ax = plt.subplots(figsize=(8.5, 11))

    ax.text(0.5, 0.95, 'Learning Game Results',
            ha='center', fontsize=18, fontweight='bold',
            transform=ax.transAxes)

    score = results.get('score', 0)

    # Score display
    score_color = 'green' if score >= 80 else 'orange' if score >= 60 else 'red'

    ax.text(0.5, 0.80, f'Final Score: {score:.1f}/100',
            ha='center', fontsize=24, fontweight='bold',
            color=score_color, transform=ax.transAxes)

    ax.text(0.5, 0.70, f'Reaction: {results.get("reaction", "Unknown")}',
            ha='center', fontsize=14, transform=ax.transAxes)

    ax.text(0.5, 0.65, f'Your Catalyst: {results.get("user_catalyst", "Unknown")}',
            ha='center', fontsize=14, transform=ax.transAxes)

    # Performance assessment
    if score >= 80:
        assessment = "Excellent! You have a strong understanding of catalyst chemistry."
    elif score >= 60:
        assessment = "Good work! You're on the right track."
    else:
        assessment = "Keep learning! Review the ideal catalysts for this reaction."

    ax.text(0.5, 0.50, assessment,
            ha='center', fontsize=12, style='italic',
            transform=ax.transAxes, wrap=True)

    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def _add_comparison_pages(pdf: PdfPages, results: Dict):
    """Add comparison result pages to PDF."""
    fig, ax = plt.subplots(figsize=(8.5, 11))

    ax.text(0.5, 0.95, 'Quantum vs Classical Comparison',
            ha='center', fontsize=18, fontweight='bold',
            transform=ax.transAxes)

    ax.text(0.5, 0.88, f'Analysis Type: {results.get("comparison_type", "Unknown")}',
            ha='center', fontsize=14, transform=ax.transAxes)

    ax.text(0.5, 0.83, f'Molecule: {results.get("molecule", "Unknown")}',
            ha='center', fontsize=14, transform=ax.transAxes)

    comparison_text = """
    ─────────────────────────────────────────────────

    This comparison demonstrates the quantum advantage
    in computational chemistry and machine learning.

    Quantum methods (VQE, QSVM) provide more accurate
    results compared to classical approximations.

    ─────────────────────────────────────────────────
    """

    ax.text(0.5, 0.60, comparison_text,
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes, family='monospace')

    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def _add_methodology_page(pdf: PdfPages):
    """Add methodology explanation page."""
    fig, ax = plt.subplots(figsize=(8.5, 11))

    ax.text(0.5, 0.95, 'Methodology',
            ha='center', fontsize=18, fontweight='bold',
            transform=ax.transAxes)

    methodology_text = """
    QUANTUM ALGORITHMS

    • VQE (Variational Quantum Eigensolver)
      - Hybrid quantum-classical algorithm
      - Finds molecular ground state energies
      - More accurate than Hartree-Fock

    • QSVM (Quantum Support Vector Machine)
      - Quantum kernel methods for classification
      - Enhanced pattern recognition
      - Scores catalyst effectiveness

    • QGAN (Quantum Generative Adversarial Network)
      - Generates novel catalyst candidates
      - Quantum-enhanced feature learning

    ─────────────────────────────────────────────────

    CLASSICAL BASELINES

    • Hartree-Fock (HF)
      - Mean-field approximation
      - Fast but less accurate

    • DFT (Density Functional Theory)
      - Exchange-correlation functionals
      - Balance of speed and accuracy

    • Classical ML
      - Random Forest, SVM, Gradient Boosting
      - Standard machine learning methods

    ─────────────────────────────────────────────────

    CHEMISTRY MODELS

    • D-band Model
      - Predicts catalyst-adsorbate binding
      - Nørskov's theoretical framework

    • BEP Relation
      - Brønsted-Evans-Polanyi correlation
      - Estimates activation energies

    ─────────────────────────────────────────────────

    Platform: Quantum Catalyst Discovery Platform
    Framework: Qiskit, RDKit, Scikit-learn
    License: Educational & Research Use
    """

    ax.text(0.1, 0.85, methodology_text,
            verticalalignment='top', fontsize=9,
            transform=ax.transAxes, family='monospace')

    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def export_to_csv(results_history: List[Dict], filename: str = None) -> str:
    """
    Export results history to CSV format.

    Args:
        results_history: List of result dictionaries
        filename: Output CSV filename

    Returns:
        Path to generated CSV file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_catalyst_data_{timestamp}.csv"

    # Flatten data
    csv_data = []
    for result in results_history:
        row = {
            'Timestamp': result.get('timestamp', ''),
            'Type': result.get('type', ''),
            'Reaction': result.get('reaction', 'N/A'),
        }

        if result.get('type') == 'AI Discovery' and 'candidates' in result:
            row['Num_Candidates'] = len(result['candidates'])
            if result['candidates']:
                row['Top_Score'] = result['candidates'][0].get('catalyst_score', 0)
                row['Top_Catalyst'] = result['candidates'][0].get('smiles', '')

        elif result.get('type') == 'Learning Game':
            row['User_Catalyst'] = result.get('user_catalyst', '')
            row['Score'] = result.get('score', 0)

        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(filename, index=False)

    return filename


def export_to_json(results_history: List[Dict], filename: str = None) -> str:
    """
    Export results history to JSON format.

    Args:
        results_history: List of result dictionaries
        filename: Output JSON filename

    Returns:
        Path to generated JSON file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_catalyst_data_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results_history, f, indent=2)

    return filename


def generate_summary_stats(results_history: List[Dict]) -> Dict:
    """
    Generate summary statistics from results history.

    Args:
        results_history: List of result dictionaries

    Returns:
        Dictionary with summary statistics
    """
    stats = {
        'total_analyses': len(results_history),
        'by_type': {},
        'reactions_analyzed': set(),
        'catalysts_tested': set(),
        'avg_scores': {}
    }

    for result in results_history:
        result_type = result.get('type', 'Unknown')
        stats['by_type'][result_type] = stats['by_type'].get(result_type, 0) + 1

        if 'reaction' in result:
            stats['reactions_analyzed'].add(result['reaction'])

        if result.get('type') == 'AI Discovery' and 'candidates' in result:
            for cand in result['candidates']:
                stats['catalysts_tested'].add(cand.get('smiles', ''))

        elif result.get('type') == 'Learning Game':
            if 'user_catalyst' in result:
                stats['catalysts_tested'].add(result['user_catalyst'])

            if 'score' in result:
                if result_type not in stats['avg_scores']:
                    stats['avg_scores'][result_type] = []
                stats['avg_scores'][result_type].append(result['score'])

    # Convert sets to lists and calculate averages
    stats['reactions_analyzed'] = list(stats['reactions_analyzed'])
    stats['catalysts_tested'] = list(stats['catalysts_tested'])

    for result_type, scores in stats['avg_scores'].items():
        stats['avg_scores'][result_type] = np.mean(scores) if scores else 0

    return stats
