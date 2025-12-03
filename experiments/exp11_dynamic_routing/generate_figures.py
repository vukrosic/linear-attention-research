"""
Generate all figures for the research paper

Run after training both baseline and dynamic models:
    python generate_figures.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Use publication-quality settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 300

# Color palette
COLORS = {
    'gdn': '#4A90E2',      # Blue
    'softmax': '#F5A623',  # Orange
    'baseline': '#4A90E2',
    'dynamic': '#F5A623',
    'grid': '#E0E0E0',
    'bg': '#F8F9FA',
}


def load_results(config_name):
    """Load training results for a configuration"""
    results_dir = Path(__file__).parent / f"results_{config_name}"
    results_file = results_dir / "training_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)


def figure_2_training_curves(baseline_results, dynamic_results, save_path):
    """Generate Figure 2: Training curves comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Dynamics: Baseline vs Dynamic Routing', fontsize=16, fontweight='bold')
    
    # Extract history data (you'll need to modify based on your actual data structure)
    # This is a placeholder - adjust to match your results format
    baseline_train = baseline_results.get('train_history', [])
    baseline_val = baseline_results.get('val_history', [])
    dynamic_train = dynamic_results.get('train_history', [])
    dynamic_val = dynamic_results.get('val_history', [])
    
    # Subplot 1: Training Loss
    if baseline_train and dynamic_train:
        b_steps = [h['step'] for h in baseline_train]
        b_loss = [h['loss'] for h in baseline_train]
        d_steps = [h['step'] for h in dynamic_train]
        d_loss = [h['loss'] for h in dynamic_train]
        
        axes[0, 0].plot(b_steps, b_loss, color=COLORS['baseline'], 
                       linewidth=2, label='Baseline', linestyle='-')
        axes[0, 0].plot(d_steps, d_loss, color=COLORS['dynamic'], 
                       linewidth=2, label='Dynamic', linestyle='--')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].set_title('Training Loss Over Time')
        axes[0, 0].grid(True, alpha=0.3, color=COLORS['grid'])
        axes[0, 0].legend()
    
    # Subplot 2: Validation Loss
    if baseline_val and dynamic_val:
        b_steps = [h['step'] for h in baseline_val]
        b_loss = [h['loss'] for h in baseline_val]
        d_steps = [h['step'] for h in dynamic_val]
        d_loss = [h['loss'] for h in dynamic_val]
        
        axes[0, 1].plot(b_steps, b_loss, color=COLORS['baseline'], 
                       linewidth=2, marker='o', label='Baseline', linestyle='-')
        axes[0, 1].plot(d_steps, d_loss, color=COLORS['dynamic'], 
                       linewidth=2, marker='s', label='Dynamic', linestyle='--')
        
        # Mark best validation points
        best_b_idx = np.argmin(b_loss)
        best_d_idx = np.argmin(d_loss)
        axes[0, 1].scatter(b_steps[best_b_idx], b_loss[best_b_idx], 
                          s=200, marker='*', color='gold', edgecolors='black', 
                          linewidths=2, zorder=5, label='Best')
        
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Validation Loss')
        axes[0, 1].set_title('Validation Loss (Primary Metric)')
        axes[0, 1].grid(True, alpha=0.3, color=COLORS['grid'])
        axes[0, 1].legend()
    
    # Subplot 3: Validation Accuracy
    if baseline_val and dynamic_val:
        b_acc = [h.get('accuracy', 0) * 100 for h in baseline_val]
        d_acc = [h.get('accuracy', 0) * 100 for h in dynamic_val]
        
        axes[1, 0].plot(b_steps, b_acc, color=COLORS['baseline'], 
                       linewidth=2, marker='o', label='Baseline')
        axes[1, 0].plot(d_steps, d_acc, color=COLORS['dynamic'], 
                       linewidth=2, marker='s', label='Dynamic', linestyle='--')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Next-Token Accuracy (%)')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].grid(True, alpha=0.3, color=COLORS['grid'])
        axes[1, 0].legend()
    
    # Subplot 4: Load Balance Loss (Dynamic only)
    if dynamic_train:
        d_aux = [h.get('aux_loss', 0) for h in dynamic_train]
        
        # Baseline is flat at 0
        axes[1, 1].axhline(y=0, color='gray', linewidth=2, 
                          linestyle='-', label='Baseline (N/A)')
        axes[1, 1].plot(d_steps, d_aux, color=COLORS['dynamic'], 
                       linewidth=2, label='Dynamic', linestyle='--')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Load Balance Loss')
        axes[1, 1].set_title('Load Balancing Loss (Dynamic Only)')
        axes[1, 1].grid(True, alpha=0.3, color=COLORS['grid'])
        axes[1, 1].legend()
        axes[1, 1].annotate('Target: Decreasing trend',
                           xy=(0.5, 0.95), xycoords='axes fraction',
                           ha='center', va='top', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 2 saved: {save_path}")
    plt.close()


def figure_3_routing_distribution(dynamic_results, save_path):
    """Generate Figure 3: Routing distribution over time"""
    
    # Extract routing statistics over time
    # You'll need to log this during training
    # Placeholder: assume we have this data
    val_history = dynamic_results.get('val_history', [])
    
    if not val_history or 'layer_1_gdn_pct' not in val_history[0]:
        print("⚠️  Routing statistics not found in results")
        print("   Make sure to log routing stats during training")
        return
    
    steps = [h['step'] for h in val_history]
    l1_gdn = [h.get('layer_1_gdn_pct', 50) for h in val_history]
    l1_attn = [h.get('layer_1_attn_pct', 50) for h in val_history]
    l2_gdn = [h.get('layer_2_gdn_pct', 50) for h in val_history]
    l2_attn = [h.get('layer_2_attn_pct', 50) for h in val_history]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Routing Distribution Evolution During Training', 
                 fontsize=16, fontweight='bold')
    
    # Layer 1
    axes[0].plot(steps, l1_gdn, color=COLORS['gdn'], 
                linewidth=2.5, label='GDN')
    axes[0].plot(steps, l1_attn, color=COLORS['softmax'], 
                linewidth=2.5, label='Softmax')
    axes[0].axhline(y=50, color='black', linestyle=':', 
                   linewidth=1.5, label='Balanced (50%)')
    axes[0].axhline(y=90, color='red', linestyle=':', 
                   linewidth=1, alpha=0.5, label='Collapse threshold')
    axes[0].axhline(y=10, color='red', linestyle=':', 
                   linewidth=1, alpha=0.5)
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Routing Percentage (%)')
    axes[0].set_title('Layer 1: Routing Distribution')
    axes[0].set_ylim([0, 100])
    axes[0].grid(True, alpha=0.3, color=COLORS['grid'])
    axes[0].legend(loc='upper right')
    
    # Layer 2
    axes[1].plot(steps, l2_gdn, color=COLORS['gdn'], 
                linewidth=2.5, label='GDN')
    axes[1].plot(steps, l2_attn, color=COLORS['softmax'], 
                linewidth=2.5, label='Softmax')
    axes[1].axhline(y=50, color='black', linestyle=':', 
                   linewidth=1.5, label='Balanced (50%)')
    axes[1].axhline(y=90, color='red', linestyle=':', 
                   linewidth=1, alpha=0.5, label='Collapse threshold')
    axes[1].axhline(y=10, color='red', linestyle=':', 
                   linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Routing Percentage (%)')
    axes[1].set_title('Layer 2: Routing Distribution')
    axes[1].set_ylim([0, 100])
    axes[1].grid(True, alpha=0.3, color=COLORS['grid'])
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 3 saved: {save_path}")
    plt.close()


def main():
    """Generate all figures"""
    print("="*70)
    print("Generating Figures for Research Paper")
    print("="*70)
    
    # Load results
    try:
        baseline_results = load_results('baseline')
        print("✓ Loaded baseline results")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Run: python run_experiment.py --config baseline")
        return
    
    try:
        dynamic_results = load_results('dynamic')
        print("✓ Loaded dynamic results")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Run: python run_experiment.py --config dynamic")
        return
    
    # Create figures directory
    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Generate each figure
    print("\nGenerating figures...")
    
    try:
        figure_2_training_curves(
            baseline_results, 
            dynamic_results,
            figures_dir / "figure2_training_curves.png"
        )
    except Exception as e:
        print(f"⚠️  Could not generate Figure 2: {e}")
    
    try:
        figure_3_routing_distribution(
            dynamic_results,
            figures_dir / "figure3_routing_distribution.png"
        )
    except Exception as e:
        print(f"⚠️  Could not generate Figure 3: {e}")
    
    print("\n" + "="*70)
    print("✅ Figure generation complete!")
    print(f"   Figures saved to: {figures_dir}/")
    print("\nNext steps:")
    print("   1. Review generated figures")
    print("   2. Create Figure 1 (architecture diagram) manually")
    print("   3. Create Figure 4 (architecture comparison) manually")
    print("   4. Fill in results placeholders in PAPER.md")
    print("="*70)


if __name__ == "__main__":
    main()
