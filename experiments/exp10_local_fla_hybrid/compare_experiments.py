"""
Compare all exp10 experiment variants
Shows training curves and results for all completed experiments

Usage:
    python compare_experiments.py
    python compare_experiments.py --variants h100_deltanet h100_transformer h100_hybrid_sparse
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_experiment_results(exp_variant):
    """Load results for a specific H100 experiment variant"""
    exp_dir = Path(__file__).parent
    
    # All H100 variants use results_{variant_name}
    results_dir = exp_dir / f"results_{exp_variant}"
    results_file = results_dir / "training_results.json"
    
    if not results_file.exists():
        return None
    
    with open(results_file) as f:
        return json.load(f)


def get_all_completed_experiments():
    """Find all completed H100 experiment variants"""
    exp_dir = Path(__file__).parent
    variants = []
    
    # Check all H100 variants
    h100_variants = [
        ('h100_deltanet', 'H100 Pure DeltaNet'),
        ('h100_transformer', 'H100 Pure Transformer'),
        ('h100_hybrid_sparse', 'H100 Hybrid Sparse (17%)'),
        ('h100_hybrid_alternating', 'H100 Hybrid Alternating (50%)'),
        ('h100_hybrid_late', 'H100 Hybrid Late (33%)'),
    ]
    
    for var_id, var_name in h100_variants:
        if (exp_dir / f"results_{var_id}" / "training_results.json").exists():
            variants.append((var_id, var_name))
    
    return variants


def compare_experiments(variants=None):
    """Compare multiple experiment variants"""
    
    if variants is None:
        # Auto-detect all completed experiments
        variants = get_all_completed_experiments()
        if not variants:
            print("❌ No completed experiments found!")
            return
        variant_ids = [v[0] for v in variants]
    else:
        # Use specified variants
        variant_ids = variants
    
    print("="*70)
    print("EXPERIMENT 10 - COMPARISON (Local FLA Clone)")
    print("="*70)
    print(f"Comparing {len(variant_ids)} experiment variants\n")
    
    # Load all results
    results_data = []
    for var_id in variant_ids:
        results = load_experiment_results(var_id)
        if results:
            results_data.append((var_id, results))
            print(f"✓ Loaded: {var_id}")
        else:
            print(f"✗ Not found: {var_id}")
    
    if not results_data:
        print("\n❌ No experiment results to compare!")
        return
    
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}\n")
    
    # Print comparison table
    print(f"{'Experiment':<30} {'Model':<20} {'Val Loss':<12} {'Time (s)':<12}")
    print("-" * 74)
    
    for var_id, results in results_data:
        exp_name = results.get('experiment_name', var_id)
        model_str = f"{results['config']['hidden_size']}d, {results['config']['num_layers']}L"
        val_loss = results['results']['best_val_loss']
        time = results['results']['total_time']
        
        print(f"{exp_name:<30} {model_str:<20} {val_loss:<12.4f} {time:<12.1f}")
    
    print("\n" + "="*70)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Val Loss comparison
    var_names = []
    val_losses = []
    colors = []
    
    color_map = {
        'deltanet': '#3498db',
        'transformer': '#e74c3c',
        'hybrid_sparse': '#2ecc71',
        'hybrid_alternating': '#f39c12',
        'hybrid_late': '#9b59b6',
    }
    
    for var_id, results in results_data:
        var_names.append(var_id.replace('_', '\n'))
        val_losses.append(results['results']['best_val_loss'])
        
        # Assign color based on experiment type
        color = '#95a5a6'  # default gray
        for key, col in color_map.items():
            if key in var_id:
                color = col
                break
        colors.append(color)
    
    ax1.bar(range(len(var_names)), val_losses, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Experiment Variant', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Best Validation Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Validation Loss Comparison (Local FLA)', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(var_names)))
    ax1.set_xticklabels(var_names, rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=min(val_losses), color='green', linestyle='--', alpha=0.5, label='Best')
    ax1.legend()
    
    # Plot 2: Training time comparison
    train_times = [results['results']['total_time'] for _, results in results_data]
    
    ax2.bar(range(len(var_names)), train_times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Experiment Variant', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(var_names)))
    ax2.set_xticklabels(var_names, rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save plot
    exp_dir = Path(__file__).parent
    plot_path = exp_dir / "experiment_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: {plot_path}")
    
    # Print best performer
    best_idx = val_losses.index(min(val_losses))
    best_var_id, best_results = results_data[best_idx]
    
    print(f"\n🏆 BEST PERFORMER:")
    print(f"   Experiment: {best_results.get('experiment_name', best_var_id)}")
    print(f"   Val Loss: {best_results['results']['best_val_loss']:.4f}")
    print(f"   Model: {best_results['config']['hidden_size']}d, {best_results['config']['num_layers']}L")
    print(f"   FLA Source: {best_results.get('fla_source', 'local_clone')}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare exp10 experiment variants')
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Specific variants to compare (default: all completed)')
    args = parser.parse_args()
    
    compare_experiments(args.variants)


if __name__ == "__main__":
    main()
