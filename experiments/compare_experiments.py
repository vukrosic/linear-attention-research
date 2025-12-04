import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np

def load_history(results_dir):
    history_path = Path(results_dir) / 'val_history.json'
    if not history_path.exists():
        print(f"Warning: No history found at {history_path}")
        return None
    
    with open(history_path, 'r') as f:
        return json.load(f)

def plot_comparison(baseline_dir, dynamic_dir, output_dir='comparison_results'):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    baseline_hist = load_history(baseline_dir)
    dynamic_hist = load_history(dynamic_dir)
    
    if not baseline_hist or not dynamic_hist:
        return

    # Extract data
    b_steps = [x['step'] for x in baseline_hist]
    b_loss = [x['loss'] for x in baseline_hist]
    b_ppl = [x['perplexity'] for x in baseline_hist]
    
    d_steps = [x['step'] for x in dynamic_hist]
    d_loss = [x['loss'] for x in dynamic_hist]
    d_ppl = [x['perplexity'] for x in dynamic_hist]
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(b_steps, b_loss, label='Baseline (Static)', marker='o', linestyle='--')
    plt.plot(d_steps, d_loss, label='Dynamic Routing', marker='s')
    plt.xlabel('Steps')
    plt.ylabel('Validation Loss')
    plt.title('Training Convergence: Baseline vs Dynamic')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'loss_comparison.png')
    plt.close()
    
    # Plot Perplexity
    plt.figure(figsize=(10, 6))
    plt.plot(b_steps, b_ppl, label='Baseline (Static)', marker='o', linestyle='--')
    plt.plot(d_steps, d_ppl, label='Dynamic Routing', marker='s')
    plt.xlabel('Steps')
    plt.ylabel('Perplexity')
    plt.title('Perplexity: Baseline vs Dynamic')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'ppl_comparison.png')
    plt.close()
    
    print(f"Plots saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default='results_baseline')
    parser.add_argument('--dynamic', type=str, default='results_dynamic')
    args = parser.parse_args()
    
    plot_comparison(args.baseline, args.dynamic)
