import torch
import time
import argparse
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add root to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp11_dynamic_routing.models import create_model
from experiments.exp11_dynamic_routing.config import get_config

def benchmark_model(model_path, config_name, device='cuda'):
    print(f"\nBenchmarking {config_name}...")
    
    # Load config and model
    config = get_config(config_name)
    model = create_model(config)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Generate dummy input
    batch_size = 1
    seq_len = 512
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    # Warmup
    print("  Warming up...")
    with torch.no_grad():
        for _ in range(5):
            model(input_ids)
            
    # Measure Latency
    print("  Measuring latency...")
    latencies = []
    num_runs = 50
    
    # Reset stats if dynamic
    if hasattr(model, 'get_routing_stats'):
        model.get_routing_stats(reset=True)
        
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(input_ids)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # ms
            
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    print(f"  Latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
    
    stats = {
        'latency_mean': avg_latency,
        'latency_std': std_latency,
        'routing': {}
    }
    
    # Get routing stats if applicable
    if hasattr(model, 'get_routing_stats'):
        routing = model.get_routing_stats(reset=False)
        print("  Routing Stats:")
        for k, v in routing.items():
            print(f"    {k}: {v:.1f}%")
        stats['routing'] = routing
        
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_ckpt', type=str, default='checkpoints/best_model.pt') # Adjust default as needed
    parser.add_argument('--dynamic_ckpt', type=str, default='checkpoints_dynamic/best_model.pt')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = {}
    
    # Benchmark Baseline
    if Path(args.baseline_ckpt).exists():
        results['baseline'] = benchmark_model(args.baseline_ckpt, 'baseline', device)
    else:
        print(f"Baseline checkpoint not found at {args.baseline_ckpt}")
        
    # Benchmark Dynamic
    if Path(args.dynamic_ckpt).exists():
        results['dynamic'] = benchmark_model(args.dynamic_ckpt, 'dynamic', device)
    else:
        print(f"Dynamic checkpoint not found at {args.dynamic_ckpt}")
        
    # Compare
    if 'baseline' in results and 'dynamic' in results:
        b_lat = results['baseline']['latency_mean']
        d_lat = results['dynamic']['latency_mean']
        diff = (d_lat - b_lat) / b_lat * 100
        
        print("\n" + "="*50)
        print("COMPARISON SUMMARY")
        print("="*50)
        print(f"Baseline Latency: {b_lat:.2f} ms")
        print(f"Dynamic Latency:  {d_lat:.2f} ms")
        print(f"Difference:       {diff:+.1f}%")
        
        if diff < 0:
            print("✓ Dynamic model is FASTER")
        else:
            print("⚠ Dynamic model is SLOWER (expected due to routing overhead if not sparse enough)")

if __name__ == "__main__":
    main()
