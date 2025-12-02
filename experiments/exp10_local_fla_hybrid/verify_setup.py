#!/usr/bin/env python3
"""
Verification script for Experiment 10
Tests that local FLA imports work correctly
"""

import sys
import os
from pathlib import Path

print("="*70)
print("EXPERIMENT 10 - VERIFICATION SCRIPT")
print("="*70)

# Setup paths
root_dir = Path(__file__).parent.parent.parent
fla_path = root_dir / 'flash-linear-attention'

# Add root to path for experiment imports
sys.path.insert(0, str(root_dir))

print(f"\n1. Checking directories...")
print(f"   Root: {root_dir}")
print(f"   FLA: {fla_path}")

if not fla_path.exists():
    print(f"\n❌ ERROR: FLA clone not found at {fla_path}")
    print(f"   Please clone the repository:")
    print(f"   git clone https://github.com/fla-org/flash-linear-attention")
    sys.exit(1)

print(f"   ✓ FLA directory exists")

# Add to path
if str(fla_path) not in sys.path:
    sys.path.insert(0, str(fla_path))

print(f"\n2. Testing FLA imports...")
try:
    import fla
    print(f"   ✓ FLA imported successfully")
    print(f"   FLA location: {fla.__file__}")
    
    # Verify it's the local clone
    if 'flash-linear-attention' in fla.__file__:
        print(f"   ✓ Using LOCAL FLA clone (correct)")
    else:
        print(f"   ⚠ Using pip-installed FLA (unexpected)")
        print(f"   This may happen if FLA is installed via pip")
    
    # Try importing models
    from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    print(f"   ✓ GatedDeltaNetConfig imported")
    print(f"   ✓ GatedDeltaNetForCausalLM imported")
    
except ImportError as e:
    print(f"\n❌ ERROR: Failed to import FLA")
    print(f"   {e}")
    sys.exit(1)

print(f"\n3. Testing experiment imports...")
try:
    from experiments.exp10_local_fla_hybrid.config import ExperimentConfig, get_h100_hybrid_sparse
    print(f"   ✓ Config imported")
    
    from experiments.exp10_local_fla_hybrid.models import GatedDeltaNetWrapper, count_parameters
    print(f"   ✓ Models imported")
    
except ImportError as e:
    print(f"\n❌ ERROR: Failed to import experiment modules")
    print(f"   {e}")
    sys.exit(1)

print(f"\n4. Testing model creation...")
try:
    config = get_h100_hybrid_sparse()
    config.num_hidden_layers = 2  # Small model for testing
    config.hidden_size = 128
    config.vocab_size = 1000
    config.max_steps = 10  # Small for testing
    
    print(f"   Creating test model...")
    model = GatedDeltaNetWrapper(config)
    param_count = sum(p.numel() for p in model.parameters())
    
    print(f"   ✓ Model created successfully")
    print(f"   Parameters: {param_count:,}")
    
    # Verify it's a hybrid model
    if config.attn_config:
        print(f"   ✓ Hybrid model with attention on layers: {config.attn_config['layers']}")
    
except Exception as e:
    print(f"\n❌ ERROR: Failed to create model")
    print(f"   {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n5. Checking file structure...")
exp_dir = Path(__file__).parent
required_files = [
    '__init__.py',
    'README.md',
    'COMPARISON.md',
    'SUMMARY.md',
    'config.py',
    'models.py',
    'run_experiment.py',
    'inference.py',
    'compare_experiments.py',
    'requirements.txt',
]

all_exist = True
for filename in required_files:
    filepath = exp_dir / filename
    if filepath.exists():
        print(f"   ✓ {filename}")
    else:
        print(f"   ✗ {filename} (missing)")
        all_exist = False

if not all_exist:
    print(f"\n⚠ Some files are missing")
else:
    print(f"\n   ✓ All required files present")

print(f"\n{'='*70}")
print(f"✅ VERIFICATION COMPLETE - Experiment 10 is ready!")
print(f"{'='*70}")
print(f"\nNext steps:")
print(f"  1. Install flash-attn: pip install flash-attn --no-build-isolation")
print(f"  2. Run training: python run_experiment.py")
print(f"  3. Run inference: python inference.py")
print(f"  4. Compare results: python compare_experiments.py")
print(f"{'='*70}\n")
