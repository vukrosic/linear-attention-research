# Experiment 10: Hybrid DeltaNet Architecture (Local FLA Clone)

This experiment is similar to Experiment 7, but uses the **local cloned FLA repository** (`flash-linear-attention/fla`) instead of the pip-installed package.

## Key Differences from Experiment 7

**FLA Import Source:**
- **Exp 7:** Imports from pip-installed `fla` package
- **Exp 10:** Imports from local clone at `flash-linear-attention/fla`

This allows for:
- ✅ Easy debugging and modifications to FLA source code
- ✅ Testing unreleased FLA features or custom patches
- ✅ No dependency on pip installation

---

## Setup

### Prerequisites
1. **Flash Attention** (for transformer baseline)
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Verify local FLA clone exists:**
   ```bash
   ls -la flash-linear-attention/fla
   ```

### Installation
```bash
cd experiments/exp10_local_fla_hybrid
pip install -r requirements.txt  # Only installs flash-attn
```

---

## Usage

All commands are identical to Experiment 7:

### Train Default (H100 Hybrid Sparse 17%)
```bash
python run_experiment.py
```

### Train Specific Architecture
```bash
python run_experiment.py --experiment h100_deltanet          # Pure DeltaNet
python run_experiment.py --experiment h100_transformer       # Pure Transformer
python run_experiment.py --experiment h100_hybrid_sparse     # 17% attention [5,11]
python run_experiment.py --experiment h100_hybrid_alternating # 50% attention (every other)
python run_experiment.py --experiment h100_hybrid_late       # 33% attention (last 4 layers)
```

### Resume Training
```bash
python run_experiment.py --resume checkpoints_h100_deltanet/best_model.pt
python run_experiment.py --resume checkpoints_h100_deltanet/best_model.pt --extend-steps 5000
```

---

## Available Architectures

Same as Experiment 7:

### Pure Architectures
- **h100_deltanet (0%)**: Pure DeltaNet - O(n) complexity
- **h100_transformer (100%)**: Pure attention - O(n²) complexity

### Hybrid Architectures (DeltaNet + Attention Mix)
- **h100_hybrid_sparse (17%)**: 2/12 layers attention [5, 11] ⭐ **Best from Exp 7**
- **h100_hybrid_alternating (50%)**: 6/12 layers attention (every other)
- **h100_hybrid_late (33%)**: 4/12 layers attention [8, 9, 10, 11]

All other hybrid percentages (8%, 25%, 42%, 58%, 67%, 75%, 83%, 92%) are also available via `config.py`.

---

## Model Configuration (H100)

- **Architecture**: 768d × 12L × 12H (~188M-302M params)
- **Sequence Length**: 1024 tokens
- **Batch Size**: 48 (49,152 tokens/step)
- **Training**: 1000 steps default
- **Learning Rates**: 
  - DeltaNet: 1e-3
  - Hybrids/Transformer: 2e-3

---

## Technical Details

### Import Mechanism

**models.py:**
```python
# Add flash-linear-attention to path
fla_path = os.path.join(root_dir, 'flash-linear-attention')
sys.path.insert(0, fla_path)

# Import from local clone
from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
```

**run_experiment.py:**
```python
# Add flash-linear-attention to path
fla_path = os.path.join(root_dir, 'flash-linear-attention')
sys.path.insert(0, fla_path)
```

The import path modification ensures that Python prioritizes the local clone over any pip-installed version.

---

## Expected Results

Since this uses the same FLA code (just from a different source), results should match Experiment 7:

- **Winner**: Hybrid Sparse 17% (val_loss ~4.055)
- **Pure DeltaNet**: val_loss ~4.396
- **Pure Transformer**: val_loss ~5.146

---

## Debugging & Development

### Modifying FLA Source
1. Make changes in `flash-linear-attention/fla/`
2. No need to reinstall - changes are immediately available
3. Run experiment: `python run_experiment.py`

### Verify Local Import
```bash
python -c "import sys; sys.path.insert(0, 'flash-linear-attention'); import fla; print(fla.__file__)"
```

Should print: `flash-linear-attention/fla/__init__.py`

---

## Reference

For architecture details and research findings, see:
- **Experiment 7 README**: `../exp7_hybrid_deltanet_ablation/README.md`
- **Video Overview**: [Watch on YouTube](https://www.youtube.com/watch?v=tf3ESMqDOTY)

---

## Files

- `config.py` - Model configurations (same as Exp 7)
- `models.py` - Model wrapper with **local FLA imports**
- `run_experiment.py` - Training script with **local FLA imports**
- `requirements.txt` - Dependencies (flash-attn only, no FLA)
- `README.md` - This file
